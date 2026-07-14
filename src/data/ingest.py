"""Ingestão de CSV: lê de caminho ou de bytes, valida e faz uma limpeza leve.

O agente recebe CSVs "do mundo real" — separador incerto, colunas numéricas com
vírgula decimal, espaços sobrando. Aqui normalizamos o suficiente para a EDA
funcionar sem alterar a semântica dos dados (não imputamos nem removemos linhas).
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

MAX_ROWS = 1_000_000        # trava de segurança contra arquivos gigantes
_CANDIDATES = [",", ";", "\t", "|"]
_NO_DELIM = "\x01"          # sentinela: força leitura em coluna única


def _detect_sep(header_line: str) -> str:
    """Detecta o separador a partir da linha de cabeçalho.

    Restringe aos separadores comuns (`, ; tab |`) — nunca `.` ou `,` decimais —
    e, se nenhum aparece no cabeçalho, trata o arquivo como coluna única.
    """
    counts = {d: header_line.count(d) for d in _CANDIDATES}
    best = max(counts, key=lambda d: counts[d])
    return best if counts[best] > 0 else _NO_DELIM


def _to_text(source: str | Path | bytes) -> str:
    if isinstance(source, bytes):
        return source.decode("utf-8", errors="replace")
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def load_csv(source: str | Path | bytes, sep: str | None = None) -> pd.DataFrame:
    """Carrega um CSV de um caminho, de bytes ou de um buffer.

    `sep=None` detecta o separador de forma robusta (sem confundir com decimais).
    """
    text = _to_text(source)
    first_line = text.split("\n", 1)[0].strip()
    if not first_line:
        raise ValueError("CSV vazio ou sem colunas legíveis.")

    delimiter = sep or _detect_sep(first_line)
    try:
        df = pd.read_csv(io.StringIO(text), sep=delimiter, engine="c")
    except Exception as exc:                       # parsing falhou
        raise ValueError(f"CSV ilegível: {exc}") from exc

    if df.empty or df.shape[1] == 0:
        raise ValueError("CSV vazio ou sem colunas legíveis.")
    if len(df) > MAX_ROWS:
        df = df.head(MAX_ROWS)
    return clean(df)


# Um valor é "só separador de milhar" quando os grupos batem certinho:
# primeiro grupo de 1-3 dígitos sem zero à esquerda, seguido de grupos de 3.
# Casa "1.234" e "1.234.567"; não casa "1234.56" (cauda de 2) nem "0.500".
_THOUSANDS_ONLY = {
    ".": r"[+-]?[1-9]\d{0,2}(?:\.\d{3})+",
    ",": r"[+-]?[1-9]\d{0,2}(?:,\d{3})+",
}


def _sniff_decimal_sep(s: pd.Series) -> str | None:
    """Descobre o separador decimal da coluna: ',' (BR), '.' (US) ou None.

    None significa "a coluna não tem casas decimais" — ou não há separador
    algum, ou o que existe é separador de milhar.

    A regra forte: quando '.' e ',' aparecem no MESMO número, o **último** é o
    decimal — `1.234,56` -> ',' e `1,234.56` -> '.'. É o único sinal que
    desambigua BR de US sem depender de locale.

    Com um separador só, ele é de milhar apenas se TODOS os valores tiverem
    formato de agrupamento (`1.234`); caso contrário é decimal (`1234.56`).
    """
    has = {sep: s.str.contains(sep, regex=False, na=False) for sep in (".", ",")}

    both = s[has["."] & has[","]]
    if len(both):
        comma_is_last = both.str.rfind(",") > both.str.rfind(".")
        return "," if comma_is_last.mean() > 0.5 else "."

    for sep in (".", ","):
        vals = s[has[sep]]
        if not len(vals):
            continue
        if vals.str.fullmatch(_THOUSANDS_ONLY[sep]).all():
            return None
        return sep
    return None


def _coerce_numeric(col: pd.Series) -> pd.Series:
    """Tenta converter uma coluna de texto em número, detectando BR ou US.

    Detecta o formato pela coluna inteira (via `_sniff_decimal_sep`) e então
    normaliza para o padrão que o `to_numeric` entende: remove o separador de
    milhar e deixa '.' como decimal.
    """
    s = col.astype(str).str.strip()
    decimal = _sniff_decimal_sep(s)

    if decimal == ",":                                        # BR: 1.234,56
        candidate = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    elif decimal == ".":                                      # US: 1,234.56
        candidate = s.str.replace(",", "", regex=False)
    else:                                                     # sem decimais: 1.234 / 1,234
        candidate = s.str.replace(".", "", regex=False).str.replace(",", "", regex=False)
    return pd.to_numeric(candidate, errors="coerce")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza não destrutiva: nomes de coluna e tipos numéricos óbvios."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        # Cobre object, string e category — o dtype de texto varia entre versões
        # do pandas, então testamos "não numérico" em vez de "== object".
        if not pd.api.types.is_numeric_dtype(df[col]):
            num = _coerce_numeric(df[col])
            # Só adota a conversão se a grande maioria virou número de fato.
            if len(num) and num.notna().mean() > 0.9:
                df[col] = num
            else:
                df[col] = df[col].astype(str).str.strip().replace({"": None, "nan": None})
    return df
