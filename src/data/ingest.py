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


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza não destrutiva: nomes de coluna e tipos numéricos óbvios."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        if df[col].dtype == "object":
            stripped = df[col].astype(str).str.strip()
            # Tenta converter "1.234,56" (padrão BR) ou "1234.56" em número.
            as_num = pd.to_numeric(
                stripped.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                errors="coerce",
            )
            # Só adota a conversão se a grande maioria virou número de fato.
            if stripped.notna().sum() and as_num.notna().mean() > 0.9:
                df[col] = as_num
            else:
                df[col] = stripped.replace({"": None, "nan": None})
    return df
