"""Gera o exemplo de saída mostrado no README.

Monta um CSV com problemas de qualidade de propósito (faltantes, outliers,
correlação forte entre duas colunas), roda o agente com o **fallback
determinístico** — sem chave de API, exatamente como no CI — e salva:

- `reports/figures/exemplo_graficos.png` — os gráficos que o agente produz;
- `reports/exemplo_relatorio.pdf` — o relatório executivo completo;
- imprime a narrativa gerada, que é colada no README.

Uso:
    python examples/gerar_exemplo.py
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

os.environ["AIDA_DISABLE_LLM"] = "1"  # usa o fallback: reproduzível e sem custo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.agent import analyze_dataframe, report_pdf  # noqa: E402

FIGURAS = ROOT / "reports" / "figures"
RELATORIO = ROOT / "reports" / "exemplo_relatorio.pdf"


def montar_csv_exemplo(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """Base de assinaturas com defeitos plantados, para o agente ter o que achar."""
    rng = np.random.default_rng(seed)

    meses = rng.integers(1, 60, n)
    mensalidade = rng.normal(89, 25, n).round(2).clip(20, 200)

    df = pd.DataFrame({
        "meses_de_casa": meses,
        "mensalidade": mensalidade,
        # Correlacionada com as duas acima: o agente deve sinalizar a redundância.
        "total_gasto": (meses * mensalidade * rng.normal(1, 0.05, n)).round(2),
        "chamados_suporte": rng.poisson(1.4, n),
        "plano": rng.choice(["basico", "padrao", "premium"], n, p=[0.5, 0.3, 0.2]),
        "regiao": rng.choice(["sudeste", "sul", "nordeste", "norte"], n),
    })

    # O alvo tem sinal de verdade (cliente novo e com muitos chamados cancela
    # mais), senão o demo mostraria o agente reportando um modelo aleatório.
    logit = -0.6 - 0.06 * df["meses_de_casa"] + 0.55 * df["chamados_suporte"]
    df["cancelou"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    # Faltantes concentrados numa coluna só.
    df.loc[rng.choice(n, size=int(0.12 * n), replace=False), "mensalidade"] = np.nan
    # Outliers de cobrança.
    df.loc[rng.choice(n, size=8, replace=False), "total_gasto"] *= 12

    return df


def salvar_graficos(charts: dict[str, bytes]) -> Path:
    """Junta os PNGs devolvidos pelo agente numa figura única para o README."""
    itens = [(nome, png) for nome, png in charts.items() if png]
    fig, axs = plt.subplots(1, len(itens), figsize=(6.5 * len(itens), 4.6))
    axs = np.atleast_1d(axs)

    for ax, (nome, png) in zip(axs, itens):
        ax.imshow(mpimg.imread(io.BytesIO(png), format="png"))
        ax.set_title(nome, fontsize=11)
        ax.axis("off")

    fig.suptitle("Gráficos gerados automaticamente pelo agente", fontsize=13)
    fig.tight_layout()

    FIGURAS.mkdir(parents=True, exist_ok=True)
    destino = FIGURAS / "exemplo_graficos.png"
    fig.savefig(destino, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return destino


def main() -> None:
    df = montar_csv_exemplo()
    resultado = analyze_dataframe(df, target="cancelou")

    figura = salvar_graficos(resultado.charts)
    RELATORIO.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO.write_bytes(report_pdf(resultado, title="Relatório de Análise — base de exemplo"))

    print(f"figura : {figura}")
    print(f"pdf    : {RELATORIO} ({RELATORIO.stat().st_size // 1024} KB)")
    print("\n--- narrativa gerada (fallback determinístico) ---\n")
    print(resultado.narrative)


if __name__ == "__main__":
    main()
