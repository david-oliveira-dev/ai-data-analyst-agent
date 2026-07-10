"""Geração de gráficos da EDA como PNG (bytes), prontos para PDF e dashboard.

Usa o backend headless "Agg" (sem display) — essencial para rodar em servidor,
container e CI. Cada função devolve os bytes de um PNG; quem chama decide se
embute no PDF, mostra no Streamlit ou salva em disco.
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")            # backend sem interface gráfica
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402


def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def histograms(df: pd.DataFrame, max_cols: int = 6) -> bytes | None:
    """Grade de histogramas das colunas numéricas (até `max_cols`)."""
    num = df.select_dtypes(include=np.number)
    cols = list(num.columns)[:max_cols]
    if not cols:
        return None
    n = len(cols)
    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, min(3, n), figsize=(4 * min(3, n), 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        num[col].dropna().plot(kind="hist", bins=30, ax=ax, color="#4C72B0")
        ax.set_title(col)
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Distribuição das variáveis numéricas")
    return _fig_to_png(fig)


def correlation_heatmap(df: pd.DataFrame) -> bytes | None:
    """Mapa de calor de correlação entre as colunas numéricas."""
    num = df.select_dtypes(include=np.number)
    if num.shape[1] < 2:
        return None
    corr = num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(1.1 * len(corr) + 2, 1.1 * len(corr) + 1))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlação (Pearson)")
    return _fig_to_png(fig)


def missing_bar(df: pd.DataFrame) -> bytes | None:
    """Barras de percentual de valores faltantes por coluna (só as com falta)."""
    miss = (df.isna().mean() * 100)
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, max(2, 0.4 * len(miss))))
    miss.plot(kind="barh", ax=ax, color="#C44E52")
    ax.set_xlabel("% faltante")
    ax.set_title("Valores faltantes por coluna")
    ax.invert_yaxis()
    return _fig_to_png(fig)


def build_all(df: pd.DataFrame) -> dict[str, bytes]:
    """Gera todos os gráficos disponíveis e devolve {nome: png_bytes}."""
    charts = {
        "histograms": histograms(df),
        "correlation": correlation_heatmap(df),
        "missing": missing_bar(df),
    }
    return {name: png for name, png in charts.items() if png is not None}
