"""Fixtures compartilhadas: um dataset sintético realista e seu CSV em bytes.

O ambiente de teste força o fallback determinístico do narrador via a env var
`AIDA_DISABLE_LLM`, para nenhum teste chamar a API da Claude.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _disable_llm(monkeypatch):
    """Nenhum teste deve chamar a API real — sempre usa o fallback."""
    monkeypatch.setenv("AIDA_DISABLE_LLM", "1")


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """Dataset com numéricas correlacionadas, categórica, faltantes e alvo binário."""
    rng = np.random.default_rng(42)
    n = 400
    idade = rng.integers(18, 70, size=n)
    renda = idade * 120 + rng.normal(0, 500, size=n)          # correlaciona com idade
    score = rng.normal(600, 80, size=n)
    cidade = rng.choice(["SP", "RJ", "BH", "POA"], size=n, p=[0.4, 0.3, 0.2, 0.1])
    # Alvo binário influenciado por score e renda.
    prob = 1 / (1 + np.exp(-(score - 600) / 60 - (renda - renda.mean()) / renda.std()))
    churn = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "idade": idade, "renda": renda.round(2), "score": score.round(1),
        "cidade": cidade, "churn": churn,
    })
    # Injeta alguns faltantes em 'renda'.
    df.loc[rng.choice(n, size=20, replace=False), "renda"] = np.nan
    return df


@pytest.fixture()
def sample_csv_bytes(sample_df) -> bytes:
    return sample_df.to_csv(index=False).encode("utf-8")
