"""EDA automática: transforma um DataFrame num *perfil* estruturado (dict).

O perfil é o insumo central do sistema — alimenta o relatório PDF, o dashboard e
o prompt do LLM. É deliberadamente serializável (só tipos nativos) para poder ser
logado, cacheado e enviado à API sem surpresas.

Cobre: dimensões, tipos, faltantes, estatísticas numéricas, cardinalidade
categórica, correlações fortes e detecção simples de outliers (regra do IQR).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Acima disto uma coluna numérica é tratada como contínua para outliers/correlação.
_MIN_NUMERIC_FOR_STATS = 1


def _missing(df: pd.DataFrame) -> dict:
    miss = df.isna().sum()
    return {
        col: {"count": int(n), "pct": round(float(n) / len(df) * 100, 2)}
        for col, n in miss.items() if n > 0
    }


def _numeric_summary(df: pd.DataFrame) -> dict:
    num = df.select_dtypes(include=np.number)
    out = {}
    for col in num.columns:
        s = num[col].dropna()
        if len(s) < _MIN_NUMERIC_FOR_STATS:
            continue
        out[col] = {
            "count": int(s.count()),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std(ddof=0)), 4),
            "min": round(float(s.min()), 4),
            "p25": round(float(s.quantile(0.25)), 4),
            "median": round(float(s.median()), 4),
            "p75": round(float(s.quantile(0.75)), 4),
            "max": round(float(s.max()), 4),
        }
    return out


def _categorical_summary(df: pd.DataFrame, top: int = 5) -> dict:
    cat = df.select_dtypes(exclude=np.number)
    out = {}
    for col in cat.columns:
        s = df[col].dropna()
        counts = s.value_counts().head(top)
        out[col] = {
            "unique": int(s.nunique()),
            "top_values": {str(k): int(v) for k, v in counts.items()},
        }
    return out


def _outliers_iqr(df: pd.DataFrame) -> dict:
    """Conta outliers por coluna numérica pela regra 1.5×IQR."""
    num = df.select_dtypes(include=np.number)
    out = {}
    for col in num.columns:
        s = num[col].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((s < lo) | (s > hi)).sum())
        if n:
            out[col] = {"count": n, "pct": round(n / len(s) * 100, 2)}
    return out


def _strong_correlations(df: pd.DataFrame, threshold: float = 0.6) -> list[dict]:
    """Pares de colunas numéricas com |correlação| ≥ threshold."""
    num = df.select_dtypes(include=np.number)
    if num.shape[1] < 2:
        return []
    corr = num.corr(numeric_only=True)
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) >= threshold:
                pairs.append({"a": cols[i], "b": cols[j], "corr": round(float(r), 3)})
    return sorted(pairs, key=lambda p: abs(p["corr"]), reverse=True)


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Constrói o perfil completo do dataset."""
    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    return {
        "shape": {"rows": int(len(df)), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": dtypes,
        "numeric_columns": list(df.select_dtypes(include=np.number).columns),
        "categorical_columns": list(df.select_dtypes(exclude=np.number).columns),
        "missing": _missing(df),
        "numeric_summary": _numeric_summary(df),
        "categorical_summary": _categorical_summary(df),
        "outliers": _outliers_iqr(df),
        "correlations": _strong_correlations(df),
        "duplicated_rows": int(df.duplicated().sum()),
    }
