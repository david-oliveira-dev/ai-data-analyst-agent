"""Testes da geração de gráficos (devem sair PNGs válidos)."""
from __future__ import annotations

from src.analysis import charts as charts_mod


def _is_png(b: bytes) -> bool:
    return b[:8] == b"\x89PNG\r\n\x1a\n"


def test_histograms_returns_png(sample_df):
    png = charts_mod.histograms(sample_df)
    assert png and _is_png(png)


def test_correlation_heatmap_returns_png(sample_df):
    png = charts_mod.correlation_heatmap(sample_df)
    assert png and _is_png(png)


def test_missing_bar_returns_png(sample_df):
    png = charts_mod.missing_bar(sample_df)   # 'renda' tem faltantes
    assert png and _is_png(png)


def test_build_all_keys(sample_df):
    charts = charts_mod.build_all(sample_df)
    assert {"histograms", "correlation", "missing"}.issubset(charts)
    assert all(_is_png(v) for v in charts.values())


def test_no_numeric_no_histogram():
    import pandas as pd
    df = pd.DataFrame({"cidade": ["SP", "RJ", "SP"]})
    assert charts_mod.histograms(df) is None
