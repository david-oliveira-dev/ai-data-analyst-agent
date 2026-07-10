"""Testes da geração de PDF."""
from __future__ import annotations

from src.agent import analyze_dataframe, report_pdf
from src.analysis import charts as charts_mod
from src.analysis.profiler import profile_dataframe
from src.llm.narrator import fallback_summary
from src.report.pdf_report import build_pdf


def _is_pdf(b: bytes) -> bool:
    return b[:5] == b"%PDF-"


def test_build_pdf_returns_pdf_bytes(sample_df):
    profile = profile_dataframe(sample_df)
    narrative = fallback_summary(profile)
    charts = charts_mod.build_all(sample_df)
    pdf = build_pdf(profile, narrative, charts)
    assert _is_pdf(pdf)
    assert len(pdf) > 1000            # PDF com conteúdo real


def test_build_pdf_without_charts(sample_df):
    profile = profile_dataframe(sample_df)
    pdf = build_pdf(profile, "# Título\n\nTexto do relatório.", charts=None)
    assert _is_pdf(pdf)


def test_report_pdf_from_result(sample_df):
    result = analyze_dataframe(sample_df, target="churn")
    pdf = report_pdf(result)
    assert _is_pdf(pdf)
