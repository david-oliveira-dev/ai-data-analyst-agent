"""Orquestrador do AI Data Analyst Agent.

Junta as peças num fluxo único: ingestão → EDA (perfil) → modelagem opcional →
narrativa executiva (LLM/fallback) → gráficos → PDF. É o ponto de entrada que a
API e o dashboard usam, para que a lógica de negócio fique fora deles.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.analysis import charts as charts_mod
from src.analysis.profiler import profile_dataframe
from src.data.ingest import load_csv
from src.llm.narrator import Narrator
from src.modeling.auto_model import run_automodel
from src.report.pdf_report import build_pdf


@dataclass
class AnalysisResult:
    profile: dict
    narrative: str
    model_analysis: dict | None = None
    charts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Versão serializável (sem os bytes dos gráficos) para respostas JSON."""
        return {
            "profile": self.profile,
            "narrative": self.narrative,
            "model_analysis": self.model_analysis,
            "charts_available": list(self.charts.keys()),
        }


def analyze(
    source: str | bytes,
    target: str | None = None,
    use_llm: bool | None = None,
    make_charts: bool = True,
) -> AnalysisResult:
    """Executa a análise completa de um CSV e devolve o resultado estruturado."""
    df = load_csv(source)
    profile = profile_dataframe(df)

    model_analysis = None
    if target:
        model_analysis = run_automodel(df, target)

    narrative = Narrator(use_llm=use_llm).narrate(profile, model_analysis)
    charts = charts_mod.build_all(df) if make_charts else {}
    return AnalysisResult(profile=profile, narrative=narrative,
                          model_analysis=model_analysis, charts=charts)


def analyze_dataframe(df: pd.DataFrame, target: str | None = None,
                      use_llm: bool | None = None) -> AnalysisResult:
    """Mesma análise a partir de um DataFrame já carregado (usado no dashboard)."""
    profile = profile_dataframe(df)
    model_analysis = run_automodel(df, target) if target else None
    narrative = Narrator(use_llm=use_llm).narrate(profile, model_analysis)
    return AnalysisResult(profile=profile, narrative=narrative,
                          model_analysis=model_analysis, charts=charts_mod.build_all(df))


def report_pdf(result: AnalysisResult, title: str = "Relatório de Análise de Dados") -> bytes:
    """Gera o PDF do resultado da análise."""
    return build_pdf(result.profile, result.narrative, result.charts, title=title)
