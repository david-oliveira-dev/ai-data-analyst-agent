"""Testes do narrador: fallback determinístico e uso de cliente injetado."""
from __future__ import annotations

from src.analysis.profiler import profile_dataframe
from src.llm.narrator import Narrator, fallback_summary


def test_fallback_mentions_shape_and_quality(sample_df):
    profile = profile_dataframe(sample_df)
    text = fallback_summary(profile)
    assert "400" in text
    assert "renda" in text            # coluna com faltantes é citada
    assert "Recomendações" in text


def test_narrator_uses_fallback_when_disabled(sample_df):
    # AIDA_DISABLE_LLM=1 vem da fixture autouse.
    profile = profile_dataframe(sample_df)
    text = Narrator().narrate(profile)
    assert "Relatório executivo" in text


class _StubMessages:
    def __init__(self, text):
        self._text = text

    def create(self, **kwargs):
        block = type("B", (), {"type": "text", "text": self._text})()
        return type("R", (), {"content": [block]})()


class _StubClient:
    def __init__(self, text):
        self.messages = _StubMessages(text)


def test_narrator_uses_injected_client(sample_df):
    profile = profile_dataframe(sample_df)
    narrator = Narrator(use_llm=True, client=_StubClient("NARRATIVA DO LLM"))
    assert narrator.narrate(profile) == "NARRATIVA DO LLM"


def test_narrator_falls_back_on_client_error(sample_df):
    class Boom:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("sem rede")

    profile = profile_dataframe(sample_df)
    text = Narrator(use_llm=True, client=Boom()).narrate(profile)
    # Falhou a chamada → caiu no fallback determinístico.
    assert "Relatório executivo" in text


def test_narrative_includes_modeling_when_present(sample_df):
    from src.modeling.auto_model import run_automodel
    profile = profile_dataframe(sample_df)
    model = run_automodel(sample_df, target="churn")
    text = fallback_summary(profile, model)
    assert "Modelagem preditiva" in text
    assert "churn" in text
