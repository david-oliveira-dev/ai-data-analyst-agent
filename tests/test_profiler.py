"""Testes da EDA automática (profiler)."""
from __future__ import annotations

from src.analysis.profiler import profile_dataframe


def test_profile_basic_shape(sample_df):
    p = profile_dataframe(sample_df)
    assert p["shape"] == {"rows": 400, "columns": 5}
    assert set(p["numeric_columns"]) >= {"idade", "renda", "score"}
    assert "cidade" in p["categorical_columns"]


def test_detects_missing(sample_df):
    p = profile_dataframe(sample_df)
    assert "renda" in p["missing"]
    assert p["missing"]["renda"]["count"] == 20


def test_numeric_summary_has_quartiles(sample_df):
    p = profile_dataframe(sample_df)
    stats = p["numeric_summary"]["idade"]
    assert stats["min"] <= stats["median"] <= stats["max"]
    assert {"mean", "std", "p25", "p75"}.issubset(stats)


def test_detects_correlation(sample_df):
    # idade e renda foram construídas correlacionadas.
    p = profile_dataframe(sample_df)
    pares = {(c["a"], c["b"]) for c in p["correlations"]}
    assert ("idade", "renda") in pares or ("renda", "idade") in pares


def test_profile_is_json_serializable(sample_df):
    import json
    p = profile_dataframe(sample_df)
    json.dumps(p)          # não pode conter tipos numpy/pandas
