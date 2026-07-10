"""Testes da modelagem automática."""
from __future__ import annotations

from src.modeling.auto_model import detect_task, run_automodel


def test_detect_task_classification(sample_df):
    assert detect_task(sample_df["churn"]) == "classification"
    assert detect_task(sample_df["cidade"]) == "classification"


def test_detect_task_regression(sample_df):
    assert detect_task(sample_df["renda"]) == "regression"


def test_classification_runs_and_ranks(sample_df):
    res = run_automodel(sample_df, target="churn")
    assert res["task"] == "classification"
    assert res["best_model"] in {"LogisticRegression", "RandomForest"}
    # O alvo foi construído a partir de score/renda → sinal aprendível.
    top = res["leaderboard"][0]
    assert top.get("roc_auc", top.get("f1", 0)) > 0.6


def test_regression_reports_r2(sample_df):
    res = run_automodel(sample_df, target="renda")
    assert res["task"] == "regression"
    assert res["metric"] == "r2"
    assert "r2" in res["leaderboard"][0]


def test_feature_importance_maps_to_original_columns(sample_df):
    res = run_automodel(sample_df, target="churn")
    feats = {f["feature"] for f in res["feature_importance"]}
    # 'cidade' (one-hot) deve reagregar de volta ao nome original.
    assert feats.issubset(set(sample_df.columns) - {"churn"})


def test_missing_target_raises(sample_df):
    import pytest
    with pytest.raises(ValueError):
        run_automodel(sample_df, target="inexistente")
