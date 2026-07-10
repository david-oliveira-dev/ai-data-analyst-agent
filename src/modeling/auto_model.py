"""Modelagem automática opcional: dado um alvo, treina e compara baselines.

Se o usuário indica uma coluna-alvo, o agente vai além da EDA descritiva: detecta
se o problema é **classificação** ou **regressão**, monta um pré-processamento
robusto (imputação + one-hot + escala), treina modelos de referência e devolve um
placar com a métrica apropriada e a importância das variáveis do vencedor.

É deliberadamente "baseline": o objetivo é dar um sinal rápido de "dá para
prever isto?" e quais variáveis pesam — não um modelo de produção.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def detect_task(y: pd.Series) -> str:
    """Heurística: poucos valores distintos ou não-numérico → classificação."""
    if y.dtype == object or str(y.dtype) == "category":
        return "classification"
    return "classification" if y.nunique() <= 10 else "regression"


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = list(X.select_dtypes(include=np.number).columns)
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=20)),
    ])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


def _models(task: str) -> dict:
    if task == "classification":
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        }
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }


def _score(task: str, model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    if task == "classification":
        out = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "f1": round(float(f1_score(y_test, pred, average="weighted", zero_division=0)), 4),
        }
        # ROC-AUC só faz sentido no caso binário com probabilidade.
        if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
            out["roc_auc"] = round(float(roc_auc_score(y_test, proba)), 4)
        return out
    return {"r2": round(float(r2_score(y_test, pred)), 4)}


def run_automodel(df: pd.DataFrame, target: str, seed: int = 42) -> dict:
    """Treina/avalia baselines para prever `target`. Devolve placar + importâncias."""
    if target not in df.columns:
        raise ValueError(f"Coluna-alvo '{target}' não existe no dataset.")

    data = df.dropna(subset=[target])
    X, y = data.drop(columns=[target]), data[target]
    task = detect_task(y)

    stratify = y if task == "classification" and y.nunique() > 1 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=stratify)

    pre = _preprocessor(X)
    leaderboard, fitted = [], {}
    for name, est in _models(task).items():
        pipe = Pipeline([("pre", pre), ("model", est)])
        pipe.fit(X_tr, y_tr)
        row = {"model": name, **_score(task, pipe, X_te, y_te)}
        leaderboard.append(row)
        fitted[name] = pipe

    key = "r2" if task == "regression" else ("roc_auc" if "roc_auc" in leaderboard[0] else "f1")
    leaderboard.sort(key=lambda r: r.get(key, 0), reverse=True)
    best = leaderboard[0]["model"]

    return {
        "task": task,
        "target": target,
        "metric": key,
        "leaderboard": leaderboard,
        "best_model": best,
        "feature_importance": _importance(fitted[best], X.columns),
    }


def _importance(pipe: Pipeline, original_cols) -> list[dict]:
    """Importância agregada de volta às colunas originais (top 10)."""
    model = pipe.named_steps["model"]
    pre = pipe.named_steps["pre"]
    try:
        names = pre.get_feature_names_out()
    except Exception:
        return []

    if hasattr(model, "feature_importances_"):
        weights = model.feature_importances_
    elif hasattr(model, "coef_"):
        weights = np.abs(np.ravel(model.coef_))
    else:
        return []

    # Reagrega o one-hot ('cat__cidade_SP') de volta à coluna original ('cidade').
    agg: dict[str, float] = {}
    for name, w in zip(names, weights):
        base = name.split("__", 1)[-1]
        for col in original_cols:
            if base == col or base.startswith(f"{col}_"):
                agg[col] = agg.get(col, 0.0) + float(w)
                break
    top = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return [{"feature": k, "importance": round(v, 4)} for k, v in top]
