"""Testes de integração da API FastAPI (com o LLM desligado via env)."""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    return TestClient(app)


def _upload(csv_bytes: bytes):
    return {"file": ("dados.csv", io.BytesIO(csv_bytes), "text/csv")}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["llm_enabled"] is False        # AIDA_DISABLE_LLM=1 na fixture


def test_analyze_returns_profile_and_narrative(client, sample_csv_bytes):
    r = client.post("/analyze", files=_upload(sample_csv_bytes))
    assert r.status_code == 200
    body = r.json()
    assert body["profile"]["shape"]["rows"] == 400
    assert "narrative" in body and body["narrative"]
    assert set(body["charts_available"]) <= {"histograms", "correlation", "missing"}


def test_analyze_with_target_runs_modeling(client, sample_csv_bytes):
    r = client.post("/analyze?target=churn", files=_upload(sample_csv_bytes))
    assert r.status_code == 200
    model = r.json()["model_analysis"]
    assert model is not None
    assert model["task"] == "classification"


def test_analyze_invalid_target_returns_422(client, sample_csv_bytes):
    r = client.post("/analyze?target=inexistente", files=_upload(sample_csv_bytes))
    assert r.status_code == 422


def test_report_returns_pdf(client, sample_csv_bytes):
    r = client.post("/report", files=_upload(sample_csv_bytes))
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-"


def test_rejects_non_csv(client):
    r = client.post("/report", files={"file": ("x.png", io.BytesIO(b"\x89PNG"), "image/png")})
    assert r.status_code == 415


def test_rejects_empty_file(client):
    r = client.post("/analyze", files={"file": ("x.csv", io.BytesIO(b""), "text/csv")})
    assert r.status_code == 400
