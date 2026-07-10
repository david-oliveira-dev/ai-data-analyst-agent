"""API FastAPI do AI Data Analyst Agent.

Endpoints:
- `GET  /health`   — saúde do serviço e se o LLM está habilitado.
- `POST /analyze`  — recebe um CSV (multipart) e devolve a análise em JSON
                     (perfil + narrativa + modelagem opcional).
- `POST /report`   — recebe um CSV e devolve o relatório executivo em PDF.

O parâmetro opcional `target` (query) liga a modelagem preditiva sobre aquela
coluna. `use_llm` permite forçar/desligar a narrativa por LLM na requisição.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from src.agent import analyze, report_pdf

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="AI Data Analyst Agent",
    description="Analisa CSVs e produz relatórios executivos (EDA + LLM + PDF).",
    version="1.0.0",
)

MAX_BYTES = 50 * 1024 * 1024      # limite de upload: 50 MB


async def _read_csv(file: UploadFile) -> bytes:
    if file.filename and not file.filename.lower().endswith((".csv", ".txt")):
        raise HTTPException(status_code=415, detail="Envie um arquivo .csv")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Arquivo maior que 50 MB.")
    return content


def _run(content: bytes, target: str | None, use_llm: bool | None):
    try:
        return analyze(content, target=target, use_llm=use_llm)
    except ValueError as exc:                 # CSV inválido / alvo inexistente
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    import os

    llm_on = os.getenv("AIDA_DISABLE_LLM", "").lower() not in ("1", "true", "yes")
    return {"status": "ok", "llm_enabled": llm_on}


@app.post("/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    target: str | None = Query(default=None, description="Coluna-alvo p/ modelagem"),
    use_llm: bool | None = Query(default=None, description="Forçar/desligar narrativa por LLM"),
) -> dict:
    content = await _read_csv(file)
    result = _run(content, target, use_llm)
    return result.to_dict()


@app.post("/report")
async def report_endpoint(
    file: UploadFile = File(...),
    target: str | None = Query(default=None),
    use_llm: bool | None = Query(default=None),
) -> Response:
    content = await _read_csv(file)
    result = _run(content, target, use_llm)
    pdf = report_pdf(result)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="relatorio.pdf"'},
    )
