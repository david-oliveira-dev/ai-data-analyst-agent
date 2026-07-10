# CLAUDE.md — AI Data Analyst Agent

Contexto e convenções para qualquer sessão do Claude Code trabalhando neste repo.

## O que é
Projeto de portfólio (Cientista de Dados Pleno): um **agente de IA que analisa
CSVs e produz relatórios executivos** — EDA automática, gráficos, narrativa por
LLM (Claude) e PDF. Roteiro completo em **`BUILD_BRIEF.md`**.

## Como trabalhar aqui
- Construção **incremental**: uma etapa do BUILD_BRIEF por vez, com commit ao fim.
- Explique decisões técnicas nos commits e no README (é portfólio).
- **Commits sem a linha `Co-Authored-By`.**
- Não comitar dados enviados, PDFs gerados nem segredos. Chave por env var.

## Stack
Python 3.12, Pandas/NumPy, scikit-learn, matplotlib, **Anthropic SDK (Claude
Opus 4.8)**, reportlab (PDF), FastAPI, Streamlit, SQLAlchemy/PostgreSQL (opcional),
Docker, pytest.

## Integração com a Claude (LLM)
- Use o **SDK oficial `anthropic`** e o modelo **`claude-opus-4-8`** com
  `thinking={"type": "adaptive"}`. Credencial resolvida do ambiente
  (`ANTHROPIC_API_KEY` ou perfil `ant`).
- **O LLM é sempre opcional.** `src/llm/narrator.py` cai num **fallback
  determinístico** (resumo montado do perfil) quando: não há chave, `AIDA_DISABLE_LLM`
  está setado, ou a chamada falha. Isso mantém a app de pé e os testes/CI rodando
  **sem** credencial.
- Nos testes, o LLM é desligado via `AIDA_DISABLE_LLM=1` (fixture autouse) e o
  cliente pode ser injetado (`Narrator(client=...)`).

## Ambiente
- Use `venv`; o `pip` global da máquina do dono é bloqueado (PEP 668).
- Máquina **sem GPU** e com HD mecânico. Sem dependências pesadas (sem TF/torch).
- Gráficos usam o backend headless `Agg` (roda em servidor/CI).

## Comandos
```bash
uvicorn app.main:app --reload        # API (/docs)
streamlit run app/dashboard.py       # dashboard
pytest -q                            # testes (LLM desligado)
```
