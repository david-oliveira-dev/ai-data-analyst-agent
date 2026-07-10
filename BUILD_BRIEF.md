# BUILD_BRIEF — AI Data Analyst Agent

Roteiro incremental. Cada etapa fecha com testes verdes + commit.

## Objetivo
Agente que recebe um CSV e devolve um relatório executivo automático: EDA,
gráficos, narrativa por LLM e PDF — para analistas e gestores.

## Etapas
1. **Ingestão** — leitura robusta de CSV (separador inferido, decimal BR),
   limpeza não destrutiva. → `src/data/ingest.py`
2. **EDA automática** — perfil estruturado (dimensões, tipos, faltantes,
   estatísticas, cardinalidade, outliers IQR, correlações). → `src/analysis/profiler.py`
3. **Gráficos** — histogramas, heatmap de correlação e faltantes como PNG
   (backend headless). → `src/analysis/charts.py`
4. **Modelagem opcional** — dado um alvo, detecta classificação/regressão,
   treina baselines e compara + importância. → `src/modeling/auto_model.py`
5. **Narrativa por LLM** — Claude (Opus 4.8) escreve o relatório executivo, com
   fallback determinístico sem chave. → `src/llm/narrator.py`
6. **PDF** — compõe o relatório (narrativa + tabelas + gráficos). → `src/report/pdf_report.py`
7. **Orquestrador** — junta o fluxo ponta a ponta. → `src/agent.py`
8. **API + dashboard** — FastAPI (`/analyze`, `/report`) e Streamlit. → `app/`
9. **Empacotamento** — Docker, docker-compose, CI.
10. **Documentação** — README, arquitetura, relatório técnico.

## Definição de pronto
- `pytest -q` verde local e no CI (com `AIDA_DISABLE_LLM=1`).
- API sobe e responde `/health`, `/analyze` e `/report` (PDF).
- README com contexto de negócio, arquitetura, uso e trade-offs.
