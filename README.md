# AI Data Analyst Agent

Agente de IA que **recebe um CSV e devolve um relatório executivo automático**:
EDA (análise exploratória) automática, gráficos, uma **narrativa em linguagem
natural escrita pela Claude** e um **PDF** pronto para compartilhar — mais uma
etapa opcional de **modelagem preditiva** quando você indica a coluna-alvo.
Projeto de portfólio para **Cientista de Dados Pleno**.

![CI](https://github.com/david-oliveira-dev/ai-data-analyst-agent/actions/workflows/ci.yml/badge.svg)

## Contexto de negócio
Analistas gastam horas repetindo o mesmo ritual em cada nova base: abrir o CSV,
checar tipos e faltantes, plotar distribuições, procurar correlações e então
**escrever o resumo** que o gestor de fato vai ler. Este agente automatiza esse
ritual ponta a ponta: sobe o arquivo, e em segundos há um relatório executivo com
os números, os problemas de qualidade, os padrões relevantes e recomendações — no
navegador ou em PDF. Público-alvo: **analistas e gestores**.

## Stack
Python 3.12 · Pandas/NumPy · scikit-learn · matplotlib · **Anthropic SDK
(Claude Opus 4.8)** · reportlab (PDF) · FastAPI · Streamlit · SQLAlchemy/PostgreSQL
· Docker · pytest

## Arquitetura
```
 CSV ─▶ Ingestão ─▶ EDA automática (perfil) ─┬─▶ Gráficos (PNG)
                                             ├─▶ Modelagem opcional (alvo)
                                             ▼
                                    Narrador (Claude Opus 4.8)
                                    └─ fallback determinístico sem chave
                                             ▼
                        Relatório executivo ─▶ PDF · API · Dashboard
```
Detalhes, fluxograma e trade-offs em [`reports/ARCHITECTURE.md`](reports/ARCHITECTURE.md).

## O diferencial: LLM com rede de segurança
A narrativa executiva é escrita pela **Claude** (modelo `claude-opus-4-8`, com
pensamento adaptativo), a partir do **perfil estatístico** do dataset — não dos
dados brutos (menos tokens, menos exposição de dados sensíveis).

Se não houver `ANTHROPIC_API_KEY`, se a variável `AIDA_DISABLE_LLM` estiver
setada, ou se a chamada falhar, o agente **cai automaticamente num resumo
determinístico** montado a partir do próprio perfil. Resultado: a aplicação nunca
quebra por causa do LLM, e os **testes e o CI rodam sem qualquer credencial**.

## Como executar

### Local (venv)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (opcional) habilita a narrativa por LLM:
export ANTHROPIC_API_KEY=sk-ant-...

uvicorn app.main:app --reload        # API em http://localhost:8000/docs
streamlit run app/dashboard.py       # dashboard em http://localhost:8501
pytest -q                            # testes (usam o fallback, sem chave)
```

### Docker
```bash
ANTHROPIC_API_KEY=sk-ant-... docker compose up --build   # API (8000) + dashboard (8501)
```

## Uso da API
```bash
# Análise em JSON (perfil + narrativa); ?target=coluna liga a modelagem
curl -X POST "http://localhost:8000/analyze?target=churn" \
  -F "file=@clientes.csv"

# Relatório executivo em PDF
curl -X POST "http://localhost:8000/report" \
  -F "file=@clientes.csv" -o relatorio.pdf
```

| Endpoint | Método | O que faz |
|---|---|---|
| `/health` | GET | Saúde do serviço e se o LLM está habilitado |
| `/analyze` | POST | CSV → JSON (perfil, narrativa, modelagem opcional) |
| `/report` | POST | CSV → relatório executivo em **PDF** |

## O que o relatório traz
- **Panorama**: dimensões, tipos, duplicatas, colunas com faltantes.
- **Qualidade dos dados**: faltantes por coluna, outliers (regra do IQR), duplicatas.
- **Padrões**: correlações fortes entre variáveis numéricas.
- **Gráficos**: histogramas, heatmap de correlação, barras de faltantes.
- **Modelagem** (se você indicar um alvo): classificação/regressão detectada
  automaticamente, placar de baselines (Regressão Logística/Linear vs Random
  Forest) e importância das variáveis.
- **Narrativa executiva** escrita pela Claude (ou o resumo determinístico).

## Estrutura
```
src/
  data/       ingest.py
  analysis/   profiler.py · charts.py
  modeling/   auto_model.py
  llm/        narrator.py (Claude + fallback)
  report/     pdf_report.py
  agent.py    orquestrador ponta a ponta
app/          main.py (FastAPI) · dashboard.py (Streamlit)
tests/        7 arquivos (ingestão, EDA, gráficos, modelagem, narrador, PDF, API)
reports/      ARCHITECTURE.md
```

## Decisões e melhorias futuras
Ver [`reports/ARCHITECTURE.md`](reports/ARCHITECTURE.md) — inclui por que o perfil
é o contrato central, por que o LLM é opcional com fallback, e um roteiro de
evolução (cache de narrativa, perguntas em linguagem natural sobre o CSV via tool
use, tipos semânticos, streaming, suporte a Excel/Parquet).

---
Projeto **05** do portfólio de Data Science Pleno. Anteriores: Customer Churn
MLOps, Credit Risk Scoring, Sales Demand Forecasting, Fraud Detection.
