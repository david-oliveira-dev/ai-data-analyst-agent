# Arquitetura — AI Data Analyst Agent

## Visão geral

```
   CSV (upload) ─▶ Ingestão ─▶ EDA automática (perfil dict, serializável)
   (API/dashboard)   │              │
                     │              ├─▶ Gráficos (PNG headless)
                     │              ├─▶ Modelagem opcional (alvo → baselines + importância)
                     ▼              ▼
              Orquestrador ─▶ Narrador (LLM)
                                   │  Claude Opus 4.8 (thinking adaptativo)
                                   │  └─▶ fallback determinístico (sem chave/erro)
                                   ▼
                          Narrativa executiva
                                   │
                    ┌──────────────┴───────────────┐
                    ▼                                ▼
              PDF (reportlab)                  Respostas
        narrativa + tabelas + gráficos    API FastAPI / Dashboard Streamlit
```

## Fluxograma (requisição `/report`)

```
POST /report (CSV, target?) ─▶ valida upload (tipo/tamanho)
        │
        ▼
   ingest.load_csv ─▶ profiler.profile_dataframe ─▶ [target?] auto_model.run_automodel
        │                                                     │
        ▼                                                     ▼
   charts.build_all ───────────────▶ Narrator.narrate(profile, modelagem)
        │                                     │ LLM disponível? → Claude ; senão → fallback
        ▼                                     ▼
   pdf_report.build_pdf(perfil, narrativa, gráficos) ─▶ 200 application/pdf
```

## Decisões técnicas e trade-offs

- **Perfil como contrato central (dict serializável).** A EDA vira um dicionário
  de tipos nativos que alimenta PDF, dashboard e o prompt do LLM. Serializável =
  logável, cacheável e enviável à API sem surpresas de tipo numpy/pandas.
- **LLM opcional com fallback determinístico.** O valor do agente é a narrativa,
  mas ela **nunca** pode derrubar o sistema. Sem chave, com `AIDA_DISABLE_LLM`, ou
  em qualquer erro (rede, rate limit), cai num resumo montado do próprio perfil —
  o que também torna testes e CI independentes de credencial e de rede.
- **`claude-opus-4-8` + thinking adaptativo.** Modelo mais capaz para redigir
  análise executiva; o pensamento adaptativo deixa o modelo calibrar o esforço.
- **Modelagem "baseline" de propósito.** Objetivo é sinalizar "dá para prever
  isto?" e quais variáveis pesam — não um modelo de produção; por isso pré-proc
  robusto + 2 modelos + métrica adequada à tarefa, sem tuning caro.
- **Gráficos em backend `Agg`.** Sem display: roda em container e CI; PNG em bytes
  serve tanto o PDF quanto o Streamlit.
- **`prompt` do LLM só recebe o perfil (não os dados brutos).** Menos tokens,
  menos exposição de dados sensíveis, respostas mais focadas.

## Melhorias futuras
- Cache de narrativa por hash do perfil (evita recomputar/reenviar).
- Detecção de tipos semânticos (datas, IDs, moeda) para EDA mais rica.
- Perguntas em linguagem natural sobre o CSV (tool use / SQL sobre o DataFrame).
- Streaming da narrativa no dashboard e no endpoint.
- Suporte a Excel/Parquet além de CSV; amostragem para arquivos muito grandes.
```
