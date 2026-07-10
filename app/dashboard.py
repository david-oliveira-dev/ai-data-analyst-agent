"""Dashboard Streamlit do AI Data Analyst Agent.

Fluxo para o usuário de negócio: sobe um CSV, escolhe (opcional) a coluna-alvo,
e recebe a EDA automática, os gráficos, a narrativa executiva e o PDF para baixar.

Rode com:
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.agent import analyze_dataframe, report_pdf
from src.data.ingest import clean

st.set_page_config(page_title="AI Data Analyst Agent", page_icon="📊", layout="wide")
st.title("📊 AI Data Analyst Agent")
st.caption("Suba um CSV e receba uma análise executiva automática — EDA, gráficos, narrativa e PDF.")

uploaded = st.file_uploader("Arquivo CSV", type=["csv", "txt"])
if uploaded is None:
    st.info("Envie um CSV para começar.")
    st.stop()

df = clean(pd.read_csv(uploaded, sep=None, engine="python"))
st.success(f"Carregado: {len(df):,} linhas × {df.shape[1]} colunas".replace(",", "."))
st.dataframe(df.head(20), use_container_width=True)

col1, col2 = st.columns([3, 1])
target = col1.selectbox("Coluna-alvo para modelagem (opcional)", ["(nenhuma)"] + list(df.columns))
use_llm = col2.toggle("Usar LLM", value=False, help="Requer ANTHROPIC_API_KEY configurada")

if st.button("Analisar", type="primary"):
    with st.spinner("Analisando..."):
        result = analyze_dataframe(
            df, target=None if target == "(nenhuma)" else target, use_llm=use_llm
        )

    st.subheader("Relatório executivo")
    st.markdown(result.narrative)

    st.subheader("Gráficos")
    for name, png in result.charts.items():
        st.image(png, caption=name, use_container_width=True)

    if result.model_analysis:
        st.subheader("Modelagem preditiva")
        st.write(f"Tarefa: **{result.model_analysis['task']}** · "
                 f"melhor: **{result.model_analysis['best_model']}**")
        st.dataframe(pd.DataFrame(result.model_analysis["leaderboard"]), use_container_width=True)
        st.dataframe(pd.DataFrame(result.model_analysis["feature_importance"]),
                     use_container_width=True)

    st.download_button(
        "⬇️ Baixar relatório em PDF",
        data=report_pdf(result),
        file_name="relatorio.pdf",
        mime="application/pdf",
    )
