"""Narrativa executiva do dataset via Claude — com fallback determinístico.

O diferencial do agente é traduzir o perfil estatístico (frio) em um **relatório
executivo** que um gestor lê e entende: o que é o dataset, qualidade dos dados,
padrões relevantes e recomendações. Isso é feito pela API da Claude
(modelo Opus 4.8, pensamento adaptativo).

Design importante para produção e testes:
- Se não há credencial (`ANTHROPIC_API_KEY`/perfil `ant`), se `AIDA_DISABLE_LLM`
  está setado, ou se a chamada falha por qualquer motivo, cai num **resumo
  determinístico** gerado a partir do próprio perfil. Assim a aplicação nunca
  quebra e os testes/CI rodam **sem** chave.
- O cliente pode ser injetado (para testes) via `Narrator(client=...)`.
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger("aida.narrator")

MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = (
    "Você é um cientista de dados sênior escrevendo um relatório executivo em "
    "português do Brasil para gestores não técnicos. A partir do perfil "
    "estatístico de um dataset, escreva uma análise clara e acionável. Seja "
    "concreto: cite números do perfil, aponte problemas de qualidade (faltantes, "
    "outliers, duplicatas), destaque correlações relevantes e termine com "
    "recomendações práticas. Não invente colunas ou valores que não estejam no "
    "perfil. Estruture em seções curtas com títulos."
)


class Narrator:
    def __init__(self, use_llm: bool | None = None, client=None, model: str = MODEL):
        self.model = model
        self._client = client
        # use_llm padrão: ligado, a menos que AIDA_DISABLE_LLM esteja setado
        # para um valor "verdadeiro" (1/true).
        if use_llm is None:
            disable = os.getenv("AIDA_DISABLE_LLM", "").lower()
            use_llm = disable not in ("1", "true", "yes")
        self.use_llm = bool(use_llm)

    def _get_client(self):
        if self._client is not None:
            return self._client
        import anthropic  # import tardio: só quando realmente formos usar

        self._client = anthropic.Anthropic()   # resolve credencial do ambiente
        return self._client

    def _call_llm(self, profile: dict, target_analysis: dict | None) -> str:
        payload = {"perfil": profile}
        if target_analysis:
            payload["modelagem"] = target_analysis
        user = (
            "Escreva o relatório executivo a partir deste perfil (JSON):\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )
        resp = self._get_client().messages.create(
            model=self.model,
            max_tokens=4000,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def narrate(self, profile: dict, target_analysis: dict | None = None) -> str:
        """Gera a narrativa; cai no fallback determinístico em qualquer falha."""
        if self.use_llm:
            try:
                text = self._call_llm(profile, target_analysis)
                if text:
                    return text
            except Exception as exc:  # sem chave, rede, rate limit, etc.
                logger.warning("LLM indisponível (%s); usando fallback determinístico.", exc)
        return fallback_summary(profile, target_analysis)


def fallback_summary(profile: dict, target_analysis: dict | None = None) -> str:
    """Resumo executivo determinístico, montado direto do perfil (sem LLM)."""
    rows = profile["shape"]["rows"]
    cols = profile["shape"]["columns"]
    lines = [
        "# Relatório executivo (resumo automático)",
        "",
        "## Visão geral",
        f"O dataset tem **{rows:,} linhas** e **{cols} colunas** "
        f"({len(profile['numeric_columns'])} numéricas, "
        f"{len(profile['categorical_columns'])} categóricas).".replace(",", "."),
    ]

    # Qualidade dos dados.
    quality = []
    if profile["missing"]:
        pior = max(profile["missing"].items(), key=lambda kv: kv[1]["pct"])
        quality.append(
            f"{len(profile['missing'])} coluna(s) com valores faltantes — "
            f"a pior é `{pior[0]}` com {pior[1]['pct']}% faltante."
        )
    if profile["duplicated_rows"]:
        quality.append(f"{profile['duplicated_rows']} linha(s) duplicada(s).")
    if profile["outliers"]:
        quality.append(f"{len(profile['outliers'])} coluna(s) com outliers pela regra do IQR.")
    lines += ["", "## Qualidade dos dados", "- " + "\n- ".join(quality) if quality else
              "", "## Qualidade dos dados", "Sem faltantes, duplicatas ou outliers relevantes."]
    if quality:
        lines = lines[:-2]  # remove o ramo "sem problemas" quando há problemas

    # Padrões / correlações.
    if profile["correlations"]:
        top = profile["correlations"][0]
        lines += [
            "", "## Padrões relevantes",
            f"Correlação mais forte: `{top['a']}` × `{top['b']}` (r = {top['corr']}). "
            f"Há {len(profile['correlations'])} par(es) com |correlação| ≥ 0,6.",
        ]

    # Modelagem, se houve.
    if target_analysis:
        best = target_analysis["best_model"]
        metric = target_analysis["metric"]
        score = next((r[metric] for r in target_analysis["leaderboard"] if r["model"] == best), None)
        feats = ", ".join(f["feature"] for f in target_analysis["feature_importance"][:3])
        lines += [
            "", "## Modelagem preditiva",
            f"Alvo `{target_analysis['target']}` ({target_analysis['task']}): o melhor baseline "
            f"foi **{best}** ({metric} = {score}). Variáveis mais influentes: {feats}.",
        ]

    lines += [
        "", "## Recomendações",
        "- Tratar faltantes e duplicatas antes de análises mais profundas."
        if (profile["missing"] or profile["duplicated_rows"]) else
        "- Base íntegra o suficiente para seguir para análises mais profundas.",
        "- Investigar as colunas com outliers antes de usar médias em decisões.",
    ]
    return "\n".join(lines)
