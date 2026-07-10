"""Compõe o relatório executivo em PDF (reportlab) a partir do perfil + narrativa.

Monta um documento com: capa/título, a narrativa executiva (do LLM ou fallback),
tabelas-resumo do perfil e os gráficos da EDA embutidos. Devolve os bytes do PDF
— quem chama decide se salva em disco ou serve pela API.
"""
from __future__ import annotations

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("H1x", parent=styles["Heading1"], spaceBefore=6, spaceAfter=6))
    styles.add(ParagraphStyle("H2x", parent=styles["Heading2"], spaceBefore=6, spaceAfter=4))
    styles.add(ParagraphStyle("Bodyx", parent=styles["BodyText"], leading=14))
    return styles


def _narrative_flowables(narrative: str, styles) -> list:
    """Converte a narrativa em markdown simples (#, ##, -) em flowables."""
    out = []
    for raw in narrative.splitlines():
        line = raw.rstrip()
        if not line:
            out.append(Spacer(1, 4))
        elif line.startswith("## "):
            out.append(Paragraph(line[3:], styles["H2x"]))
        elif line.startswith("# "):
            out.append(Paragraph(line[2:], styles["H1x"]))
        elif line.startswith("- "):
            out.append(Paragraph("• " + line[2:], styles["Bodyx"]))
        else:
            out.append(Paragraph(line.replace("**", ""), styles["Bodyx"]))
    return out


def _overview_table(profile: dict) -> Table:
    data = [
        ["Linhas", f"{profile['shape']['rows']:,}".replace(",", ".")],
        ["Colunas", str(profile["shape"]["columns"])],
        ["Numéricas", str(len(profile["numeric_columns"]))],
        ["Categóricas", str(len(profile["categorical_columns"]))],
        ["Linhas duplicadas", str(profile["duplicated_rows"])],
        ["Colunas com faltantes", str(len(profile["missing"]))],
    ]
    t = Table(data, colWidths=[6 * cm, 6 * cm])
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF2F7")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def _png_image(png: bytes, max_width: float = 16 * cm) -> Image:
    img = Image(io.BytesIO(png))
    ratio = img.imageHeight / float(img.imageWidth)
    img.drawWidth = min(max_width, img.imageWidth)
    img.drawHeight = img.drawWidth * ratio
    return img


def build_pdf(profile: dict, narrative: str, charts: dict[str, bytes] | None = None,
              title: str = "Relatório de Análise de Dados") -> bytes:
    """Gera o PDF completo e devolve os bytes."""
    charts = charts or {}
    styles = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=title,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.6 * cm)

    story: list = [
        Paragraph(title, styles["Title"]),
        Paragraph(f"Gerado em {datetime.now():%d/%m/%Y %H:%M}", styles["Bodyx"]),
        Spacer(1, 10),
        Paragraph("Panorama do dataset", styles["H2x"]),
        _overview_table(profile),
        Spacer(1, 12),
    ]
    story += _narrative_flowables(narrative, styles)

    if charts:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Gráficos", styles["H2x"]))
        for png in charts.values():
            story.append(_png_image(png))
            story.append(Spacer(1, 8))

    doc.build(story)
    return buf.getvalue()
