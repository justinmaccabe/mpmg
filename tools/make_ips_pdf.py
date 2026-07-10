"""Render ips.md into a branded MPMG PDF (black + gold, Times)."""
import re

from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle, NextPageTemplate,
                                PageBreak)

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(_ROOT, "ips.md")
OUT = os.path.join(_ROOT, "ips.pdf")

GOLD = HexColor("#C9A227")
INK = HexColor("#1A1A1A")
CHARCOAL = HexColor("#14181F")
GREY = HexColor("#6B7078")
HAIR = HexColor("#D9D2BC")

W, H = letter
MARGIN = 0.9 * inch

# Glyphs missing from the built-in Times encoding
SAFE = {"≈": "~", "≤": "<=", "≥": ">=", "Δ": "Delta", "σ": "sigma", "★": "*",
        "◆": "", " ": " "}


def clean(s):
    for k, v in SAFE.items():
        s = s.replace(k, v)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", s)
    return s


body = ParagraphStyle("body", fontName="Times-Roman", fontSize=10.5, leading=15,
                      textColor=INK, spaceAfter=8, alignment=4)  # justified
h2 = ParagraphStyle("h2", fontName="Times-Bold", fontSize=14.5, leading=18,
                    textColor=CHARCOAL, spaceBefore=18, spaceAfter=2)
h3 = ParagraphStyle("h3", fontName="Times-Bold", fontSize=12, leading=15,
                    textColor=CHARCOAL, spaceBefore=12, spaceAfter=4)
subtitle = ParagraphStyle("subtitle", fontName="Times-Italic", fontSize=10,
                          leading=14, textColor=GREY, spaceAfter=14)
cellL = ParagraphStyle("cellL", fontName="Times-Bold", fontSize=9.5, leading=12.5,
                       textColor=CHARCOAL)
cellR = ParagraphStyle("cellR", fontName="Times-Roman", fontSize=9.5, leading=12.5,
                       textColor=INK)
cellH = ParagraphStyle("cellH", fontName="Times-Bold", fontSize=9.5, leading=12,
                       textColor=GOLD)


def gold_rule(width=W - 2 * MARGIN):
    t = Table([[""]], colWidths=[width], rowHeights=[1.2])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.9, GOLD)]))
    return t


def md_table(rows):
    ncols = max(len(r) for r in rows)
    data = []
    for i, r in enumerate(rows):
        r = r + [""] * (ncols - len(r))
        if i == 0:
            data.append([Paragraph(clean(c), cellH) for c in r])
        else:
            data.append([Paragraph(clean(c), cellL if j == 0 else cellR)
                         for j, c in enumerate(r)])
    avail = W - 2 * MARGIN
    if ncols == 2:
        widths = [avail * 0.30, avail * 0.70]
    elif ncols == 3:
        widths = [avail * 0.5, avail * 0.25, avail * 0.25]
    else:
        widths = [avail / ncols] * ncols
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), CHARCOAL),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, HAIR),
        ("LINEBELOW", (0, 0), (-1, 0), 0.9, GOLD),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def cover(canv, doc):
    canv.saveState()
    canv.setFillColor(HexColor("#0B0E13"))
    canv.rect(0, 0, W, H, fill=1, stroke=0)
    # monogram
    size = 1.15 * inch
    x, y = W / 2 - size / 2, H - 3.4 * inch
    canv.setStrokeColor(GOLD)
    canv.setLineWidth(1.4)
    canv.roundRect(x, y, size, size, 6, fill=0, stroke=1)
    canv.setFillColor(GOLD)
    canv.setFont("Times-Bold", 52)
    canv.drawCentredString(W / 2, y + size * 0.24, "M")
    # titles
    canv.setFillColor(HexColor("#F4F4F4"))
    canv.setFont("Times-Roman", 26)
    canv.drawCentredString(W / 2, H - 4.55 * inch, "Maccabe Portfolio Management Group")
    canv.setFillColor(GOLD)
    canv.setFont("Times-Roman", 12.5)
    t = "P R I V A T E   W E A L T H   ·   Q U A N T I T A T I V E   S T R A T E G Y"
    canv.drawCentredString(W / 2, H - 4.95 * inch, t)
    canv.setStrokeColor(GOLD)
    canv.setLineWidth(0.8)
    canv.line(W / 2 - 1.6 * inch, H - 5.45 * inch, W / 2 + 1.6 * inch, H - 5.45 * inch)
    canv.setFillColor(HexColor("#F4F4F4"))
    canv.setFont("Times-Roman", 19)
    canv.drawCentredString(W / 2, H - 6.1 * inch, "Investment Policy Statement")
    canv.setFillColor(GREY)
    canv.setFont("Times-Italic", 10.5)
    canv.drawCentredString(W / 2, H - 6.45 * inch,
                           "v1.0  ·  Adopted  ·  Prepared May 27, 2026")
    canv.restoreState()


def interior(canv, doc):
    canv.saveState()
    canv.setStrokeColor(GOLD)
    canv.setLineWidth(0.6)
    canv.line(MARGIN, 0.62 * inch, W - MARGIN, 0.62 * inch)
    canv.setFillColor(GREY)
    canv.setFont("Times-Italic", 8.5)
    canv.drawString(MARGIN, 0.45 * inch,
                    "Maccabe Portfolio Management Group  ·  Investment Policy Statement")
    canv.setFont("Times-Roman", 8.5)
    canv.drawRightString(W - MARGIN, 0.45 * inch, f"{doc.page - 1}")
    canv.restoreState()


doc = BaseDocTemplate(OUT, pagesize=letter, leftMargin=MARGIN, rightMargin=MARGIN,
                      topMargin=MARGIN, bottomMargin=0.95 * inch,
                      title="MPMG Investment Policy Statement",
                      author="Justin Maccabe")
frame = Frame(MARGIN, 0.95 * inch, W - 2 * MARGIN, H - MARGIN - 0.95 * inch,
              id="main")
doc.addPageTemplates([
    PageTemplate(id="cover", frames=[frame], onPage=cover),
    PageTemplate(id="interior", frames=[frame], onPage=interior),
])

story = [NextPageTemplate("interior"), PageBreak()]
lines = open(SRC, encoding="utf-8").read().splitlines()
i = 0
while i < len(lines):
    ln = lines[i].rstrip()
    if not ln.strip():
        i += 1
        continue
    if ln.startswith("# "):        # doc title — covered by the cover page
        i += 1
        continue
    if ln.startswith("### "):
        story.append(Paragraph(clean(ln[4:]), h3))
        i += 1
        continue
    if ln.startswith("## "):
        story.append(Paragraph(clean(ln[3:]), h2))
        story.append(gold_rule())
        story.append(Spacer(1, 7))
        i += 1
        continue
    if ln.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            if not re.match(r"^[\s:|-]+$", "".join(cells)):
                rows.append(cells)
            i += 1
        if rows:
            story.append(Spacer(1, 3))
            story.append(md_table(rows))
            story.append(Spacer(1, 8))
        continue
    if ln.startswith("*") and ln.endswith("*") and not ln.startswith("**"):
        story.append(Paragraph(clean(ln.strip("*")), subtitle))
        i += 1
        continue
    story.append(Paragraph(clean(ln), body))
    i += 1

doc.build(story)
print("wrote", OUT)
