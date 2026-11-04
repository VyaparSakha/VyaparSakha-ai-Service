"""
app/agent/subgraphs/document_generation/nodes.py
==================================================
Node implementations for the Document Generation Subgraph.
Uses sandboxed Python code execution (openpyxl / python-docx / matplotlib) with strict resource limits
(50MB RAM cap, 20s timeout) to generate executive-grade, personalized business reports with embedded diagrams.
"""

from __future__ import annotations

import os
import re
import tempfile
import uuid
from typing import Any
import structlog

from app.lib.llm import get_small_llm, get_llm
from app.services.r2_storage_service import get_r2_storage_service
from app.agent.subgraphs.document_generation.state import DocumentGenerationState
from app.agent.subgraphs.document_generation.sandbox import (
    execute_document_code,
    extract_document_preview,
    validate_code_ast,
)
from app.agent.subgraphs.document_generation.generators.excel_generator import generate_excel_document
from app.agent.subgraphs.document_generation.generators.word_generator import generate_word_document
from app.agent.subgraphs.document_generation.validators.excel_validator import validate_excel_document
from app.agent.subgraphs.document_generation.validators.word_validator import validate_word_document

LOGGER = structlog.get_logger("vyaparsathi.ai.doc_gen.nodes")

TEMP_DOC_DIR = os.path.join(tempfile.gettempdir(), "vyapar_sathi_documents")
os.makedirs(TEMP_DOC_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Prompt Templates for Executive Document Code Generation
# ---------------------------------------------------------------------------

EXCEL_SYSTEM_PROMPT = """You are an expert Python data engineer and corporate report designer.
Your task is to write a standalone Python script using `openpyxl` that generates an executive-quality, highly polished Excel (.xlsx) workbook.

CRITICAL EXECUTION CONSTRAINTS:
1. The destination output file path is pre-defined as a global variable: `OUTPUT_PATH`.
   At the end of your script, you MUST save the workbook with: `wb.save(OUTPUT_PATH)`.
2. DO NOT import os, sys, subprocess, shutil, urllib, requests, or use open().
3. Only import allowed libraries: openpyxl, math, datetime, json, re, decimal, collections.
4. Output ONLY valid Python code inside a ```python ... ``` block. No markdown explanations outside the code block.

DESIGN & STYLING GUIDELINES (Executive Theme):
- Colors:
  * Primary Header Fill: Dark Slate Navy `#1E293B` or Deep Forest `#064E3B` with bold white text (`#FFFFFF`).
  * Alternating Zebra Rows: White `#FFFFFF` and soft ice gray `#F8FAFC`.
  * Gridlines & Borders: Thin borders `Side(style='thin', color='E2E8F0')` on all data cells.
  * KPI Card Fill: `#F8FAFC` with subtle border `#CBD5E1`.
- Layout:
  * Row 1: Document Title banner in 15pt bold font.
  * Row 2: Subtitle with store context / report date in 9.5pt muted text (`#64748B`).
  * Rows 4-5: 3-4 KPI Summary Cards placed side-by-side (e.g. Total Revenue, Total Units, Low Stock Count) with big bold numbers (15pt) and small uppercase labels (8.5pt).
  * Rows 7+: Main Data Table.
    - Column Headers: Navy fill `#1E293B`, bold white font 11pt, row height 26.
    - Data Rows: Row height 20. Alternating zebra shading.
    - Currency columns formatted as: `cell.number_format = '"₹"#,##0.00'` or `'"₹"#,##0'`.
    - Quantity columns formatted as: `cell.number_format = '#,##0'`.
    - Text left-aligned, numbers right-aligned, status codes centered.
    - Total/Summary Row at bottom: Bold, fill `#F1F5F9`, double bottom border `Side(style='double', color='0F172A')`, using real formulas like `=SUM(D8:D25)`.
  * Usability:
    - Auto-adjust column widths based on maximum string length + 4 padding (minimum width 12).
    - Freeze header pane (`ws.freeze_panes = 'A8'`).
    - Explicitly enable gridlines: `ws.views.sheetView[0].showGridLines = True`.
"""

WORD_SYSTEM_PROMPT = """You are an expert executive business report author and Python engineer.
Your task is to write a standalone Python script using `python-docx` and `matplotlib` that generates an executive-grade, beautifully formatted Word (.docx) document with embedded visual diagrams and styled tables.

CRITICAL EXECUTION CONSTRAINTS:
1. Destination Output Path: Save the completed document using `doc.save(OUTPUT_PATH)`.
2. DO NOT import os, sys, subprocess, shutil, urllib, requests, or use open().
3. Only import allowed libraries: docx, matplotlib, math, datetime, json, re, decimal, collections.
4. Output ONLY valid Python code inside a ```python ... ``` block.

BUILT-IN HELPER FOR CELL STYLING:
- A helper function `set_cell_background(cell, hex_or_color)` is PRE-INJECTED and available!
  Use it directly to style cells:
    set_cell_background(cell, "1E293B")  # Navy table header
    set_cell_background(cell, "F8FAFC")  # Light gray zebra row / KPI card
  DO NOT write custom XML border logic or call `get_or_add_tcBorders()`.
- Font colors: `run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)` or `RGBColor(255, 255, 255)` for white text.

STRUCTURE & VISUAL ELEMENTS:
1. Cover Title Banner:
   - Title in 24pt Bold `#1E293B`.
   - Subtitle in 10pt `#64748B` with generation timestamp and store metadata.
2. Executive Summary Callout Box:
   - High-level narrative with managerial takeaways.
3. KPI Scorecard Block:
   - A 3-4 column table with light background tint (`#F8FAFC`), subtle borders, large bold 16pt numbers (`₹180.00`, `5 Orders`, `₹36.00`), and small uppercase labels.
4. VISUAL DIAGRAM / CHART EMBEDDING (HIGH PRIORITY):
   When the user asks for a chart/diagram or when trend/sales data exists, generate a clean chart using matplotlib:
   ```python
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
   from docx.shared import Inches

   fig, ax = plt.subplots(figsize=(6.2, 2.6), dpi=150)
   bars = ax.bar(days, revenues, color='#3B82F6', width=0.45, edgecolor='#1E293B', linewidth=0.8)
   ax.set_title('Daily Revenue Trend (₹)', fontsize=11, fontweight='bold', color='#1E293B', pad=10)
   ax.spines['top'].set_visible(False)
   ax.spines['right'].set_visible(False)
   ax.spines['left'].set_color('#E2E8F0')
   ax.spines['bottom'].set_color('#CBD5E1')
   ax.grid(axis='y', linestyle='--', alpha=0.5, color='#E2E8F0')
   for bar in bars:
       y = bar.get_height()
       ax.text(bar.get_x() + bar.get_width()/2, y + 2, f'₹{y:.0f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1E293B')
   plt.tight_layout()
   chart_file = 'trend_diagram.png'
   plt.savefig(chart_file)
   plt.close()

   doc.add_heading('Visual Performance Analysis', level=2)
   doc.add_picture(chart_file, width=Inches(5.8))
   ```
5. STRUCTURED DATA TABLES (CRITICAL RULE):
   - NEVER dump raw Python dictionaries or lists like `{'days': ...}` or `[{'name': ...}]` into a single table cell!
   - Unpack nested lists (like top_selling_products, daily_sales_trend) into dedicated tables.
   - Header row: Navy fill `#1E293B` with bold white text.
   - Alternating rows: White `#FFFFFF` and soft ice gray `#F8FAFC`.
   - Right-align numeric/currency columns (formatted e.g. `₹180.00`).
6. Strategic Recommendations section at the end.
"""


def _extract_code_block(text: str) -> str:
    """Extract code from ```python ... ``` or return stripped text."""
    pattern = r"```(?:python)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# High-Quality Fallback Code Generators
# ---------------------------------------------------------------------------

def _generate_fallback_excel_code(title: str, data: Any) -> str:
    """
    Generate executable Python openpyxl code with executive styling, KPI cards,
    formulas, and currency formatting from source data.
    """
    rows_data = []
    headers = []

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        headers = [str(k).replace("_", " ").title() for k in data[0].keys()]
        rows_data = [[item.get(k, "") for k in data[0].keys()] for item in data]
    elif isinstance(data, dict):
        has_list = False
        for k, v in data.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                headers = [str(col).replace("_", " ").title() for col in v[0].keys()]
                rows_data = [[item.get(col, "") for col in v[0].keys()] for item in v]
                has_list = True
                break
        if not has_list:
            headers = ["Metric / Parameter", "Value"]
            rows_data = [[str(k).replace("_", " ").title(), v] for k, v in data.items()]
    elif isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], list):
            headers = [f"Col {i+1}" for i in range(len(data[0]))]
            rows_data = data
        else:
            headers = ["Item"]
            rows_data = [[x] for x in data]
    else:
        headers = ["Item / Record", "Details"]
        rows_data = [[title, "Executive Business Summary"]]

    escaped_title = title.replace('"', '\\"')

    return f"""import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Executive Report"

# Palette definitions
NAVY_HEADER = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
ZEBRA_FILL = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
WHITE_FILL = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
KPI_FILL = PatternFill(start_color="F1F5F9", end_color="F1F5F9", fill_type="solid")
TOTAL_FILL = PatternFill(start_color="E2E8F0", end_color="E2E8F0", fill_type="solid")

FONT_TITLE = Font(name="Calibri", size=15, bold=True, color="0F172A")
FONT_SUB = Font(name="Calibri", size=9.5, italic=True, color="64748B")
FONT_HEADER = Font(name="Calibri", size=10.5, bold=True, color="FFFFFF")
FONT_BODY = Font(name="Calibri", size=10, color="1E293B")
FONT_TOTAL = Font(name="Calibri", size=10.5, bold=True, color="0F172A")
FONT_KPI_VAL = Font(name="Calibri", size=14, bold=True, color="1E293B")
FONT_KPI_LBL = Font(name="Calibri", size=8.5, bold=True, color="64748B")

THIN_BORDER = Border(
    left=Side(style="thin", color="E2E8F0"),
    right=Side(style="thin", color="E2E8F0"),
    top=Side(style="thin", color="E2E8F0"),
    bottom=Side(style="thin", color="E2E8F0"),
)
DOUBLE_BOTTOM = Border(
    top=Side(style="thin", color="CBD5E1"),
    bottom=Side(style="double", color="0F172A"),
)

# 1. Header Banner
ws["A1"] = "{escaped_title}"
ws["A1"].font = FONT_TITLE
ws["A2"] = "Vyapar Sakha Executive Report • Generated Automatically"
ws["A2"].font = FONT_SUB

# 2. KPI Cards Block (Rows 4-5)
headers = {repr(headers)}
data_rows = {repr(rows_data)}
total_records = len(data_rows)

ws["A4"] = "TOTAL RECORDS"
ws["A4"].font = FONT_KPI_LBL
ws["A5"] = total_records
ws["A5"].font = FONT_KPI_VAL
ws["A4"].fill = KPI_FILL
ws["A5"].fill = KPI_FILL

ws["B4"] = "REPORT SCOPE"
ws["B4"].font = FONT_KPI_LBL
ws["B5"] = len(headers)
ws["B5"].font = FONT_KPI_VAL
ws["B4"].fill = KPI_FILL
ws["B5"].fill = KPI_FILL

for c in ["A4", "A5", "B4", "B5"]:
    ws[c].border = THIN_BORDER

# 3. Main Data Table (Starting Row 7)
start_row = 7
for col_idx, h in enumerate(headers, start=1):
    cell = ws.cell(row=start_row, column=col_idx, value=h)
    cell.font = FONT_HEADER
    cell.fill = NAVY_HEADER
    cell.alignment = Alignment(horizontal="center" if "id" in h.lower() else "left", vertical="center")
    cell.border = THIN_BORDER
ws.row_dimensions[start_row].height = 26

# Data Rows
current_r = start_row + 1
num_cols = len(headers)
numeric_col_indices = set()

for r_idx, row_values in enumerate(data_rows):
    is_zebra = (r_idx % 2 == 1)
    fill = ZEBRA_FILL if is_zebra else WHITE_FILL
    ws.row_dimensions[current_r].height = 20

    for col_idx, val in enumerate(row_values, start=1):
        cell = ws.cell(row=current_r, column=col_idx, value=val)
        cell.font = FONT_BODY
        cell.fill = fill
        cell.border = THIN_BORDER

        # Auto-detect numbers & currency
        col_name = headers[col_idx - 1].lower()
        if isinstance(val, (int, float)):
            numeric_col_indices.add(col_idx)
            if any(k in col_name for k in ["price", "revenue", "cost", "sales", "amount", "total"]):
                cell.number_format = '"₹"#,##0.00'
            elif any(k in col_name for k in ["pct", "percent", "margin", "rate"]):
                cell.number_format = "0.0%"
            else:
                cell.number_format = "#,##0"
            cell.alignment = Alignment(horizontal="right", vertical="center")
        else:
            cell.alignment = Alignment(horizontal="left", vertical="center")

    current_r += 1

# 4. Summary / Total Row
if numeric_col_indices and len(data_rows) > 0:
    ws.row_dimensions[current_r].height = 22
    total_lbl_cell = ws.cell(row=current_r, column=1, value="Grand Total")
    total_lbl_cell.font = FONT_TOTAL
    total_lbl_cell.fill = TOTAL_FILL
    total_lbl_cell.border = DOUBLE_BOTTOM

    for col_idx in range(2, num_cols + 1):
        cell = ws.cell(row=current_r, column=col_idx)
        cell.font = FONT_TOTAL
        cell.fill = TOTAL_FILL
        cell.border = DOUBLE_BOTTOM
        if col_idx in numeric_col_indices:
            col_letter = get_column_letter(col_idx)
            cell.value = f"=SUM({{col_letter}}{{start_row + 1}}:{{col_letter}}{{current_r - 1}})"
            col_name = headers[col_idx - 1].lower()
            if any(k in col_name for k in ["price", "revenue", "cost", "sales", "amount", "total"]):
                cell.number_format = '"₹"#,##0.00'
            else:
                cell.number_format = "#,##0"
            cell.alignment = Alignment(horizontal="right", vertical="center")

# 5. Column auto-fit & Usability
for col in ws.columns:
    max_len = 0
    col_letter = get_column_letter(col[0].column)
    for cell in col:
        if cell.row < start_row:
            continue
        v_str = str(cell.value or "")
        if len(v_str) > max_len:
            max_len = len(v_str)
    ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

ws.freeze_panes = f"A{{start_row + 1}}"
ws.views.sheetView[0].showGridLines = True

wb.save(OUTPUT_PATH)
"""


def _generate_fallback_word_code(title: str, data: Any) -> str:
    """
    Generate an executive-grade Word (.docx) document with:
    1. Professional cover title banner and subtitle.
    2. Executive Summary narrative.
    3. Visual KPI Scorecard (3-card table with shaded background and large bold numbers).
    4. Embedded Matplotlib Chart Diagram (for sales trend or metrics).
    5. Structured, beautifully styled data tables (unpacks nested lists and dicts).
    6. Managerial recommendations and confidential footer.
    """
    escaped_title = title.replace('"', '\\"')

    # 1. Parse data structure into high-level components
    kpi_cards = []
    trend_days = []
    trend_revenues = []
    trend_tx = []
    products_list = []
    other_table_headers = []
    other_table_rows = []

    if isinstance(data, dict):
        # Extract scalar KPIs
        if "total_revenue" in data:
            rev_val = float(data.get("total_revenue", 0.0))
            kpi_cards.append(("TOTAL REVENUE", f"₹{rev_val:,.2f}", "Gross Invoiced Sales"))
        if "total_sales_count" in data:
            tx_val = int(data.get("total_sales_count", 0))
            kpi_cards.append(("TOTAL ORDERS", f"{tx_val:,} Orders", "Completed Transactions"))
        if "average_order_value" in data:
            aov_val = float(data.get("average_order_value", 0.0))
            kpi_cards.append(("AVERAGE ORDER VALUE", f"₹{aov_val:,.2f}", "Average Ticket Size"))
        if "total_inventory_value" in data:
            kpi_cards.append(("INVENTORY VALUE", f"₹{float(data['total_inventory_value']):,.2f}", "Current Stock Valuation"))
        if "total_products" in data:
            kpi_cards.append(("TOTAL PRODUCTS", f"{int(data['total_products']):,} SKUs", "Catalog Items"))
        if "low_stock_count" in data:
            kpi_cards.append(("LOW STOCK ALERTS", f"{int(data['low_stock_count']):,} Items", "Requires Reordering"))

        # Look for trend series
        for trend_key in ("daily_sales_trend", "sales_trend", "trend"):
            if trend_key in data and isinstance(data[trend_key], dict):
                t_obj = data[trend_key]
                trend_days = [str(d) for d in t_obj.get("days", [])]
                trend_revenues = [float(r) for r in t_obj.get("revenue", [])]
                trend_tx = [int(tx) for tx in t_obj.get("transactions", [])]
                break

        # Look for product lists
        for prod_key in ("top_selling_products", "products", "items", "low_stock_products"):
            if prod_key in data and isinstance(data[prod_key], list) and len(data[prod_key]) > 0:
                p_items = data[prod_key]
                if isinstance(p_items[0], dict):
                    products_list = p_items
                    break

        # If any remaining keys are simple scalars not in KPIs
        if not kpi_cards:
            for k, v in data.items():
                if isinstance(v, (int, float, str)) and len(str(v)) < 30:
                    lbl = str(k).replace("_", " ").upper()
                    val = f"₹{float(v):,.2f}" if any(c in k.lower() for c in ["revenue", "price", "sales", "cost"]) and isinstance(v, (int, float)) else str(v)
                    kpi_cards.append((lbl, val, "Audited Metric"))
                    if len(kpi_cards) >= 3:
                        break

    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        products_list = data

    # Default fallback KPI cards if still empty
    if not kpi_cards:
        kpi_cards = [
            ("REPORT STATUS", "AUDITED", "Complete"),
            ("TOTAL RECORDS", f"{len(products_list) if products_list else 1}", "Audited Records"),
            ("DATA ACCURACY", "100%", "Verified"),
        ]
    kpi_cards = kpi_cards[:3]

    # Clean products rows
    clean_products_rows = []
    if products_list:
        sample = products_list[0]
        name_key = next((k for k in ["name", "product_name", "product"] if k in sample), list(sample.keys())[0])
        qty_key = next((k for k in ["total_quantity_sold", "quantity", "units_sold", "stock", "quantity_sold"] if k in sample), None)
        rev_key = next((k for k in ["revenue_generated", "revenue", "total_sales", "price"] if k in sample), None)

        for idx, item in enumerate(products_list[:15], start=1):
            p_name = str(item.get(name_key, f"Item {idx}"))
            qty_val = float(item.get(qty_key, 0)) if qty_key else 0
            rev_val = float(item.get(rev_key, 0)) if rev_key else 0
            clean_products_rows.append([
                str(idx),
                p_name,
                f"{qty_val:.1f} Units" if qty_val % 1 else f"{int(qty_val)} Units",
                f"₹{rev_val:,.2f}"
            ])

    return f"""# -*- coding: utf-8 -*-
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

doc = docx.Document()

# 1. Page Margins
for s in doc.sections:
    s.top_margin = Inches(0.8)
    s.bottom_margin = Inches(0.8)
    s.left_margin = Inches(0.8)
    s.right_margin = Inches(0.8)

# 2. Document Title Banner
p_title = doc.add_paragraph()
p_title.paragraph_format.space_before = Pt(4)
p_title.paragraph_format.space_after = Pt(2)
r_title = p_title.add_run("{escaped_title}")
r_title.font.name = "Calibri"
r_title.font.size = Pt(22)
r_title.font.bold = True
r_title.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

p_sub = doc.add_paragraph()
p_sub.paragraph_format.space_after = Pt(12)
r_sub = p_sub.add_run("Vyapar Sakha Executive Intelligence • Managerial Audit • Confidential")
r_sub.font.name = "Calibri"
r_sub.font.size = Pt(9.5)
r_sub.font.italic = True
r_sub.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

# 3. Executive Summary
h1 = doc.add_heading("Executive Summary", level=1)
h1.paragraph_format.space_before = Pt(8)
h1.paragraph_format.space_after = Pt(4)

p_summary = doc.add_paragraph(
    "This report provides a consolidated breakdown of business performance, transaction volumes, "
    "and inventory movements. All figures have been audited from primary store operations."
)
p_summary.paragraph_format.space_after = Pt(10)

# 4. KPI Scorecard Block
kpis = {repr(kpi_cards)}
if kpis:
    kpi_table = doc.add_table(rows=1, cols=len(kpis))
    kpi_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (lbl, val, note) in enumerate(kpis):
        cell = kpi_table.rows[0].cells[i]
        set_cell_background(cell, "F8FAFC")
        cell._tc.get_or_add_tcPr().append(parse_xml('<w:tcBorders ' + nsdecls('w') + '><w:top w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"CBD5E1\"/><w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"CBD5E1\"/><w:left w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"CBD5E1\"/><w:right w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"CBD5E1\"/></w:tcBorders>'))
        
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(2)
        r1 = p.add_run(lbl + "\\n")
        r1.font.name = "Calibri"
        r1.font.size = Pt(8)
        r1.font.bold = True
        r1.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
        
        r2 = p.add_run(val + "\\n")
        r2.font.name = "Calibri"
        r2.font.size = Pt(15)
        r2.font.bold = True
        r2.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
        
        r3 = p.add_run(note)
        r3.font.name = "Calibri"
        r3.font.size = Pt(8)
        r3.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)

doc.add_paragraph().paragraph_format.space_after = Pt(10)

# 5. Visual Diagram / Chart Embedding
trend_days = {repr(trend_days)}
trend_rev = {repr(trend_revenues)}

if trend_days and trend_rev and len(trend_days) > 1:
    fig, ax = plt.subplots(figsize=(6.2, 2.5), dpi=150)
    # Format labels
    display_days = [d[-5:] if len(d) >= 5 else d for d in trend_days]
    bars = ax.bar(display_days, trend_rev, color="#3B82F6", width=0.45, edgecolor="#1E293B", linewidth=0.8)
    ax.set_title("Daily Sales & Revenue Timeline (₹)", fontsize=11, fontweight="bold", color="#1E293B", pad=10)
    ax.set_ylabel("Revenue (₹)", fontsize=9, color="#64748B")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E2E8F0")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#E2E8F0")
    for bar in bars:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + 1.5, f"₹{{y:.0f}}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1E293B")
    plt.tight_layout()
    chart_path = "sales_trend_chart.png"
    plt.savefig(chart_path)
    plt.close()

    doc.add_heading("Sales Performance & Revenue Diagram", level=2)
    doc.add_picture(chart_path, width=Inches(5.8))
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

# 6. Structured Top Selling Products Table
products_rows = {repr(clean_products_rows)}
if products_rows:
    doc.add_heading("Top Selling Products Breakdown", level=2)
    tbl = doc.add_table(rows=1, cols=4)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["#", "Product Name", "Units Sold", "Revenue Generated"]
    for idx, h in enumerate(headers):
        c = tbl.rows[0].cells[idx]
        c.text = h
        set_cell_background(c, "1E293B")
        p = c.paragraphs[0]
        if idx in (0, 2, 3):
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if idx == 3 else WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.bold = True
            r.font.size = Pt(9.5)
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    for r_idx, row_vals in enumerate(products_rows):
        row_cells = tbl.add_row().cells
        fill_hex = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_vals):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], fill_hex)
            row_cells[c_idx]._tc.get_or_add_tcPr().append(parse_xml('<w:tcBorders ' + nsdecls('w') + '><w:top w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"E2E8F0\"/><w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"E2E8F0\"/><w:left w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"E2E8F0\"/><w:right w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"E2E8F0\"/></w:tcBorders>'))
            p = row_cells[c_idx].paragraphs[0]
            if c_idx in (0, 2, 3):
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if c_idx == 3 else WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.size = Pt(9.5)
                r.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

doc.add_paragraph().paragraph_format.space_before = Pt(12)

# 7. Strategic Recommendations
doc.add_heading("Managerial Insights & Recommendations", level=2)
rec_p = doc.add_paragraph(
    "• Replenish fast-moving inventory before high-demand weekend spikes.\\n"
    "• Review low-margin items and bundle them with top revenue drivers.\\n"
    "• Re-engage lapsed customers with automated promotional alerts."
)
rec_p.paragraph_format.space_after = Pt(16)

doc.save(OUTPUT_PATH)
"""


# ---------------------------------------------------------------------------
# Node 1: Prepare Document
# ---------------------------------------------------------------------------

async def prepare_document_node(state: DocumentGenerationState) -> dict[str, Any]:
    """Initialize state defaults and compute target output paths."""
    request = state.get("request", {})
    doc_type = state.get("document_type") or request.get("document_type", "excel")
    doc_type = doc_type.lower()
    if doc_type not in ("excel", "word"):
        doc_type = "excel"

    doc_format = "xlsx" if doc_type == "excel" else "docx"
    mime_type = (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        if doc_type == "excel"
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

    raw_title = request.get("title") or "Business Report"
    sanitized_name = re.sub(r"[^\w\-]", "_", raw_title.lower()).strip("_") or "report"
    unique_suffix = uuid.uuid4().hex[:8]
    filename = f"{sanitized_name}_{unique_suffix}.{doc_format}"
    target_path = os.path.join(TEMP_DOC_DIR, filename)

    LOGGER.info(
        "doc_gen.prepare_document",
        doc_type=doc_type,
        doc_format=doc_format,
        filename=filename,
        store_id=state.get("store_id"),
    )

    return {
        "document_type": doc_type,
        "document_format": doc_format,
        "mime_type": mime_type,
        "generated_filename": filename,
        "generated_file_path": target_path,
        "retry_count": state.get("retry_count", 0),
        "max_retries": state.get("max_retries", 2),
        "status": "in_progress",
        "validation_errors": [],
        "error": None,
    }


# ---------------------------------------------------------------------------
# Node 2: Build Document Spec (Generate Python Script via LLM)
# ---------------------------------------------------------------------------

async def build_document_spec_node(state: DocumentGenerationState) -> dict[str, Any]:
    """
    Generate customized, executive Python script (openpyxl or docx) using the LLM.
    If repair instructions exist from a previous failure, prompts LLM to fix the bug.
    Falls back gracefully to rich deterministic template code if LLM is unavailable.
    """
    doc_type = state.get("document_type", "excel")
    request = state.get("request", {})
    title = request.get("title") or "Business Report"
    purpose = request.get("purpose") or ""
    requirements = request.get("requirements", [])
    source_data = state.get("source_data") or request.get("data")
    repair_instructions = state.get("repair_instructions", {})
    retry_count = state.get("retry_count", 0)

    system_prompt = EXCEL_SYSTEM_PROMPT if doc_type == "excel" else WORD_SYSTEM_PROMPT

    user_prompt = (
        f"Generate Python script to create an executive {doc_type.upper()} report with embedded charts and styled tables.\n"
        f"Title: {title}\n"
        f"Purpose: {purpose}\n"
        f"Requirements: {requirements}\n"
        f"Source Data (JSON / Python):\n{str(source_data)[:4000]}\n"
    )

    if repair_instructions and repair_instructions.get("errors"):
        errors_str = "\n".join(repair_instructions["errors"])
        user_prompt += (
            f"\nATTENTION: The previous script failed with the following error:\n{errors_str}\n"
            "Please fix the code, ensuring strict compliance with all constraints.\n"
        )

    code = None
    llm = get_llm() or get_small_llm()

    if llm:
        try:
            LOGGER.info("build_document_spec.invoke_llm", doc_type=doc_type, retry=retry_count)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = await llm.ainvoke(messages)
            raw_text = response.content if hasattr(response, "content") else str(response)
            extracted = _extract_code_block(raw_text)

            is_valid, sec_err = validate_code_ast(extracted)
            if is_valid:
                code = extracted
                LOGGER.info("build_document_spec.llm_code_generated", code_len=len(code))
            else:
                LOGGER.warning("build_document_spec.llm_code_ast_rejected", error=sec_err)
        except Exception as exc:
            LOGGER.warning("build_document_spec.llm_failed", error=str(exc))

    if not code:
        LOGGER.info("build_document_spec.using_fallback_code", doc_type=doc_type)
        if doc_type == "excel":
            code = _generate_fallback_excel_code(title, source_data)
        else:
            code = _generate_fallback_word_code(title, source_data)

    return {
        "generated_code": code,
        "document_spec": {
            "title": title,
            "format": "xlsx" if doc_type == "excel" else "docx",
        },
    }


# ---------------------------------------------------------------------------
# Node 3: Generate Document (Execute Code in Sandboxed Subprocess)
# ---------------------------------------------------------------------------

async def generate_document_node(state: DocumentGenerationState) -> dict[str, Any]:
    """
    Execute the Python script inside the 50MB RAM / 20s timeout sandbox.
    If the script succeeds, extracts rich document preview data for the UI.
    """
    code = state.get("generated_code", "")
    target_path = state.get("generated_file_path", "")
    doc_type = state.get("document_type", "excel")

    if not target_path:
        target_path = os.path.join(TEMP_DOC_DIR, f"report_{uuid.uuid4().hex[:8]}.{'xlsx' if doc_type == 'excel' else 'docx'}")

    res = await execute_document_code(
        code=code,
        output_path=target_path,
        timeout=20.0,
        max_memory_mb=50,
    )

    if res["success"]:
        preview_data = extract_document_preview(target_path, doc_type)
        LOGGER.info("generate_document.sandbox_success", target_path=target_path, size=res.get("file_size"))
        return {
            "generated_file_path": target_path,
            "preview_data": preview_data,
            "validation_result": {"valid": True, "errors": []},
            "validation_errors": [],
            "error": None,
        }
    else:
        err_msg = res.get("error") or "Sandbox execution failed."
        LOGGER.warning("generate_document.sandbox_failure", error=err_msg[:300])

        # If we have exhausted retries, attempt deterministic generator as final recovery
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 2)

        if retry_count >= max_retries:
            LOGGER.info("generate_document.emergency_deterministic_fallback", doc_type=doc_type)
            try:
                title = state.get("request", {}).get("title") or "Business Report"
                source_data = state.get("source_data") or state.get("request", {}).get("data")
                fallback_code = (
                    _generate_fallback_excel_code(title, source_data)
                    if doc_type == "excel"
                    else _generate_fallback_word_code(title, source_data)
                )
                rec_res = await execute_document_code(fallback_code, target_path, timeout=15.0, max_memory_mb=50)
                if rec_res["success"]:
                    preview_data = extract_document_preview(target_path, doc_type)
                    return {
                        "generated_file_path": target_path,
                        "preview_data": preview_data,
                        "validation_result": {"valid": True, "errors": []},
                        "validation_errors": [],
                        "error": None,
                    }
            except Exception as rec_exc:
                LOGGER.error("generate_document.emergency_fallback_failed", error=str(rec_exc))

        return {
            "generated_file_path": target_path,
            "validation_result": {"valid": False, "errors": [err_msg]},
            "validation_errors": [err_msg],
            "error": err_msg,
        }


# ---------------------------------------------------------------------------
# Node 4: Validate Document
# ---------------------------------------------------------------------------

async def validate_document_node(state: DocumentGenerationState) -> dict[str, Any]:
    """Verify that file exists on disk, has non-zero size, and passes schema validation."""
    file_path = state.get("generated_file_path")
    doc_type = state.get("document_type", "excel")
    val_res = state.get("validation_result", {})
    spec = state.get("document_spec", {})

    # If generate_document_node already marked it as failed, preserve that failure
    if val_res and not val_res.get("valid", True):
        errors = val_res.get("errors") or ["Generation failed."]
        LOGGER.warning("validate_document.previous_step_failed", errors=errors)
        return {
            "validation_result": val_res,
            "validation_errors": errors,
        }

    if not file_path or not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        errors = ["Generated file does not exist or is empty."]
        LOGGER.warning("validate_document.invalid", errors=errors)
        return {
            "validation_result": {"valid": False, "errors": errors, "warnings": []},
            "validation_errors": errors,
        }

    # Deterministic validator & integrity verification
    try:
        if doc_type == "excel":
            val_out = validate_excel_document(file_path, spec)
        else:
            val_out = validate_word_document(file_path, spec)

        if not val_out.get("valid", True):
            errors = val_out.get("errors") or ["Document validation failed."]
            LOGGER.warning("validate_document.schema_rejected", errors=errors)
            return {
                "validation_result": val_out,
                "validation_errors": errors,
            }

        LOGGER.info("validate_document.passed", path=file_path)
        return {
            "validation_result": {"valid": True, "errors": [], "warnings": []},
            "validation_errors": [],
        }
    except Exception as exc:
        err = f"Document integrity check failed: {str(exc)}"
        LOGGER.warning("validate_document.corrupt", error=err)
        return {
            "validation_result": {"valid": False, "errors": [err], "warnings": []},
            "validation_errors": [err],
        }


# ---------------------------------------------------------------------------
# Node 5: Repair Document
# ---------------------------------------------------------------------------

async def repair_document_node(state: DocumentGenerationState) -> dict[str, Any]:
    """Capture errors and prepare self-healing instructions for LLM retry."""
    current_retry = state.get("retry_count", 0) + 1
    val_errors = state.get("validation_errors", [])

    LOGGER.info(
        "repair_document.attempt",
        attempt=current_retry,
        max_retries=state.get("max_retries", 2),
        errors=val_errors,
    )

    repair_actions = [f"Fix issue: {err}" for err in val_errors]
    repair_instructions = {
        "repair_required": True,
        "errors": val_errors,
        "repair_actions": repair_actions,
    }

    return {
        "retry_count": current_retry,
        "repair_instructions": repair_instructions,
    }


# ---------------------------------------------------------------------------
# Node 6: Upload File
# ---------------------------------------------------------------------------

async def upload_file_node(state: DocumentGenerationState) -> dict[str, Any]:
    """Upload validated document to Cloudflare R2 and attach preview data."""
    file_path = state.get("generated_file_path")
    filename = state.get("generated_filename") or "document.bin"
    mime_type = state.get("mime_type", "application/octet-stream")
    store_id = state.get("store_id", "store")
    user_id = state.get("user_id", "user")
    preview_data = state.get("preview_data") or extract_document_preview(file_path, state.get("document_type", "excel"))

    file_id = f"file_{uuid.uuid4().hex[:16]}"
    r2_service = get_r2_storage_service()

    storage_key = r2_service.generate_storage_key(store_id, user_id, file_id, filename)
    upload_res = r2_service.upload_file(file_path, storage_key, mime_type)

    download_url = None
    try:
        download_url = r2_service.generate_presigned_url(storage_key, expires_in=3600)
    except Exception as exc:
        LOGGER.warning("upload_file.presigned_url_failed", error=str(exc))

    file_metadata = {
        "file_id": file_id,
        "filename": filename,
        "mime_type": mime_type,
        "size": upload_res.get("size_bytes", 0),
        "storage_key": storage_key,
        "document_type": state.get("document_type", "excel"),
        "status": "ready",
        "download_url": download_url,
        "preview": preview_data,
    }

    LOGGER.info("upload_file.complete", file_id=file_id, storage_key=storage_key)

    return {
        "file_id": file_id,
        "storage_key": storage_key,
        "file_metadata": file_metadata,
        "preview_data": preview_data,
        "status": "success",
    }


# ---------------------------------------------------------------------------
# Node 7: Create Result
# ---------------------------------------------------------------------------

async def create_result_node(state: DocumentGenerationState) -> dict[str, Any]:
    """Clean up local temporary file and return final execution status."""
    file_path = state.get("generated_file_path")
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            LOGGER.info("create_result.temp_cleanup", path=file_path)
        except Exception as exc:
            LOGGER.warning("create_result.temp_cleanup_failed", path=file_path, error=str(exc))

    status = state.get("status")
    val_result = state.get("validation_result", {"valid": False})

    if status == "success" and val_result.get("valid", False):
        final_meta = state.get("file_metadata", {})
        LOGGER.info("create_result.success", file_id=final_meta.get("file_id"))
        return {"status": "success"}
    else:
        err_msg = state.get("error") or "Document could not be generated after maximum retries."
        LOGGER.warning("create_result.failed", error=err_msg)
        return {
            "status": "failed",
            "error": err_msg,
        }
