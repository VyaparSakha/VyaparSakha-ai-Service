"""
tests/test_sandbox.py
======================
Unit tests for the AST security screening and sandboxed document execution.
"""

import os
import pytest
from app.agent.subgraphs.document_generation.sandbox import (
    validate_code_ast,
    execute_document_code,
    extract_document_preview,
)


def test_validate_code_ast_disallowed_modules():
    code_os = "import os\nos.remove('file.txt')"
    safe, err = validate_code_ast(code_os)
    assert not safe
    assert "Module 'os' is not allowed" in err

    code_sys = "import sys\nsys.exit(0)"
    safe, err = validate_code_ast(code_sys)
    assert not safe
    assert "Module 'sys' is not allowed" in err

    code_subprocess = "import subprocess\nsubprocess.run(['ls'])"
    safe, err = validate_code_ast(code_subprocess)
    assert not safe
    assert "subprocess" in err

    code_from_shutil = "from shutil import rmtree\nrmtree('.')"
    safe, err = validate_code_ast(code_from_shutil)
    assert not safe
    assert "shutil" in err


def test_validate_code_ast_disallowed_calls_and_attributes():
    code_eval = "x = eval('2 + 2')"
    safe, err = validate_code_ast(code_eval)
    assert not safe
    assert "eval" in err

    code_open = "with open('file.txt', 'w') as f: f.write('hi')"
    safe, err = validate_code_ast(code_open)
    assert not safe
    assert "open" in err

    code_subclasses = "x = ().__class__.__subclasses__()"
    safe, err = validate_code_ast(code_subclasses)
    assert not safe
    assert "__subclasses__" in err


def test_validate_code_ast_allowed_code():
    code_valid = """
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
import math
from datetime import datetime
import json
import re

wb = openpyxl.Workbook()
ws = wb.active
ws['A1'] = 'Test'
wb.save(OUTPUT_PATH)
"""
    safe, err = validate_code_ast(code_valid)
    assert safe
    assert err is None


@pytest.mark.asyncio
async def test_execute_document_code_excel_success(tmp_path):
    output_file = str(tmp_path / "test_report.xlsx")
    code = """
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Sales Summary"

ws['A1'] = "Vyapar Sakha Executive Report"
ws['A1'].font = Font(size=14, bold=True)

headers = ["Product", "Category", "Units Sold", "Unit Price", "Total Revenue"]
ws.append([])
ws.append(headers)

rows = [
    ["Basmati Rice 5kg", "Grains", 20, 450, "=C4*D4"],
    ["Sunflower Oil 1L", "Oils", 35, 140, "=C5*D5"],
    ["Wheat Flour 10kg", "Grains", 15, 380, "=C6*D6"],
]
for r in rows:
    ws.append(r)

wb.save(OUTPUT_PATH)
"""
    res = await execute_document_code(code, output_file, timeout=20.0, max_memory_mb=50)
    assert res["success"] is True
    assert os.path.exists(output_file)
    assert res["file_size"] > 0

    # Test preview extraction
    preview = extract_document_preview(output_file, "excel")
    assert preview["type"] == "excel"
    assert "Sales Summary" in preview["sheet_names"]
    assert len(preview["sheets"]) >= 1
    assert len(preview["columns"]) >= 1
    assert len(preview["rows"]) >= 1


@pytest.mark.asyncio
async def test_execute_document_code_word_success(tmp_path):
    output_file = str(tmp_path / "test_report.docx")
    code = """
import docx

doc = docx.Document()
doc.add_heading("Store Performance Audit", 0)

doc.add_heading("Executive Summary", level=1)
doc.add_paragraph("Store operations performed remarkably this month with healthy profit margins.")

doc.add_heading("Key Financials", level=2)
table = doc.add_table(rows=1, cols=2)
hdr = table.rows[0].cells
hdr[0].text = "Metric"
hdr[1].text = "Value"

row1 = table.add_row().cells
row1[0].text = "Total Sales"
row1[1].text = "₹1,25,000"

doc.save(OUTPUT_PATH)
"""
    res = await execute_document_code(code, output_file, timeout=20.0, max_memory_mb=50)
    assert res["success"] is True
    assert os.path.exists(output_file)

    preview = extract_document_preview(output_file, "word")
    assert preview["type"] == "word"
    assert "Store Performance Audit" in preview["title"]
    assert len(preview["sections"]) >= 1
    assert preview["total_tables"] >= 1


@pytest.mark.asyncio
async def test_execute_document_code_word_with_llm_shims(tmp_path):
    """Verify that docx compatibility shims (color.rgb, get_or_add_tcBorders, set_cell_background) prevent crashes."""
    output_file = str(tmp_path / "test_shims_report.docx")
    code = """
import docx
from docx.shared import RGBColor

doc = docx.Document()
tbl = doc.add_table(rows=1, cols=2)
cell = tbl.cell(0, 0)

# 1. Test set_cell_background helper with RGBColor
c = RGBColor(0x1E, 0x29, 0x3B)
set_cell_background(cell, c)

# 2. Test color.rgb attribute access (which previously failed with AttributeError)
formatted_hex = f"{c.rgb:06X}"
assert formatted_hex == "1E293B"

# 3. Test tcPr.get_or_add_tcBorders() (which previously failed with AttributeError)
tcPr = cell._tc.get_or_add_tcPr()
borders = tcPr.get_or_add_tcBorders()
assert borders is not None

doc.save(OUTPUT_PATH)
"""
    res = await execute_document_code(code, output_file, timeout=20.0, max_memory_mb=50)
    assert res["success"] is True, f"Execution failed with: {res.get('error')}"
    assert os.path.exists(output_file)


