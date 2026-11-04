"""
app/agent/subgraphs/document_generation/sandbox.py
==================================================
Isolated subprocess sandbox for running LLM-generated document scripts
(openpyxl for Excel, python-docx for Word) with strict memory & execution limits.

Resource Constraints:
- Memory Cap: ~50 MB RAM per execution (critical for Render 512MB free tier)
- Timeout: ~20.0 seconds
- Security: AST validation against unauthorized imports and dangerous builtins
- Output: Validated file on disk + structured preview metadata for UI
"""

from __future__ import annotations

import ast
import asyncio
import os
import sys
import tempfile
from typing import Any, Optional
import structlog

LOGGER = structlog.get_logger("vyaparsathi.ai.doc_gen.sandbox")

# Security Allowlist: Modules permitted in document generation scripts
ALLOWED_ROOT_MODULES = {
    "openpyxl",
    "docx",
    "pandas",
    "math",
    "datetime",
    "json",
    "random",
    "re",
    "decimal",
    "collections",
    "itertools",
    "string",
}

# Forbidden function calls and globals that could violate isolation
FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "globals",
    "locals",
    "vars",
    "open",
    "breakpoint",
    "exit",
    "quit",
}

# Forbidden internal dunder attributes
FORBIDDEN_ATTRIBUTES = {
    "__subclasses__",
    "__globals__",
    "__code__",
    "__bases__",
    "__mro__",
    "__builtins__",
}


def validate_code_ast(code: str) -> tuple[bool, Optional[str]]:
    """
    Statically inspect Python code using AST to ensure it does not import
    unauthorized modules (e.g. os, sys, subprocess, socket, urllib) or call
    dangerous builtins.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as syn_err:
        return False, f"Syntax error in generated script: {syn_err.msg} (line {syn_err.lineno})"

    for node in ast.walk(tree):
        # 1. Check `import <name>`
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg not in ALLOWED_ROOT_MODULES:
                    return False, f"Module '{alias.name}' is not allowed in document sandbox."

        # 2. Check `from <module> import <name>`
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_pkg = node.module.split(".")[0]
                if root_pkg not in ALLOWED_ROOT_MODULES:
                    return False, f"Importing from '{node.module}' is not allowed in document sandbox."

        # 3. Check function calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_CALLS:
                    return False, f"Call to '{node.func.id}()' is forbidden in document sandbox."

        # 4. Check attribute access
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRIBUTES:
                return False, f"Accessing '{node.attr}' is forbidden in document sandbox."

    return True, None


def _get_preexec_limits(max_memory_mb: int):
    """
    Build platform-specific resource limit function for subprocess execution.
    On Linux (Render.com), enforces strict virtual memory (RLIMIT_AS) and heap (RLIMIT_DATA).
    On Windows, returns None since resource is POSIX only.
    """
    if sys.platform == "win32":
        return None

    def _set_limits():
        try:
            import resource
            bytes_limit = max_memory_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
            except Exception:
                pass
            try:
                resource.setrlimit(resource.RLIMIT_DATA, (bytes_limit, bytes_limit))
            except Exception:
                pass
        except ImportError:
            pass

    return _set_limits


async def execute_document_code(
    code: str,
    output_path: str,
    timeout: float = 20.0,
    max_memory_mb: int = 50,
) -> dict[str, Any]:
    """
    Execute user/LLM generated Python document script in a strictly resource-capped subprocess.

    Args:
        code: The Python script body to execute.
        output_path: Target file path where the generated document (.xlsx / .docx) must be written.
        timeout: Maximum execution time in seconds (default 20.0s).
        max_memory_mb: Maximum memory in MB (default 50MB, tuned for Render 512MB RAM free tier).

    Returns:
        dict with keys:
            success: bool
            output_path: str (if success)
            file_size: int (if success)
            stdout: str
            stderr: str
            error: str | None
    """
    # 1. AST Security screening
    is_safe, sec_err = validate_code_ast(code)
    if not is_safe:
        LOGGER.warning("sandbox.ast_security_violation", error=sec_err)
        return {
            "success": False,
            "error": f"Security restriction: {sec_err}",
            "stdout": "",
            "stderr": sec_err or "",
        }

    # Ensure target parent directory exists
    target_dir = os.path.dirname(output_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    # 2. Prepare script wrapper
    # Injects OUTPUT_PATH variable at the top
    escaped_output_path = output_path.replace("\\", "\\\\")
    script_header = (
        f"# -*- coding: utf-8 -*-\n"
        f"OUTPUT_PATH = r\"{escaped_output_path}\"\n\n"
    )
    full_script = script_header + code

    with tempfile.TemporaryDirectory(prefix="doc_sandbox_") as tmp_dir:
        script_file = os.path.join(tmp_dir, "generator.py")
        with open(script_file, "w", encoding="utf-8") as f:
            f.write(full_script)

        preexec_fn = _get_preexec_limits(max_memory_mb)

        LOGGER.info(
            "sandbox.launch_process",
            output_path=output_path,
            timeout=timeout,
            memory_limit_mb=max_memory_mb,
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                script_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=preexec_fn,
                cwd=tmp_dir,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
                stdout_str = stdout_bytes.decode("utf-8", errors="replace")
                stderr_str = stderr_bytes.decode("utf-8", errors="replace")
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
                msg = f"Execution timed out after {timeout:.1f} seconds. (Render free tier timeout protection)"
                LOGGER.warning("sandbox.timeout", error=msg)
                return {
                    "success": False,
                    "error": msg,
                    "stdout": "",
                    "stderr": "TimeoutError",
                }

            if proc.returncode != 0:
                # Check for OOM / kill signal on Linux
                if proc.returncode in (-9, -11) or "MemoryError" in stderr_str:
                    err_msg = (
                        f"Script exceeded the {max_memory_mb}MB RAM limit. "
                        "Keep dataset structures compact to fit Render free tier resources."
                    )
                else:
                    err_msg = f"Script execution failed (exit code {proc.returncode}):\n{stderr_str.strip()}"

                LOGGER.warning("sandbox.exec_failed", exit_code=proc.returncode, error=err_msg[:300])
                return {
                    "success": False,
                    "error": err_msg,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                }

            # Verify that output file was written and is not empty.
            # If the script saved to a local relative file instead of OUTPUT_PATH, detect and rescue it.
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                rescued_file = None
                for candidate in os.listdir(tmp_dir):
                    if candidate != "generator.py" and (candidate.endswith(".xlsx") or candidate.endswith(".docx")):
                        c_path = os.path.join(tmp_dir, candidate)
                        if os.path.isfile(c_path) and os.path.getsize(c_path) > 0:
                            rescued_file = c_path
                            break

                if rescued_file:
                    import shutil
                    shutil.copyfile(rescued_file, output_path)
                    LOGGER.info("sandbox.rescued_relative_save", rescued_from=rescued_file, output_path=output_path)
                else:
                    err_msg = f"Script completed with exit code 0 but did not save the file to OUTPUT_PATH ('{output_path}')."
                    LOGGER.warning("sandbox.missing_output_file", path=output_path)
                    return {
                        "success": False,
                        "error": err_msg,
                        "stdout": stdout_str,
                        "stderr": stderr_str,
                    }

            file_size = os.path.getsize(output_path)
            LOGGER.info("sandbox.exec_success", output_path=output_path, size_bytes=file_size)

            return {
                "success": True,
                "output_path": output_path,
                "file_size": file_size,
                "stdout": stdout_str,
                "stderr": stderr_str,
                "error": None,
            }

        except Exception as exc:
            LOGGER.error("sandbox.subprocess_error", error=str(exc), exc_info=True)
            return {
                "success": False,
                "error": f"Failed to launch sandbox subprocess: {str(exc)}",
                "stdout": "",
                "stderr": str(exc),
            }


def extract_document_preview(file_path: str, doc_type: str = "excel") -> dict[str, Any]:
    """
    Inspect a generated document (.xlsx or .docx) and extract rich preview metadata
    for inline card viewing and full-page interactive preview in the UI.

    For Excel:
        - Sheet names
        - Active sheet data (headers, up to 200 rows with types/formatted values)
        - Column counts and row counts
        - Quick KPI cards if identified from summary rows or cards

    For Word:
        - Title & Subtitle
        - Section headings and levels
        - Text paragraphs
        - Structured data tables
        - Total word count / paragraph count
    """
    if not os.path.exists(file_path):
        return {"error": "File does not exist on disk", "type": doc_type}

    doc_type = doc_type.lower()
    file_size = os.path.getsize(file_path)
    filename = os.path.basename(file_path)

    if doc_type == "excel" or file_path.endswith(".xlsx"):
        return _extract_excel_preview(file_path, filename, file_size)
    else:
        return _extract_word_preview(file_path, filename, file_size)


def _extract_excel_preview(file_path: str, filename: str, file_size: int) -> dict[str, Any]:
    """Extract Excel sheets, columns, rows and summary stats."""
    import openpyxl

    try:
        # Load workbook with data_only=True so formulas return their computed or stored values
        wb = openpyxl.load_workbook(file_path, data_only=True)
        sheets_data = []

        total_rows_all_sheets = 0

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            max_r = ws.max_row or 0
            max_c = ws.max_column or 0

            # Scan rows to find headers and data
            raw_rows = []
            for r_idx in range(1, min(max_r + 1, 201)):  # Capture up to 200 rows for full page preview
                row_vals = []
                has_any_val = False
                for c_idx in range(1, max_c + 1):
                    val = ws.cell(row=r_idx, column=c_idx).value
                    if val is not None:
                        has_any_val = True
                        if isinstance(val, float):
                            val = round(val, 2)
                    row_vals.append(val if val is not None else "")
                if has_any_val:
                    raw_rows.append(row_vals)

            if not raw_rows:
                sheets_data.append({
                    "name": sheet_name,
                    "columns": ["Empty Sheet"],
                    "rows": [],
                    "total_rows": 0,
                    "total_columns": 0,
                })
                continue

            # Heuristic to detect header: first row with multiple non-empty string cells
            header_row_idx = 0
            for idx, r in enumerate(raw_rows):
                str_count = sum(1 for cell in r if isinstance(cell, str) and cell.strip())
                if str_count >= 1:
                    header_row_idx = idx
                    break

            headers = [
                str(c).strip() if str(c).strip() else f"Col {i+1}"
                for i, c in enumerate(raw_rows[header_row_idx])
            ]
            data_rows = raw_rows[header_row_idx + 1:]

            # Normalize row lengths to match header count
            cleaned_rows = []
            for r in data_rows:
                padded = r[:len(headers)] + [""] * max(0, len(headers) - len(r))
                cleaned_rows.append(padded)

            total_rows_all_sheets += len(cleaned_rows)
            sheets_data.append({
                "name": sheet_name,
                "columns": headers,
                "rows": cleaned_rows,
                "total_rows": max_r,
                "total_columns": max_c,
            })

        first_sheet = sheets_data[0] if sheets_data else None

        return {
            "type": "excel",
            "filename": filename,
            "file_size": file_size,
            "sheet_count": len(sheets_data),
            "sheet_names": [s["name"] for s in sheets_data],
            "active_sheet": sheets_data[0]["name"] if sheets_data else "Sheet1",
            "sheets": sheets_data,
            # Top-level quick access for default view
            "columns": first_sheet["columns"] if first_sheet else [],
            "rows": first_sheet["rows"] if first_sheet else [],
            "total_rows": total_rows_all_sheets,
        }

    except Exception as exc:
        LOGGER.warning("extract_document_preview.excel_failed", error=str(exc))
        return {
            "type": "excel",
            "filename": filename,
            "file_size": file_size,
            "sheet_count": 1,
            "sheet_names": ["Sheet1"],
            "columns": ["Status"],
            "rows": [["Excel document generated successfully"]],
            "error": str(exc),
        }


def _extract_word_preview(file_path: str, filename: str, file_size: int) -> dict[str, Any]:
    """Extract Word document sections, headings, paragraphs, and tables."""
    import docx

    try:
        doc = docx.Document(file_path)

        sections = []
        current_section = {
            "heading": "Overview",
            "level": 1,
            "paragraphs": [],
            "tables": [],
        }

        total_words = 0
        total_paragraphs = len(doc.paragraphs)

        # Title detection
        doc_title = filename.replace(".docx", "").replace("_", " ").title()
        doc_subtitle = ""

        if doc.paragraphs:
            first_p = doc.paragraphs[0].text.strip()
            if first_p:
                doc_title = first_p
            if len(doc.paragraphs) > 1 and not doc.paragraphs[1].style.name.startswith("Heading"):
                second_p = doc.paragraphs[1].text.strip()
                if second_p and len(second_p) < 120:
                    doc_subtitle = second_p

        for p in doc.paragraphs:
            text = p.text.strip()
            if not text:
                continue
            total_words += len(text.split())

            style_name = p.style.name.lower()
            if "heading 1" in style_name:
                if current_section["paragraphs"] or current_section["tables"]:
                    sections.append(current_section)
                current_section = {
                    "heading": text,
                    "level": 1,
                    "paragraphs": [],
                    "tables": [],
                }
            elif "heading 2" in style_name:
                if current_section["paragraphs"] or current_section["tables"]:
                    sections.append(current_section)
                current_section = {
                    "heading": text,
                    "level": 2,
                    "paragraphs": [],
                    "tables": [],
                }
            elif "heading" in style_name:
                if current_section["paragraphs"] or current_section["tables"]:
                    sections.append(current_section)
                current_section = {
                    "heading": text,
                    "level": 3,
                    "paragraphs": [],
                    "tables": [],
                }
            else:
                if text != doc_title and text != doc_subtitle:
                    current_section["paragraphs"].append(text)

        # Extract tables
        extracted_tables = []
        for tbl in doc.tables:
            tbl_rows = []
            for row in tbl.rows:
                tbl_rows.append([cell.text.strip() for cell in row.cells])
            if tbl_rows:
                headers = tbl_rows[0]
                rows = tbl_rows[1:] if len(tbl_rows) > 1 else []
                extracted_tables.append({
                    "headers": headers,
                    "rows": rows,
                })

        if extracted_tables:
            current_section["tables"].extend(extracted_tables)

        if current_section["paragraphs"] or current_section["tables"] or not sections:
            sections.append(current_section)

        return {
            "type": "word",
            "filename": filename,
            "file_size": file_size,
            "title": doc_title,
            "subtitle": doc_subtitle,
            "total_words": total_words,
            "total_paragraphs": total_paragraphs,
            "total_tables": len(extracted_tables),
            "sections": sections,
        }

    except Exception as exc:
        LOGGER.warning("extract_document_preview.word_failed", error=str(exc))
        return {
            "type": "word",
            "filename": filename,
            "file_size": file_size,
            "title": filename.replace(".docx", "").replace("_", " ").title(),
            "subtitle": "Word Document",
            "sections": [
                {
                    "heading": "Document Content",
                    "level": 1,
                    "paragraphs": ["Word document generated successfully."],
                    "tables": [],
                }
            ],
            "error": str(exc),
        }
