"""
app/agent/tools/visual_builders.py
==================================
Converts raw analytical tool results into validated VisualSpec dictionaries
(charts, tables, KPI cards) to stream to Next.js frontend via SSE `event: visual`.
"""

from __future__ import annotations
from typing import Any, Optional
import json


def _extract_dict(obj: Any) -> Optional[dict]:
    """Helper to reliably extract a python dict from strings, Pydantic models, or ToolMessages."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return obj.model_dump()
    if hasattr(obj, "dict") and callable(obj.dict):
        return obj.dict()
    if hasattr(obj, "content"):
        return _extract_dict(obj.content)
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return None


def build_visual_from_tool(tool_name: str, result: Any) -> Optional[dict]:
    """
    Given a tool name and its executed result, produce a validated VisualSpec
    dictionary or None if the tool does not yield a visual.
    """
    data = _extract_dict(result)
    if not data or not isinstance(data, dict):
        return None

    # 1. Daily Sales Trend / Sales Over Time -> Line Chart
    if tool_name in ("get_daily_sales_trend", "get_sales_over_time"):
        days_list = data.get("days") or []
        rev_list = data.get("revenue") or []
        tx_list = data.get("transactions") or []
        trend_dir = str(data.get("trend", "flat")).upper()
        avg_rev = float(data.get("avg_daily_revenue", 0.0))

        if isinstance(days_list, list) and isinstance(rev_list, list) and len(days_list) > 0:
            rows = []
            for i, d in enumerate(days_list):
                # Shorten date format like "2026-10-09" -> "09 Oct" or keep clean
                date_str = str(d)
                try:
                    parts = date_str.split("-")
                    if len(parts) == 3:
                        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                        m_idx = int(parts[1]) - 1
                        date_str = f"{parts[2]} {months[m_idx]}"
                except Exception:
                    date_str = str(d)

                r = float(rev_list[i]) if i < len(rev_list) else 0.0
                tx = int(tx_list[i]) if i < len(tx_list) else 0
                rows.append({
                    "date": date_str,
                    "sales": r,
                    "orders": tx,
                })

            return {
                "type": "line_chart",
                "title": f"Daily Sales Trend ({trend_dir})",
                "description": f"Past {len(rows)} days • Avg daily revenue ₹{avg_rev:.0f}",
                "xKey": "date",
                "series": [
                    {"key": "sales", "label": "Revenue", "color": "#3b82f6"},
                ],
                "data": rows,
                "valueFormat": "currency_inr",
            }

    # 2. Top Selling Products -> Bar Chart
    if tool_name in ("get_top_selling_products", "tool_top_selling"):
        items = data.get("items") or data.get("top_products") or data.get("products") or []
        if isinstance(items, list) and len(items) > 0:
            rows = []
            for item in items[:10]:
                p = _extract_dict(item) or item
                if isinstance(p, dict):
                    name = str(p.get("name", "Product"))[:18]
                    qty = float(p.get("total_quantity_sold", p.get("quantity", p.get("units_sold", 0))))
                    rev = float(p.get("revenue_generated", p.get("revenue", p.get("total_sales", 0))))
                    rows.append({
                        "name": name,
                        "units": qty,
                        "revenue": rev,
                    })
            if rows:
                return {
                    "type": "bar_chart",
                    "title": "Top Selling Products",
                    "description": "Top items ranked by sales revenue and quantity",
                    "xKey": "name",
                    "series": [
                        {"key": "revenue", "label": "Revenue", "color": "#6366f1"},
                        {"key": "units", "label": "Units", "color": "#10b981"},
                    ],
                    "data": rows,
                    "valueFormat": "currency_inr",
                }

    # 3. Low Stock Items -> Data Table with status badges
    if tool_name in ("get_low_stock_products", "tool_low_stock"):
        items = data.get("items") or data.get("low_stock_products") or data.get("products") or []
        if isinstance(items, list) and len(items) > 0:
            rows = []
            for item in items[:50]:
                p = _extract_dict(item) or item
                if isinstance(p, dict):
                    qty = float(p.get("current_quantity", p.get("stock", p.get("quantity", 0))))
                    status = "Out of Stock" if qty <= 0 else "Low Stock"
                    rows.append({
                        "product": str(p.get("name", "Product")),
                        "category": str(p.get("category", "General")),
                        "stock": qty,
                        "price": float(p.get("price", 0)),
                        "status": status,
                    })
            if rows:
                return {
                    "type": "table",
                    "title": f"Low Stock Inventory ({len(rows)} Items)",
                    "columns": [
                        {"key": "product", "label": "Product", "dataType": "string"},
                        {"key": "category", "label": "Category", "dataType": "string"},
                        {"key": "stock", "label": "Stock", "dataType": "number", "sortable": True},
                        {"key": "price", "label": "Price", "dataType": "currency_inr", "sortable": True},
                        {"key": "status", "label": "Status", "dataType": "badge"},
                    ],
                    "rows": rows,
                    "pageSize": 5,
                }

    # 4. Inventory Summary -> KPI Cards
    if tool_name in ("get_inventory_summary", "get_store_insights"):
        total_products = data.get("total_products")
        total_valuation = data.get("total_inventory_value")
        low_stock_count = data.get("low_stock_count")
        out_of_stock_count = data.get("out_of_stock_count")

        cards = []
        if total_valuation is not None:
            cards.append({
                "label": "Inventory Valuation",
                "value": float(total_valuation),
                "trend": "up",
                "valueFormat": "currency_inr",
            })
        if total_products is not None:
            cards.append({
                "label": "Total Products",
                "value": int(total_products),
                "trend": "flat",
                "valueFormat": "number",
            })
        if low_stock_count is not None:
            cards.append({
                "label": "Low Stock Alerts",
                "value": int(low_stock_count),
                "trend": "down" if int(low_stock_count) > 0 else "flat",
                "valueFormat": "number",
            })
        if out_of_stock_count is not None:
            cards.append({
                "label": "Out of Stock",
                "value": int(out_of_stock_count),
                "trend": "down" if int(out_of_stock_count) > 0 else "flat",
                "valueFormat": "number",
            })

        if cards:
            return {
                "type": "kpi_cards",
                "cards": cards,
            }

    # 5. Sales Summary -> KPI Cards
    if tool_name in ("get_sales_summary",):
        total_rev = data.get("total_revenue")
        total_sales = data.get("total_sales_count")
        aov = data.get("average_order_value")
        days = data.get("days_covered", 30)

        cards = []
        if total_rev is not None:
            cards.append({
                "label": f"Revenue ({days}d)",
                "value": float(total_rev),
                "trend": "up",
                "valueFormat": "currency_inr",
            })
        if total_sales is not None:
            cards.append({
                "label": "Total Orders",
                "value": int(total_sales),
                "trend": "flat",
                "valueFormat": "number",
            })
        if aov is not None:
            cards.append({
                "label": "Avg Order Value",
                "value": float(aov),
                "trend": "flat",
                "valueFormat": "currency_inr",
            })

        if cards:
            return {
                "type": "kpi_cards",
                "cards": cards,
            }

    # 6. Category Performance -> Pie Chart
    if tool_name in ("get_category_performance", "get_category_stock_health"):
        categories = data.get("categories") or []
        if isinstance(categories, list) and len(categories) > 0:
            pie_data = []
            for item in categories[:8]:
                c = _extract_dict(item) or item
                if isinstance(c, dict):
                    val = float(c.get("revenue", c.get("value", c.get("stock_count", 0))))
                    if val > 0:
                        pie_data.append({
                            "label": str(c.get("category", c.get("name", "Other"))),
                            "value": val,
                        })
            if pie_data:
                return {
                    "type": "pie_chart",
                    "title": "Category-wise Performance",
                    "data": pie_data,
                    "valueFormat": "currency_inr",
                }

    # 7. Expiry Alerts -> Table
    if tool_name in ("get_expiry_alerts", "tool_expiring_soon"):
        expired = data.get("expired") or []
        critical = data.get("critical") or []
        warning = data.get("warning") or []
        all_items = []
        for x in critical:
            d = _extract_dict(x) or x
            if isinstance(d, dict):
                all_items.append({**d, "urgency": "Critical (≤7d)"})
        for x in warning:
            d = _extract_dict(x) or x
            if isinstance(d, dict):
                all_items.append({**d, "urgency": "Expiring Soon"})
        for x in expired:
            d = _extract_dict(x) or x
            if isinstance(d, dict):
                all_items.append({**d, "urgency": "Expired"})

        if all_items:
            rows = []
            for item in all_items[:50]:
                rows.append({
                    "product": str(item.get("name", "Product")),
                    "category": str(item.get("category", "General")),
                    "stock": float(item.get("quantity", 0)),
                    "expiry": str(item.get("exp_date", "")),
                    "days": int(item.get("days_left", 0)),
                    "status": item.get("urgency", "Warning"),
                })
            return {
                "type": "table",
                "title": f"Expiry Alerts ({len(rows)} Products)",
                "columns": [
                    {"key": "product", "label": "Product", "dataType": "string"},
                    {"key": "category", "label": "Category", "dataType": "string"},
                    {"key": "stock", "label": "Stock", "dataType": "number"},
                    {"key": "expiry", "label": "Expiry Date", "dataType": "date"},
                    {"key": "days", "label": "Days Left", "dataType": "number", "sortable": True},
                    {"key": "status", "label": "Urgency", "dataType": "badge"},
                ],
                "rows": rows,
                "pageSize": 5,
            }

    # 8. Document Generation -> Interactive Document Preview Spec
    if tool_name in ("invoke_document_generation", "tool_document_generation"):
        file_meta = data.get("file") if isinstance(data.get("file"), dict) else {}
        preview_data = data.get("preview") or file_meta.get("preview") or {}
        doc_type = data.get("document_type") or file_meta.get("document_type") or "excel"
        filename = file_meta.get("filename") or f"report.{'xlsx' if doc_type == 'excel' else 'docx'}"
        download_url = file_meta.get("download_url")

        return {
            "type": "document_preview",
            "documentType": doc_type,
            "filename": filename,
            "fileId": file_meta.get("file_id", ""),
            "fileSize": file_meta.get("size", 0),
            "downloadUrl": download_url,
            "storageKey": file_meta.get("storage_key", ""),
            "preview": preview_data if isinstance(preview_data, dict) else {},
        }

    return None


# Backward-compatible alias
build_visual_for_tool = build_visual_from_tool
