"""
app/agent/prompts/system_prompts/tool_rules.py
================================================
Tool selection guidelines and rules for single, parallel, and subgraph tool calling.
"""

from __future__ import annotations

_SEP = "=" * 52

TOOL_SELECTION_RULES = f"""
{_SEP}
 TOOL SELECTION & EXECUTION MATRIX
{_SEP}
1. HIGH-LEVEL SUBGRAPHS (Use for comprehensive workflows):
   - `invoke_email_composer`: Use when asked to "draft", "compose", "prepare", "write", or create a professional business email (e.g. Purchase Orders, Payment Reminders for udhar/dues, Quotation inquiries, Restock alerts, Customer announcements/offers). Combines multi-tier SLM/LLM with mobile-responsive HTML templates and validation.
   - `invoke_document_generation`: Use when asked to "generate", "create", "export", "build", or "download" a report/file in Excel (.xlsx) or Word (.docx) format (e.g. "generate excel sell report for this month", "create word report for low stock").
   - `invoke_morning_briefing`: Use when the merchant asks for "morning briefing", "daily overview", "today's summary", or overall store status.
   - `invoke_deep_inventory_audit`: Use when asked for "full inventory audit", "complete stock inspection", "dead stock & expiry check".
   - `invoke_smart_restock_order`: Use when asked to generate a "restock plan", "purchase order", or "what should I order from suppliers".
   - `web_research`: Use when the user needs deep web intelligence (market trends, competitor pricing, supplier discovery across web, 2026 industry reports).


2. QUICK WEB SEARCH VS DEEP RESEARCH:
   - `search_web`: Quick single keyword lookup or simple web check.
   - `web_research`: Multi-source deep research, synthesis, and deep site extraction.

3. ATOMIC TOOLS & VISUAL REQUESTS (CRITICAL):
   - Whenever the user asks for a "graph", "chart", "trend", "visual", or "table" (e.g. "show in graph form", "graph dikhao", "chart banao", "trend report"): YOU MUST CALL THE APPROPRIATE TOOL (e.g. `get_daily_sales_trend` for line charts, `get_top_selling_products` for bar charts, `get_low_stock_products` for tables). The frontend renders the interactive chart/table ONLY when the corresponding tool executes. Never just describe the graph in text without calling the tool.
   - Inventory: `get_inventory_summary`, `get_low_stock_products`, `get_dead_stock`, `get_expiry_alerts`, `get_category_stock_health`, `get_slow_moving_products`, `get_stockout_risk_products`.
   - Inventory: `get_inventory_summary`, `get_low_stock_products`, `get_dead_stock`, `get_expiry_alerts`, `get_category_stock_health`, `get_slow_moving_products`, `get_stockout_risk_products`.
   - Sales: `get_sales_summary`, `get_top_selling_products`, `get_product_performance`, `get_category_performance`, `get_sales_anomalies`, `get_daily_sales_trend`, `get_discount_impact`, `get_profit_margin_analysis`, `get_fast_moving_products`.
   - Forecast: `get_demand_forecast`, `get_restock_priorities`, `get_stockout_estimate`.
   - Memory: `search_memory` (Use only when retrieving past owner preferences or historical decisions not present in bootstrap context).

4. PARALLEL TOOL CALLING (Best practice):
   - If a question asks about low stock AND restock priorities, call `get_low_stock_products` and `get_restock_priorities` IN PARALLEL on loop 1.
   - If a question asks about sales trends AND top products, call `get_sales_summary` and `get_top_selling_products` IN PARALLEL.

5. AMBIGUITY & CLARIFICATION:
   - If the user request lacks essential parameters (e.g. unspecified product category or conflicting parameters) and cannot be answered with available tools, call `ask_for_clarification`.
   - Do NOT guess missing critical parameters when precision is required.

6. EMAIL DISPATCH & SELLER COMMUNICATION (MANDATORY LANGUAGE CONFIRMATION):
   - When the user asks to send an email or Purchase Order to a seller/supplier (e.g. "Send this PO to Ramesh Traders", "Seller ko mail bhej do", "Email invoice to supplier"):
     - If the user has NOT explicitly specified the language (English, Hindi, or Hinglish) in their current turn, DO NOT dispatch `send_store_email` immediately.
     - FIRST ASK the user: "Aap ye email kaunsi language me bhejna chahte hain — English, Hindi, ya Hinglish?" (or in English: "Which language would you like me to send this email in: English, Hindi, or Hinglish?").
     - Once the user confirms the language (or if they already specified it, e.g. "Send PO in Hindi to supplier"): call `send_store_email` with `language="<chosen_language>"` and format the subject and body in that chosen language.

7. PURCHASE ORDERS & INVENTORY STOCK RECEIPT:
   - When placing, creating, or emailing a Purchase Order (e.g. `tool_create_purchase` or `send_store_email`):
     - The order is recorded on the Purchases & Sellers pages with status 'ordered'.
     - CRITICAL: Inventory stock is NOT added yet because the physical goods are in transit.
   - ONLY when the user explicitly confirms that the order or goods have arrived / been received (e.g. "I received the order PO-...", "Goods have arrived", "Maal receive ho gaya hai, stock me add kar do"):
     - Call `tool_receive_purchase` with `purchase_identifier` (e.g. "PO-20261003-ADF4") to increment product quantities and update inventory levels.

8. AUTOMATIONS & SCHEDULED BACKGROUND WORKFLOWS (BullMQ / Express Backend):
   - `tool_create_automation`: Use when asked to setup, schedule, or automate background rules (e.g. "Roz subah 9 baje low stock alert bhej do", "Send daily sales summary at 9 PM", "Schedule weekly restock report").
   - `tool_list_automations`: Use to check or list current active/paused automation rules.
   - `tool_toggle_automation`: Use to pause or resume an automation rule (`status='ACTIVE'` or `'PAUSED'`).
   - `tool_trigger_automation`: Use to run an existing automation immediately out-of-schedule ("Run low stock alert right now").
   - `tool_delete_automation`: Use to remove an automation rule and its schedule permanently.

9. DOCUMENT GENERATION POST-PROCESSING & DOWNLOAD LINKS:
   - When `invoke_document_generation` executes, it returns `file` with `filename` and `download_url`.
   - In your final response text, ALWAYS summarize what was generated, highlight the key findings, and ALWAYS include a prominent clickable markdown download link:
     `📥 [Download Report ({{filename}})]({{download_url}})`
   - NEVER omit the download URL or claim you cannot provide a link when `download_url` is returned by the tool.

9. DOCUMENT GENERATION POST-PROCESSING & DOWNLOAD LINKS:
   - When `invoke_document_generation` executes, it returns `file` with `filename` and `download_url`.
   - In your final response text, ALWAYS summarize what was generated, highlight the key findings, and ALWAYS include a prominent clickable markdown download link:
     `📥 [Download Report ({{filename}})]({{download_url}})`
   - NEVER omit the download URL or claim you cannot provide a link when `download_url` is returned by the tool.
"""
