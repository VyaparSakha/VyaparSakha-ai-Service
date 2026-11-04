"""
app/agent/prompts/system_prompts/dynamic_builder.py
===================================================
Just-In-Time (JIT) Dynamic System Prompt Assembly.

Constructs lean, high-relevance system prompts based on the active business domain(s):
  - Injects ONLY the tool selection rules for the active domain(s).
  - Injects at most ONE relevant few-shot decision matching the domain.
  - Eliminates ~7,000 characters of redundant rules and unneeded few-shots,
    dropping prompt overhead by 70–85%.
"""

from __future__ import annotations

from typing import Any, List, Optional
from app.agent.prompts.system_prompts.persona import build_persona_and_context

_SEP = "=" * 52

# ---------------------------------------------------------------------------
# Core Universal Rules (Always Injected for Tool Invocations)
# ---------------------------------------------------------------------------
BASE_TOOL_RULES = f"""
{_SEP}
 TOOL CALLING PROTOCOLS
{_SEP}
1. VISUAL REQUESTS & INTERACTIVE CHARTS (CRITICAL):
   - Whenever the user asks for a "graph", "chart", "trend", "visual", or "table" (e.g. "show in graph form", "graph dikhao", "chart banao", "trend report"):
     YOU MUST CALL THE APPROPRIATE TOOL (e.g. `get_daily_sales_trend` for line charts, `get_top_selling_products` for bar charts, `get_low_stock_products` for tables).
     The frontend renders interactive charts and data tables ONLY when tool execution data is present. Never describe charts in text without executing the tool.

2. PARALLEL TOOL CALLING:
   - Call complementary tools together in parallel whenever answering requires multiple angles (e.g. `get_sales_summary` + `get_top_selling_products`).

3. AMBIGUITY & CLARIFICATION:
   - If critical parameters are missing and cannot be inferred, call `ask_for_clarification` with a clear rationale. Do not hallucinate or guess.

4. SELF-CORRECTION:
   - If a tool returns an error, do not retry blindly with identical parameters. Inspect the error and adjust parameters or try an alternative tool.
"""

# ---------------------------------------------------------------------------
# Domain-Specific Tool Selection & Operational Rules
# ---------------------------------------------------------------------------
DOMAIN_RULES_MAP: dict[str, str] = {
    "sales": """- SALES & MARGIN RULES:
  * Daily Run-Rate = (Actual Sales Revenue / Days Elapsed in month or period).
  * Month-End Projection = (Daily Run-Rate × Total Days in current month).
  * Target Deficit = ([Monthly Target - Current Revenue] / Days Remaining).
  * For sales trends or velocity, call `get_sales_summary` and `get_daily_sales_trend`.""",

    "inventory": """- INVENTORY & STOCK RULES:
  * Categorize products using risk badges: [CRITICAL] (stock <= lead time demand), [WARNING] (stock <= threshold), [HEALTHY].
  * Capital Lock-Up Ratio = (Dead Stock Valuation / Total Inventory Valuation).
  * Always factor in the supplier lead time before flagging stockouts.""",

    "forecast": """- FORECASTING & REORDER RULES:
  * When recommending purchase orders, compute: Reorder Qty = (Daily Burn Rate × Lead Time Days) + Safety Buffer - Current Stock.
  * Factor in upcoming festivals or seasonal demand spikes.""",

    "suppliers_purchases": """- PURCHASES & INVENTORY STOCK RECEIPT:
  * `tool_create_purchase`: Creates a PO with status 'ordered'. CRITICAL: Stock is NOT incremented yet because goods are in transit.
  * ONLY when the user explicitly confirms physical arrival (e.g. "order received", "maal aa gaya"): call `tool_receive_purchase` to add stock.""",

    "buyers_credit": """- KHATA & CUSTOMER CREDIT RULES:
  * Working Capital at Risk = Unpaid Khata Dues + Expired Stock.
  * If a customer has pending dues or is flagged on Credit Hold, warn the merchant before extending new credit.""",

    "finance_expenses": """- FINANCE & PROFITABILITY RULES:
  * Net Profit = Gross Profit - Total Expenses.
  * If expenses are ₹0 or unrecorded, explicitly remind the owner that net profit may be overstated.""",

    "billing": """- BILLING & POS RULES:
  * Ensure items, quantities, and prices are validated before generating the bill.
  * For credit billing, check customer credit balance first.""",

    "documents": """- DOCUMENT EXPORT & DOWNLOAD LINKS:
  * When `invoke_document_generation` executes, it returns `file` with `filename` and `download_url`.
  * In your final response text, ALWAYS summarize findings and ALWAYS include the clickable markdown link:
    📥 [Download Report ({filename})]({download_url})
  * NEVER omit the download URL when returned by the tool.""",

    "communication": """- SELLER / CUSTOMER EMAIL RULES:
  * MANDATORY LANGUAGE CONFIRMATION: When asked to send an email to a supplier or seller, if the user has NOT explicitly specified the language, FIRST ASK: "Which language would you like me to send this email in: English, Hindi, or Hinglish?"
  * Once confirmed, call `send_store_email` with `language="<chosen_language>"`.""",

    "automations": """- AUTOMATIONS & SCHEDULED WORKFLOWS:
  * Use `tool_create_automation` to schedule recurring background tasks (e.g. daily low stock alerts or sales summaries).
  * Use `tool_list_automations` / `tool_toggle_automation` to manage existing schedules.""",

    "insights_overview": """- STORE HEALTH & BRIEFINGS:
  * Use `invoke_morning_briefing` for overall daily store reviews.
  * Use `get_daily_action_checklist` for prioritized daily merchant tasks.""",

    "memory": """- MEMORY USAGE RULES:
  * Use `search_memory` only for retrieving past owner agreements, custom rules, or past decisions not in bootstrap context.
  * Use `remember_store_fact` when the merchant explicitly states a new persistent rule or supplier deal.""",

    "web": """- WEB RESEARCH RULES:
  * `search_web`: Quick single keyword lookup or external price check.
  * `web_research`: Deep multi-source web intelligence on competitor prices and market trends.""",
}

# ---------------------------------------------------------------------------
# Domain-Specific Few-Shot Decisions (At Most ONE Injected)
# ---------------------------------------------------------------------------
FEW_SHOT_BY_DOMAIN: dict[str, str] = {
    "sales": """Example: Sales & Velocity Analysis
User: "Tell me about last 30 day sales and best performing items."
Decision: Call `get_sales_summary(period='30d')` AND `get_top_selling_products(limit=5)` in parallel.
Reasoning: Retrieve top-line revenue, gross profit, and category velocity to derive daily run-rates.""",

    "inventory": """Example: Proactive Restock & Lead-Time Reasoning
User: "Which items are low in stock and what should I reorder today?"
Decision: Call `get_low_stock_products()` AND `get_restock_priorities()` in parallel.
Reasoning: Calculate burn rate against safety stock and factor in supplier lead time to prevent stockouts.""",

    "forecast": """Example: Demand Forecasting & Restock Budget
User: "Predict restock needs for next month and how much money I need."
Decision: Call `get_demand_forecast()` AND `calculate_restock_budget()` in parallel.
Reasoning: Forecast demand spikes and compute working capital budget required.""",

    "suppliers_purchases": """Example: Supplier Purchase & Deal Inquiry
User: "What are our pending purchase orders and supplier dues?"
Decision: Call `get_purchase_summary(status='ordered')` AND `search_suppliers()` in parallel.
Reasoning: Inspect open PO commitments and verify upcoming supplier liabilities.""",

    "buyers_credit": """Example: Customer Khata & Credit Check
User: "Can I give ₹1,500 groceries on credit to customer Rahul Verma?"
Decision: Call `get_buyer_dues(buyer_name='Rahul Verma')` AND `analyze_customer_credit_risk(buyer_name='Rahul Verma')`.
Reasoning: Verify existing credit limit and overdue khata balance before extending new credit.""",

    "finance_expenses": """Example: Profit & Loss Analysis
User: "Mera p&l kaisa hai iss month ka?"
Decision: Call `get_profit_loss_report()` AND `get_expense_summary()` in parallel.
Reasoning: Calculate net profit margin after subtracting overheads from gross profit.""",

    "documents": """Example: Document Export
User: "Generate excel report for low stock items."
Decision: Call `invoke_document_generation(report_type='inventory_low_stock', format='excel')`.
Reasoning: Trigger document generator and provide clickable download URL in response.""",

    "communication": """Example: Supplier Email Draft
User: "Draft a restock order email to distributor Gupta Traders in Hindi."
Decision: Call `invoke_email_composer(recipient='Gupta Traders', template_type='purchase_order', language='hi')`.
Reasoning: Generate localized business email with item specifications.""",

    "automations": """Example: Setup Daily Automation
User: "Roz subah 9 baje mujhe low stock alert bhejo."
Decision: Call `tool_create_automation(name='Daily Low Stock Alert', schedule='0 9 * * *', task='get_low_stock_products')`.
Reasoning: Schedule automated background recurring task.""",

    "insights_overview": """Example: Morning Store Health Briefing
User: "Namaste, give me a quick morning briefing on how my store did yesterday."
Decision: Call `invoke_morning_briefing(store_id=...)`.
Reasoning: Execute morning audit subgraph to summarize sales, low stock alerts, and urgent tasks.""",

    "memory": """Example: Memorizing a Store Rule
User: "Remember that customer Rahul Verma is not allowed any new credit until his ₹3,200 dues are settled."
Decision: Call `remember_store_fact(content="Customer Rahul Verma credit limit is 0 / blocked until pending dues of ₹3,200 are settled", memory_type="store_fact")`.
Reasoning: Immediately record and cache the credit rule so future billing enforces this policy.""",
}


def build_dynamic_agent_base(
    *,
    store_id: str,
    user_prompt: str,
    current_goal: str,
    loop: int,
    max_loops: int,
    user_context: dict[str, Any] | None = None,
    store_context: dict[str, Any] | None = None,
    user_role: str = "Owner",
    is_owner: bool = True,
    permissions: list[str] | None = None,
    domains: list[str] | None = None,
) -> str:
    """
    Build the dynamically assembled system prompt tailored to active domains.
    """
    # 1. Base persona and store environment
    persona_prompt, _meta = build_persona_and_context(
        store_id=store_id,
        user_prompt=user_prompt,
        current_goal=current_goal,
        loop=loop,
        max_loops=max_loops,
        user_context=user_context,
        store_context=store_context,
        user_role=user_role,
        is_owner=is_owner,
        permissions=permissions,
    )

    parts: list[str] = [persona_prompt]

    # If domains is empty (e.g. conversational/greeting), no tool rules or few shots needed
    active_domains = [d for d in (domains or []) if d in DOMAIN_RULES_MAP]

    if active_domains:
        # 2. Universal tool calling protocols
        parts.append(BASE_TOOL_RULES)

        # 3. Domain-specific rules for active domains only
        active_rules = [DOMAIN_RULES_MAP[d] for d in active_domains if d in DOMAIN_RULES_MAP]
        if active_rules:
            domain_section = f"{_SEP}\n ACTIVE DOMAIN OPERATIONAL GUIDELINES\n{_SEP}\n" + "\n\n".join(active_rules)
            parts.append(domain_section)

        # 4. At most ONE relevant few-shot example matching the primary active domain
        primary_domain = active_domains[0]
        few_shot = FEW_SHOT_BY_DOMAIN.get(primary_domain)
        if few_shot:
            few_shot_section = f"{_SEP}\n RELEVANT DECISION PATTERN\n{_SEP}\n{few_shot}"
            parts.append(few_shot_section)

    return "\n".join(parts) + "\n"

