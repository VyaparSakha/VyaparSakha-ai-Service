"""
app/agent/tools/clusters.py
===========================
Domain-specific tool clustering for Just-In-Time (JIT) dynamic tool pruning.
Drastically slashes LLM prompt token consumption from ~22,400+ tokens (all 94 tools)
down to ~1,200 - 3,000 tokens by binding ONLY the tools relevant to the active
request's business domain(s).
"""

from __future__ import annotations

import re
from typing import List, Set, Dict, Optional, Any
import structlog

from app.agent.tools.registry import (
    VYAPAR_TOOLS,
    get_tool_by_name,
    get_allowed_tool_names_for_user,
    is_tool_allowed_for_user,
)

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.tools.clusters")

# ---------------------------------------------------------------------------
# Core tools always available across analytical requests
# ---------------------------------------------------------------------------
CORE_TOOLS: set[str] = {
    "ask_for_clarification",
    "execute_python_code",
    "tool_navigate_page",
}

# ---------------------------------------------------------------------------
# Domain Tool Clusters
# ---------------------------------------------------------------------------
TOOL_CLUSTERS: dict[str, set[str]] = {
    "sales": {
        "get_sales_summary",
        "get_top_selling_products",
        "get_product_performance",
        "get_category_performance",
        "get_sales_anomalies",
        "get_daily_sales_trend",
        "get_discount_impact",
        "get_profit_margin_analysis",
        "get_fast_moving_products",
        "generate_deal_bundle_recommendation",
        "get_store_goal_progress_report",
    },
    "inventory": {
        "get_inventory_summary",
        "get_low_stock_products",
        "get_stock_history",
        "get_dead_stock",
        "get_inventory_risk",
        "get_expiry_alerts",
        "get_category_stock_health",
        "get_slow_moving_products",
        "get_stockout_risk_products",
        "get_recently_added_products",
        "search_products",
        "get_product_details",
        "get_product_history",
        "tool_create_product",
        "tool_update_product",
        "tool_adjust_stock",
        "tool_delete_product",
        "invoke_deep_inventory_audit",
    },
    "forecast": {
        "get_demand_forecast",
        "get_restock_priorities",
        "get_stockout_estimate",
        "calculate_restock_budget",
        "create_smart_purchase_order",
        "get_festival_demand_planner",
        "invoke_smart_restock_order",
    },
    "suppliers_purchases": {
        "search_suppliers",
        "get_supplier_performance",
        "get_supplier_pricing",
        "search_sellers",
        "tool_create_seller",
        "tool_update_seller",
        "tool_delete_seller",
        "get_purchase_summary",
        "search_purchases",
        "tool_create_purchase",
        "tool_receive_purchase",
        "tool_update_purchase",
        "tool_delete_purchase",
        "parse_supplier_invoice_image",
        "check_supplier_price_trends",
    },
    "buyers_credit": {
        "search_buyers",
        "get_buyer_dues",
        "analyze_customer_credit_risk",
        "tool_create_buyer",
        "tool_update_buyer",
        "tool_delete_buyer",
        "get_customer_credit_ledger",
    },
    "finance_expenses": {
        "get_profit_loss_report",
        "get_expense_summary",
        "tool_create_expense",
        "tool_update_expense",
        "tool_delete_expense",
        "get_profit_margin_analysis",
    },
    "billing": {
        "tool_add_billing_item",
        "tool_generate_bill",
        "search_products",
        "get_product_details",
        "get_customer_credit_ledger",
        "get_buyer_dues",
    },
    "documents": {
        "invoke_document_generation",
        # Document generation often needs underlying data tools to build the report
        "get_sales_summary",
        "get_top_selling_products",
        "get_daily_sales_trend",
        "get_inventory_summary",
        "get_low_stock_products",
        "get_profit_loss_report",
        "get_expense_summary",
    },
    "communication": {
        "send_store_email",
        "invoke_email_composer",
        "search_suppliers",
        "search_sellers",
        "get_supplier_performance",
    },
    "automations": {
        "tool_create_automation",
        "tool_list_automations",
        "tool_toggle_automation",
        "tool_trigger_automation",
        "tool_delete_automation",
    },
    "insights_overview": {
        "get_store_insights",
        "get_daily_action_checklist",
        "invoke_morning_briefing",
        "get_store_summary",
        "compare_stores",
        "get_store_baselines",
        "manage_proactive_insights",
        "get_sales_summary",
        "get_inventory_summary",
    },
    "memory": {
        "search_memory",
        "remember_store_fact",
        "get_owner_goals_and_preferences",
        "set_owner_goal_or_preference",
        "write_scratchpad_note",
        "read_scratchpad_notes",
        "update_scratchpad_note",
        "delete_scratchpad_note",
        "manage_merchant_memories",
    },
    "web": {
        "search_web",
        "web_research",
    },
}

# ---------------------------------------------------------------------------
# Fast Domain Keyword Matchers (English & Hinglish)
# ---------------------------------------------------------------------------
_DOMAIN_PATTERNS: dict[str, re.Pattern] = {
    "sales": re.compile(
        r"\b(sale|sales|sells|selling|bikri|becha|revenue|kamai|turnover|sold|margin|margins|"
        r"top selling|best selling|fast moving|discount|discounts|order value|sales trend|"
        r"daily sales|monthly sales|sales report|gross sales)\b",
        re.IGNORECASE,
    ),
    "inventory": re.compile(
        r"\b(inventory|stock|stockout|out of stock|maal|dead stock|expiry|expired|"
        r"slow moving|product|products|item|items|sku|stock level|quantity|godown|"
        r"warehouse|catalog|category stock)\b",
        re.IGNORECASE,
    ),
    "forecast": re.compile(
        r"\b(forecast|restock|reorder|predict|prediction|replenish|demand|festival planner|"
        r"order quantity|lead time|future demand|smart po)\b",
        re.IGNORECASE,
    ),
    "suppliers_purchases": re.compile(
        r"\b(supplier|suppliers|seller|sellers|vendor|vendors|distributor|distributors|"
        r"purchase|purchases|po-|po |invoice|receipt|challan|receive order|bought from|"
        r"price trend|price increase)\b",
        re.IGNORECASE,
    ),
    "buyers_credit": re.compile(
        r"\b(buyer|buyers|customer|customers|grahak|client|clients|udhar|khata|due|dues|"
        r"credit|owe|owes|pending payment|receivable|receivables|credit limit|baki)\b",
        re.IGNORECASE,
    ),
    "finance_expenses": re.compile(
        r"\b(p&l|profit and loss|profit|loss|expense|expenses|kharcha|spending|cost|costs|"
        r"net profit|gross profit|cash flow|balance sheet|profitability)\b",
        re.IGNORECASE,
    ),
    "billing": re.compile(
        r"\b(billing|bill|generate bill|pos|counter|checkout|cart|add to bill|parchi)\b",
        re.IGNORECASE,
    ),
    "documents": re.compile(
        r"\b(download|export|excel|xlsx|word|docx|pdf|generate report|export report|"
        r"download link|file export|spreadsheet)\b",
        re.IGNORECASE,
    ),
    "communication": re.compile(
        r"\b(email|mail|draft email|compose email|send email|bhejo mail|supplier email)\b",
        re.IGNORECASE,
    ),
    "automations": re.compile(
        r"\b(automate|automation|automations|schedule|cron|daily alert|recurring|"
        r"alert schedule|auto trigger)\b",
        re.IGNORECASE,
    ),
    "insights_overview": re.compile(
        r"\b(briefing|morning briefing|overview|health|checklist|daily action|"
        r"store status|how is my store|dukaan kaisi chal rahi|summary report|store summary)\b",
        re.IGNORECASE,
    ),
    "memory": re.compile(
        r"\b(remember|memorize|recall|scratchpad|diary|my preference|my rule|"
        r"store rule|last time you said|note down)\b",
        re.IGNORECASE,
    ),
    "web": re.compile(
        r"\b(search web|web research|google|online price|market price|competitor price|"
        r"industry report|external)\b",
        re.IGNORECASE,
    ),
}

_CONVERSATIONAL_PATTERNS = re.compile(
    r"\b(hi|hello|hey|hii|helo|hola|good morning|good evening|good afternoon|good night|goodnight|howdy|"
    r"how are you|kaise ho|kya haal|who are you|tum kaun ho|aap kaun ho|"
    r"what can you do|kya kar sakte ho|thanks|thank you|thankyou|thx|ok|okay|"
    r"bye|goodbye|cya|take care|gn|namaste|pranam)\b",
    re.IGNORECASE,
)


def detect_domains(prompt: str) -> list[str]:
    """
    Detect the active business domain(s) from the merchant prompt.
    Returns a list of matched domain keys.

    If the query is purely conversational (greeting, thanks, farewell), returns empty list [].
    If no domain matched but the query is analytical, returns a sensible baseline.
    """
    text = (prompt or "").strip()
    if not text:
        return []

    matched_domains: list[str] = []
    for domain, pattern in _DOMAIN_PATTERNS.items():
        if pattern.search(text):
            matched_domains.append(domain)

    # Conversational shortcut: if conversational phrase detected and NO business domain matched -> 0 tools
    if not matched_domains and _CONVERSATIONAL_PATTERNS.search(text):
        return []

    # If document export is requested, check if a specific subject is also mentioned
    # e.g. "generate excel sales report" -> documents + sales
    if "documents" in matched_domains and len(matched_domains) == 1:
        # Default document export to include sales + inventory summaries for context
        matched_domains.extend(["sales", "inventory"])

    # If no specific domain pattern matched, fallback to general store overview + sales
    if not matched_domains:
        matched_domains = ["insights_overview", "sales", "inventory"]

    return matched_domains


def get_pruned_tool_names(
    domains: list[str] | None = None,
    prompt: str = "",
    is_owner: bool = True,
    permissions: list[str] | None = None,
) -> set[str]:
    """
    Computes the set of tool names to bind to the LLM, strictly pruned to the
    active domain(s) and constrained by the user's RBAC permissions.
    """
    # 1. Determine active domains
    if domains is None:
        active_domains = detect_domains(prompt)
    else:
        active_domains = list(domains)

    # If explicitly empty (conversational / greeting), bind NO tools (or only clarification)
    if not active_domains:
        return set()

    # 2. Collect tool names across active domains
    candidate_names: set[str] = set(CORE_TOOLS)
    for domain in active_domains:
        cluster = TOOL_CLUSTERS.get(domain)
        if cluster:
            candidate_names.update(cluster)

    # 3. Intersect with user's RBAC allowed tools
    allowed_by_rbac = get_allowed_tool_names_for_user(is_owner=is_owner, permissions=permissions)
    final_tool_names = candidate_names.intersection(allowed_by_rbac)

    return final_tool_names


def get_pruned_tools_for_user(
    domains: list[str] | None = None,
    prompt: str = "",
    is_owner: bool = True,
    permissions: list[str] | None = None,
) -> list[Any]:
    """
    Returns the list of LangChain BaseTool objects to bind to the LLM.
    Pruned from 94 tools down to ~5-15 domain-specific tools.
    """
    tool_names = get_pruned_tool_names(
        domains=domains,
        prompt=prompt,
        is_owner=is_owner,
        permissions=permissions,
    )

    if not tool_names:
        return []

    # Map names back to tool instances
    pruned = [tool for tool in VYAPAR_TOOLS if tool.name in tool_names]

    LOGGER.info(
        "tool_clustering_pruned",
        domains=domains or detect_domains(prompt),
        tool_count=len(pruned),
        all_tools_count=len(VYAPAR_TOOLS),
        reduction_pct=round((1 - len(pruned) / len(VYAPAR_TOOLS)) * 100, 1),
    )

    return pruned
