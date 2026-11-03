from app.agent.tools.clarification import ask_for_clarification

# Inventory
from app.agent.tools.inventory.summary import get_inventory_summary
from app.agent.tools.inventory.low_stock import get_low_stock_products
from app.agent.tools.inventory.stock_history import get_stock_history
from app.agent.tools.inventory.dead_stock import get_dead_stock
from app.agent.tools.inventory.inventory_risk import get_inventory_risk
from app.agent.tools.inventory.expiry_alerts import get_expiry_alerts
from app.agent.tools.inventory.category_stock_health import get_category_stock_health
from app.agent.tools.inventory.slow_moving import get_slow_moving_products
from app.agent.tools.inventory.stockout_risk import get_stockout_risk_products
from app.agent.tools.inventory.recently_added import get_recently_added_products

# Sales
from app.agent.tools.sales.summary import get_sales_summary
from app.agent.tools.sales.top_products import get_top_selling_products
from app.agent.tools.sales.product_performance import get_product_performance
from app.agent.tools.sales.category_performance import get_category_performance
from app.agent.tools.sales.sales_anomalies import get_sales_anomalies
from app.agent.tools.sales.daily_trend import get_daily_sales_trend
from app.agent.tools.sales.discount_impact import get_discount_impact
from app.agent.tools.sales.profit_margin import get_profit_margin_analysis
from app.agent.tools.sales.fast_moving import get_fast_moving_products

# Forecast
from app.agent.tools.forecast.demand import get_demand_forecast
from app.agent.tools.forecast.restock import get_restock_priorities
from app.agent.tools.forecast.stockout import get_stockout_estimate
from app.agent.tools.forecast.restock_budget import calculate_restock_budget
from app.agent.tools.forecast.smart_purchase_order import create_smart_purchase_order
from app.agent.tools.forecast.festival_planner import get_festival_demand_planner


# Insights & Daily Action Plan
from app.agent.tools.insights.store_insights import get_store_insights
from app.agent.tools.insights.daily_action_checklist import get_daily_action_checklist

# Sales & Goals
from app.agent.tools.sales.bundle_recommendation import generate_deal_bundle_recommendation
from app.agent.tools.sales.goal_progress import get_store_goal_progress_report

# Communication (Express backend email wrapper)
from app.agent.tools.communication.send_email import send_store_email

# Web search
from app.agent.tools.web.search_web import search_web
from app.agent.tools.web.web_research import web_research

# Products
from app.agent.tools.products.search import search_products
from app.agent.tools.products.details import get_product_details
from app.agent.tools.products.history import get_product_history

# Stores
from app.agent.tools.stores.summary import get_store_summary
from app.agent.tools.stores.comparison import compare_stores

# Suppliers
from app.agent.tools.suppliers.search import search_suppliers
from app.agent.tools.suppliers.performance import get_supplier_performance
from app.agent.tools.suppliers.pricing import get_supplier_pricing

# Subgraphs
from app.agent.tools.subgraphs.invoke_briefing import invoke_morning_briefing
from app.agent.tools.subgraphs.invoke_inventory_audit import invoke_deep_inventory_audit
from app.agent.tools.subgraphs.invoke_restock_order import invoke_smart_restock_order
from app.agent.tools.subgraphs.invoke_document_generation import invoke_document_generation
from app.agent.tools.subgraphs.invoke_email_composer import invoke_email_composer

# Memory & Owner Personalization (Redis + Pinecone + Merchant Scratchpad Diary)
from app.agent.tools.memory.search import (
    search_memory,
    remember_store_fact,
    get_owner_goals_and_preferences,
    set_owner_goal_or_preference,
)
from app.agent.tools.memory.merchant_scratchpad import (
    write_scratchpad_note,
    read_scratchpad_notes,
    update_scratchpad_note,
    delete_scratchpad_note,
)

# Buyers & Credit Risk
from app.agent.tools.buyers.search import search_buyers
from app.agent.tools.buyers.dues import get_buyer_dues
from app.agent.tools.buyers.credit_risk import analyze_customer_credit_risk
from app.agent.tools.buyers.write import tool_create_buyer, tool_update_buyer, tool_delete_buyer

# Sellers
from app.agent.tools.sellers.search import search_sellers
from app.agent.tools.sellers.write import tool_create_seller, tool_update_seller, tool_delete_seller

# Purchases & Invoice OCR
from app.agent.tools.purchases.summary import get_purchase_summary
from app.agent.tools.purchases.search import search_purchases
from app.agent.tools.purchases.write import (
    tool_create_purchase,
    tool_receive_purchase,
    tool_update_purchase,
    tool_delete_purchase,
)
from app.agent.tools.purchases.invoice_ocr import parse_supplier_invoice_image

# Expenses
from app.agent.tools.expenses.summary import get_expense_summary
from app.agent.tools.expenses.write import tool_create_expense, tool_update_expense, tool_delete_expense

# Profit & Loss
from app.agent.tools.profit_loss.report import get_profit_loss_report

# Products
from app.agent.tools.products.write import tool_create_product, tool_update_product, tool_adjust_stock, tool_delete_product

# Code Execution / Python Interpreter
from app.agent.tools.code_execution.python_interpreter import execute_python_code

# Personalization & Proactive AI Tools
from app.agent.tools.memory.merchant_memories import manage_merchant_memories
from app.agent.tools.insights.proactive_insights import manage_proactive_insights
from app.agent.tools.analytics.store_baselines import get_store_baselines
from app.agent.tools.purchases.price_history import check_supplier_price_trends
from app.agent.tools.customers.credit_ledger import get_customer_credit_ledger

# Navigation
from app.agent.tools.navigation.navigate import tool_navigate_page

# Billing & POS
from app.agent.tools.billing.add_item import tool_add_billing_item
from app.agent.tools.billing.generate_bill import tool_generate_bill

# Automations & Scheduled Workflows (Express BullMQ)
from app.agent.tools.automation.manage_automation import (
    tool_create_automation,
    tool_list_automations,
    tool_toggle_automation,
    tool_trigger_automation,
    tool_delete_automation,
)

# LangGraph ToolNode will use this list
VYAPAR_TOOLS = [
    ask_for_clarification,
    execute_python_code,

    
    # Inventory
    get_inventory_summary,
    get_low_stock_products,
    get_stock_history,
    get_dead_stock,
    get_inventory_risk,
    get_expiry_alerts,
    get_category_stock_health,
    get_slow_moving_products,
    get_stockout_risk_products,
    
    # Recently added products
    get_recently_added_products,
    
    # Sales
    get_sales_summary,
    get_top_selling_products,
    get_product_performance,
    get_category_performance,
    get_sales_anomalies,
    get_daily_sales_trend,
    get_discount_impact,
    get_profit_margin_analysis,
    get_fast_moving_products,
    generate_deal_bundle_recommendation,
    get_store_goal_progress_report,
    
    # Forecast & Purchase Orders
    get_demand_forecast,
    get_restock_priorities,
    get_stockout_estimate,
    calculate_restock_budget,
    create_smart_purchase_order,
    get_festival_demand_planner,
    
    # Insights & Action Plan
    get_store_insights,
    get_daily_action_checklist,

    # Communication
    send_store_email,

    # Web search
    search_web,
    web_research,
    
    # Products
    search_products,
    get_product_details,
    get_product_history,
    
    # Stores
    get_store_summary,
    compare_stores,
    
    # Suppliers
    search_suppliers,
    get_supplier_performance,
    get_supplier_pricing,
    
    # Subgraphs
    invoke_morning_briefing,
    invoke_deep_inventory_audit,
    invoke_smart_restock_order,
    invoke_document_generation,
    invoke_email_composer,

    # Memory & Owner Personalization (Redis + Pinecone)
    search_memory,
    remember_store_fact,
    get_owner_goals_and_preferences,
    set_owner_goal_or_preference,
    write_scratchpad_note,
    read_scratchpad_notes,
    update_scratchpad_note,
    delete_scratchpad_note,

    # Buyers
    search_buyers,
    get_buyer_dues,
    analyze_customer_credit_risk,
    tool_create_buyer,
    tool_update_buyer,
    tool_delete_buyer,
    
    # Sellers
    search_sellers,
    tool_create_seller,
    tool_update_seller,
    tool_delete_seller,

    # Purchases & Invoice OCR
    get_purchase_summary,
    search_purchases,
    tool_create_purchase,
    tool_receive_purchase,
    tool_update_purchase,
    tool_delete_purchase,
    parse_supplier_invoice_image,

    # Expenses
    get_expense_summary,
    tool_create_expense,
    tool_update_expense,
    tool_delete_expense,

    # Profit & Loss
    get_profit_loss_report,
    
    # Products
    tool_create_product,
    tool_update_product,
    tool_adjust_stock,
    tool_delete_product,

    # Personalization & Proactive Intelligence
    manage_merchant_memories,
    manage_proactive_insights,
    get_store_baselines,
    check_supplier_price_trends,
    get_customer_credit_ledger,
    
    # Navigation
    tool_navigate_page,
    
    # Billing
    tool_add_billing_item,
    tool_generate_bill,

    # Automations
    tool_create_automation,
    tool_list_automations,
    tool_toggle_automation,
    tool_trigger_automation,
    tool_delete_automation,
]


def get_tool_by_name(name: str):
    """Helper to dispatch tool calls."""
    for tool in VYAPAR_TOOLS:
        if tool.name == name:
            return tool
    return None


# ---------------------------------------------------------------------------
# RBAC Tool Permission Mappings
# ---------------------------------------------------------------------------

# Base tools always accessible to any authorized store member
BASE_TOOLS = {
    "ask_for_clarification",
    "tool_navigate_page",
    "search_products",
    "get_product_details",
    "get_product_history",
    "get_inventory_summary",
    "search_web",
    "web_research",
}

# Billing tools ("billing:create")
BILLING_TOOLS = {
    "tool_add_billing_item",
    "tool_generate_bill",
    "get_customer_credit_ledger",
    "search_buyers",
    "get_buyer_dues",
    "analyze_customer_credit_risk",
    "tool_create_buyer",
    "tool_update_buyer",
}

# Inventory tools ("inventory:add" or "inventory:manage")
INVENTORY_TOOLS = {
    "get_low_stock_products",
    "get_stock_history",
    "get_dead_stock",
    "get_inventory_risk",
    "get_expiry_alerts",
    "get_category_stock_health",
    "get_slow_moving_products",
    "get_stockout_risk_products",
    "get_recently_added_products",
    "search_suppliers",
    "get_supplier_performance",
    "get_supplier_pricing",
    "search_sellers",
    "tool_create_seller",
    "tool_update_seller",
    "get_purchase_summary",
    "search_purchases",
    "tool_create_purchase",
    "tool_receive_purchase",
    "tool_update_purchase",
    "parse_supplier_invoice_image",
    "check_supplier_price_trends",
    "invoke_deep_inventory_audit",
    "invoke_smart_restock_order",
    "get_restock_priorities",
    "get_stockout_estimate",
    "calculate_restock_budget",
    "create_smart_purchase_order",
    "tool_create_product",
    "tool_update_product",
    "tool_adjust_stock",
}

# Sales analytics & reports tools ("reports:view_sales")
REPORTS_TOOLS = {
    "get_sales_summary",
    "get_top_selling_products",
    "get_product_performance",
    "get_category_performance",
    "get_sales_anomalies",
    "get_daily_sales_trend",
    "get_discount_impact",
    "get_fast_moving_products",
    "generate_deal_bundle_recommendation",
    "get_demand_forecast",
    "get_festival_demand_planner",
    "get_store_insights",
    "get_daily_action_checklist",
    "invoke_morning_briefing",
    "get_store_baselines",
    "manage_proactive_insights",
}

# Owner-only tools (require is_owner == True)
OWNER_ONLY_TOOLS = {
    "get_profit_loss_report",
    "get_profit_margin_analysis",
    "get_store_goal_progress_report",
    "get_expense_summary",
    "tool_create_expense",
    "tool_update_expense",
    "tool_delete_expense",
    "tool_delete_product",
    "tool_delete_buyer",
    "tool_delete_seller",
    "tool_delete_purchase",
    "tool_create_automation",
    "tool_list_automations",
    "tool_toggle_automation",
    "tool_trigger_automation",
    "tool_delete_automation",
    "send_store_email",
    "compare_stores",
    "get_store_summary",
    "invoke_document_generation",
    "invoke_email_composer",
    "execute_python_code",
    "search_memory",
    "remember_store_fact",
    "get_owner_goals_and_preferences",
    "set_owner_goal_or_preference",
    "write_scratchpad_note",
    "read_scratchpad_notes",
    "update_scratchpad_note",
    "delete_scratchpad_note",
    "manage_merchant_memories",
}


def get_allowed_tool_names_for_user(is_owner: bool = True, permissions: list[str] | None = None) -> set[str]:
    """
    Compute set of tool names accessible to the user based on ownership and assigned permissions.
    """
    if is_owner or (permissions and "*" in permissions):
        return {tool.name for tool in VYAPAR_TOOLS}

    perms = set(permissions or [])
    allowed = set(BASE_TOOLS)

    if "billing:create" in perms:
        allowed.update(BILLING_TOOLS)

    if "inventory:add" in perms or "inventory:manage" in perms:
        allowed.update(INVENTORY_TOOLS)

    if "reports:view_sales" in perms:
        allowed.update(REPORTS_TOOLS)

    return allowed - OWNER_ONLY_TOOLS


def get_tools_for_user(is_owner: bool = True, permissions: list[str] | None = None) -> list:
    """
    Returns list of LangChain BaseTool objects that the user is authorized to bind to LLM.
    """
    allowed_names = get_allowed_tool_names_for_user(is_owner=is_owner, permissions=permissions)
    return [tool for tool in VYAPAR_TOOLS if tool.name in allowed_names]


def is_tool_allowed_for_user(tool_name: str, is_owner: bool = True, permissions: list[str] | None = None) -> bool:
    """
    Validates if a specific tool execution is permitted for defense-in-depth security.
    """
    allowed_names = get_allowed_tool_names_for_user(is_owner=is_owner, permissions=permissions)
    return tool_name in allowed_names
