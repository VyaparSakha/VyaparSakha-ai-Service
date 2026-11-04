"""
tests/test_tool_clustering.py
=============================
Unit tests for domain-specific tool clustering and token reduction.
Verifies that:
  1. Pure conversational queries produce 0 tools bound (100% token savings).
  2. Domain queries (e.g. sales, inventory) bind only ~10-15 tools instead of all 94.
  3. Multi-domain queries combine clusters cleanly.
  4. RBAC constraints are strictly enforced within clusters.
  5. Overall schema token reduction is >= 80% for domain-scoped queries.
"""

import pytest
import tiktoken
import json
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.agent.tools.registry import VYAPAR_TOOLS
from app.agent.tools.clusters import (
    detect_domains,
    get_pruned_tool_names,
    get_pruned_tools_for_user,
    TOOL_CLUSTERS,
    CORE_TOOLS,
)


@pytest.fixture(scope="module")
def tokenizer():
    return tiktoken.get_encoding("cl100k_base")


@pytest.fixture(scope="module")
def all_tool_token_count(tokenizer):
    schemas = [convert_to_openai_tool(t) for t in VYAPAR_TOOLS]
    s_json = json.dumps(schemas)
    return len(tokenizer.encode(s_json))


def test_conversational_queries_produce_zero_tools():
    """Conversational phrases (greetings, thanks, farewells) should yield empty domains and 0 tools."""
    conversational_inputs = [
        "hi",
        "Hello!",
        "hey there",
        "good morning",
        "thank you",
        "ok thanks",
        "bye",
        "good night",
    ]
    for prompt in conversational_inputs:
        domains = detect_domains(prompt)
        assert domains == [], f"Expected empty domains for conversational prompt: {prompt}"
        tools = get_pruned_tools_for_user(domains=domains, prompt=prompt, is_owner=True)
        assert len(tools) == 0, f"Expected 0 tools for conversational prompt: {prompt}"


def test_sales_domain_pruning_and_token_reduction(tokenizer, all_tool_token_count):
    """A sales query should only bind sales + core tools, achieving > 85% token reduction."""
    prompt = "tell me about last 30 day sells"
    domains = detect_domains(prompt)
    assert "sales" in domains

    tools = get_pruned_tools_for_user(domains=domains, prompt=prompt, is_owner=True)
    tool_names = {t.name for t in tools}

    # Verify essential sales tools are present
    assert "get_sales_summary" in tool_names
    assert "get_top_selling_products" in tool_names
    assert "get_daily_sales_trend" in tool_names
    assert "ask_for_clarification" in tool_names

    # Verify unrelated tools are pruned
    assert "tool_create_automation" not in tool_names
    assert "tool_create_buyer" not in tool_names
    assert "parse_supplier_invoice_image" not in tool_names

    # Check token count and reduction
    schemas = [convert_to_openai_tool(t) for t in tools]
    s_json = json.dumps(schemas)
    sales_tokens = len(tokenizer.encode(s_json))

    reduction_pct = (all_tool_token_count - sales_tokens) / all_tool_token_count
    assert reduction_pct >= 0.85, f"Expected >= 85% reduction, got {reduction_pct * 100:.1f}%"
    assert len(tools) <= 16, f"Expected <= 16 tools bound, got {len(tools)}"


def test_inventory_domain_pruning():
    """An inventory query should bind only inventory + core tools."""
    prompt = "which products have dead stock and low quantity?"
    domains = detect_domains(prompt)
    assert "inventory" in domains

    tools = get_pruned_tools_for_user(domains=domains, prompt=prompt, is_owner=True)
    tool_names = {t.name for t in tools}

    assert "get_inventory_summary" in tool_names
    assert "get_low_stock_products" in tool_names
    assert "get_dead_stock" in tool_names
    assert "get_sales_anomalies" not in tool_names
    assert "send_store_email" not in tool_names


def test_multi_domain_query_combination():
    """A query mentioning low stock items and best selling products should combine both clusters."""
    prompt = "Which low stock items have the highest sales and best selling trend?"
    domains = detect_domains(prompt)
    assert "sales" in domains
    assert "inventory" in domains

    tools = get_pruned_tools_for_user(domains=domains, prompt=prompt, is_owner=True)
    tool_names = {t.name for t in tools}

    assert "get_sales_summary" in tool_names
    assert "get_low_stock_products" in tool_names
    assert "ask_for_clarification" in tool_names
    assert "tool_create_automation" not in tool_names


def test_rbac_permissions_respected_within_cluster():
    """If an employee lacks reports permissions, owner/report tools must not be returned."""
    # Employee with only inventory permissions
    tools = get_pruned_tools_for_user(
        domains=["sales"],
        is_owner=False,
        permissions=["inventory:manage"],
    )
    tool_names = {t.name for t in tools}

    # Should not have reports_tools like get_sales_summary
    assert "get_sales_summary" not in tool_names
    assert "get_profit_loss_report" not in tool_names

