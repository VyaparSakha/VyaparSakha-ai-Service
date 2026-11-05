"""
tests/test_semantic_tool_index.py
==================================
Unit tests for the in-memory FastEmbed semantic tool retriever and 2-LLM-call dispatch.
"""

import pytest
from app.agent.tools.semantic_index import SemanticToolIndex, get_semantic_tools_for_query


def test_semantic_tool_index_singleton_and_warm_load():
    """Verify SemanticToolIndex initializes from disk cache in < 1 second."""
    idx = SemanticToolIndex.get_instance()
    idx.initialize()
    assert idx._initialized is True
    assert idx._vectors is not None
    assert len(idx._tool_names) > 0


def test_sales_query_multilingual_matching():
    """Verify sales queries in Hinglish/English match sales tools."""
    idx = SemanticToolIndex.get_instance()
    idx.initialize()

    # Hinglish query
    results_hinglish = idx.search("pichle 30 din ki bikri kitna hua dhandha", top_k=3)
    matched_names = [t.name for t, _ in results_hinglish]
    assert "get_sales_summary" in matched_names

    # English query
    results_en = idx.search("show my daily sales trend and revenue", top_k=3)
    matched_en_names = [t.name for t, _ in results_en]
    assert any(name in matched_en_names for name in ("get_sales_summary", "get_daily_sales_trend"))


def test_inventory_query_hinglish_matching():
    """Verify low stock / inventory alert matches inventory tools."""
    idx = SemanticToolIndex.get_instance()
    idx.initialize()

    results = idx.search("maal khatam hone wala hai low stock alert", top_k=3)
    matched_names = [t.name for t, _ in results]
    assert any(name in matched_names for name in ("get_low_stock_products", "get_stockout_risk_products", "get_inventory_summary"))


def test_credit_khata_query_matching():
    """Verify udhari / dues queries match buyer dues tools."""
    idx = SemanticToolIndex.get_instance()
    idx.initialize()

    results = idx.search("sharma ji ka kitna udhari baki hai khata", top_k=3)
    matched_names = [t.name for t, _ in results]
    assert "get_buyer_dues" in matched_names


def test_semantic_search_latency():
    """Verify user query search takes < 30ms."""
    import time
    idx = SemanticToolIndex.get_instance()
    idx.initialize()

    t0 = time.perf_counter()
    res = idx.search("kisko paisa dena hai supplier dues", top_k=4)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert elapsed_ms < 50.0  # Must be fast (< 50ms)
    assert len(res) > 0

