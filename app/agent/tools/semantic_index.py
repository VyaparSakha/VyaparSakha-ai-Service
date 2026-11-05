"""
app/agent/tools/semantic_index.py
==================================
In-memory vector embedding index for Vyapar Sakha tools using fast local ONNX embeddings (FastEmbed).
Preloads tool embeddings from a cached NumPy file on disk for instant (< 2ms) startup.
Provides sub-10ms hybrid semantic tool retrieval for multilingual/Hinglish merchant queries.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, List, Tuple
import numpy as np
import structlog
from langchain_core.tools import BaseTool

from app.agent.tools.registry import VYAPAR_TOOLS

LOGGER = structlog.get_logger("vyaparsathi.ai.tools.semantic_index")

# Disable HuggingFace symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

CACHE_FILE_PATH = Path(__file__).parent / "tool_vectors_cache.npz"

TOOL_SEMANTIC_DESCRIPTIONS: dict[str, str] = {
    # Sales Domain
    "get_sales_summary": "sales summary total revenue daily sales monthly sales profit earnings turnover bikri dhandha kamai sell report 30 days today yesterday pichle 30 din",
    "get_daily_sales_trend": "daily sales trend line chart graph time series sales velocity bikri graph dhandha trend chart 30 days daily trend",
    "get_top_selling_products": "top selling products best sellers fast moving items sabse jyada bikne wala bar chart top items 30 days",
    "get_product_performance": "individual product sales performance SKU sales volume unit revenue product report",
    "get_category_performance": "category wise sales department breakdown grocery dairy snacks category comparison",
    "get_sales_anomalies": "sales anomalies drop in revenue sudden sales spike suspicious transactions unusual sales dip",
    "get_discount_impact": "discount impact promotional pricing price cut margin effect discount analysis",
    "get_profit_margin_analysis": "gross margin net margin profitability analysis profit margin fayda munafa",
    "get_fast_moving_products": "fast moving high velocity fast selling inventory turnover high demand",

    # Inventory Domain
    "get_inventory_summary": "inventory summary total stock warehouse valuation stock valuation maal ka hisab stock count",
    "get_low_stock_products": "low stock out of stock running out reorder alert stockout risk maal khatam kamti stock reorder list kam stock",
    "get_stock_history": "stock history inventory level changes historical stock depletion rate stock movement",
    "get_dead_stock": "dead stock slow moving dormant inventory unsold items dust gatherers purana maal frozen capital",
    "get_inventory_risk": "inventory risk working capital lockup high risk items overstocked capital trap",
    "get_expiry_alerts": "expiry alerts expiring soon expired products shelf life perishables kharab hone wala expiry date",
    "get_category_stock_health": "category stock health category inventory distribution stock balance by category",
    "get_slow_moving_products": "slow moving products low velocity stagnant items kam bikne wala maal",
    "get_stockout_risk_products": "stockout risk running out in 3 days lead time restock urgently maal khatam hone ka khatra",
    "get_recently_added_products": "recently added products new stock new arrivals recently created products",

    # Forecast & Restock Domain
    "get_demand_forecast": "demand forecast future demand prediction predictive sales agle hafte agle mahine kitna bikega",
    "get_restock_priorities": "restock priorities what to order supplier reorder list urgently needed items priority reorder",
    "get_stockout_estimate": "stockout estimate days until stockout runway days of inventory left kab khatam hoga",
    "calculate_restock_budget": "restock budget purchase capital needed reorder cost working capital requirement",
    "create_smart_purchase_order": "purchase order PO create order supplier order restock order maal mangwana order form",
    "get_festival_demand_planner": "festival demand diwali dhanteras holi eid puja festival sales spike planning seasonal demand",

    # Credit, Dues & Khata Domain
    "get_buyer_dues": "customer dues buyer dues khata baki udhari kis se paisa lena hai pending receivables credit balance ledger buyer owes",
    "get_seller_dues": "supplier dues seller dues distributor dues kisko paisa dena hai payables supplier balance baki amount owe suppliers",
    "get_khata_customer_summary": "khata customer summary customer ledger book credit record udhar khata hisab",
    "get_khata_supplier_summary": "khata supplier summary vendor ledger distributor payment history hisab kitab",
    "get_customer_debt_aging": "debt aging overdue payments overdue credit 30 60 90 days overdue udhari purani",

    # Finance & Expenses Domain
    "get_store_expenses": "store expenses operational expenses rent electricity utility staff salary kharcha dukan ka kharcha",
    "get_profit_loss_statement": "profit and loss statement P&L net profit gross profit EBITDA dukan ka profit loss hisab",
    "get_cash_flow_summary": "cash flow cash in cash out daily cash drawer reconciliation cash balance",

    # Store Insights & Overview
    "get_store_summary": "store summary store overview store health morning briefing dukan ka overview general status",
    "get_store_insights": "store strategic insights retail recommendations diagnostic health check co-pilot insights",
    "get_daily_action_checklist": "daily action checklist morning briefing staff checklist owner to-do list tasks today",
    "compare_stores": "compare stores multi store comparison branch performance comparison branch A vs branch B",

    # Subgraphs & Heavy Workflows
    "invoke_morning_briefing": "morning briefing daily briefing start of day report today store status subgraphs overview",
    "invoke_smart_restock_order": "smart restock workflow full purchase order creation supplier negotiation plan",
    "invoke_deep_inventory_audit": "full inventory audit deep stock inspection complete audit of dead stock and expiry",
    "invoke_email_composer": "compose email draft email write email to supplier customer payment reminder email purchase order email",
    "invoke_document_generation": "generate excel create word report download excel sell report export xlsx docx",
    "web_research": "deep web research competitor pricing market trends supplier research on internet",
    "search_web": "quick web search search internet",
    "ask_for_clarification": "ask clarification user question uncertain information needed",
    "search_products": "search products find product search item name barcode lookup",
    "get_product_details": "product details SKU specifications cost price selling price barcode",
    "get_product_history": "product sales history price change history inventory movement",
    "search_suppliers": "search suppliers find vendor list suppliers distributor phone contact",
}


class SemanticToolIndex:
    """
    Fast in-memory semantic tool index using FastEmbed (ONNX) + Hybrid Sparse/Dense reranking.
    Preloads tool embeddings from disk cache on startup (< 2ms).
    Query searches execute in < 10ms.
    """

    _instance: SemanticToolIndex | None = None

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None
        self._tool_names: list[str] = []
        self._tool_map: dict[str, BaseTool] = {}
        self._vectors: np.ndarray | None = None
        self._initialized = False

    @classmethod
    def get_instance(cls) -> SemanticToolIndex:
        if cls._instance is None:
            cls._instance = SemanticToolIndex()
        return cls._instance

    def initialize(self, tools: list[BaseTool] | None = None) -> None:
        """Load the embedding model and precompute/load normalized tool vectors in memory."""
        if self._initialized:
            return

        t0 = time.perf_counter()
        active_tools = tools or VYAPAR_TOOLS
        self._tool_map = {t.name: t for t in active_tools}
        self._tool_names = [t.name for t in active_tools]

        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name)

            # Try to load cached vectors from disk
            if CACHE_FILE_PATH.exists():
                try:
                    data = np.load(CACHE_FILE_PATH, allow_pickle=True)
                    cached_names = list(data["names"])
                    if cached_names == self._tool_names:
                        self._vectors = data["vectors"]
                        self._initialized = True
                        ms = round((time.perf_counter() - t0) * 1000, 1)
                        LOGGER.info("semantic_tool_index_loaded_from_disk_cache", tool_count=len(self._tool_names), ms=ms)
                        return
                except Exception as cache_exc:
                    LOGGER.warning("semantic_tool_index_cache_load_failed", error=str(cache_exc))

            # Generate fresh vectors and save to disk
            LOGGER.info("semantic_tool_index_generating_vectors", tool_count=len(self._tool_names))
            documents: list[str] = []
            for name in self._tool_names:
                tool = self._tool_map[name]
                desc = tool.description or ""
                semantic_desc = TOOL_SEMANTIC_DESCRIPTIONS.get(name, "")
                combined = f"{name}: {semantic_desc} {desc}".strip()
                documents.append(combined)

            raw_vecs = np.array(list(self._model.embed(documents)), dtype=np.float32)
            norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            self._vectors = raw_vecs / norms

            # Cache to disk for instant future loads
            try:
                np.savez_compressed(CACHE_FILE_PATH, vectors=self._vectors, names=np.array(self._tool_names))
                LOGGER.info("semantic_tool_index_saved_to_disk_cache", path=str(CACHE_FILE_PATH))
            except Exception as save_err:
                LOGGER.warning("semantic_tool_index_save_failed", error=str(save_err))

            self._initialized = True
            ms = round((time.perf_counter() - t0) * 1000, 1)
            LOGGER.info("semantic_tool_index_initialized", tool_count=len(self._tool_names), ms=ms)

        except Exception as exc:
            LOGGER.error("semantic_tool_index_init_failed", error=str(exc))
            self._initialized = False

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.20,
        allowed_names: set[str] | None = None,
    ) -> list[Tuple[BaseTool, float]]:
        """
        Embed the user query in < 10ms and return top matching BaseTools with hybrid scores.
        """
        if not self._initialized or self._vectors is None or self._model is None:
            self.initialize()
            if not self._initialized or self._vectors is None:
                return self._fallback_keyword_search(query, top_k, allowed_names)

        clean_query = query.strip()
        if not clean_query:
            return []

        try:
            t0 = time.perf_counter()
            q_raw = np.array(list(self._model.embed([clean_query])), dtype=np.float32)[0]
            q_norm = np.linalg.norm(q_raw)
            if q_norm == 0:
                return []
            q_vec = q_raw / q_norm

            # Dense cosine similarities
            dense_scores = np.dot(self._vectors, q_vec)

            # Lexical / keyword boost for domain terms (Hinglish/English)
            q_words = set(clean_query.lower().split())
            final_scores: list[Tuple[int, float]] = []

            for idx, tool_name in enumerate(self._tool_names):
                score = float(dense_scores[idx])
                desc_text = TOOL_SEMANTIC_DESCRIPTIONS.get(tool_name, "").lower()
                
                # Boost if exact query keywords match curated semantic terms
                match_count = sum(1 for w in q_words if len(w) > 2 and w in desc_text)
                if match_count > 0:
                    score += match_count * 0.12  # Hybrid sparse boost

                final_scores.append((idx, score))

            final_scores.sort(key=lambda x: x[1], reverse=True)

            results: list[Tuple[BaseTool, float]] = []
            for idx, score in final_scores:
                if score < min_score and len(results) >= 2:
                    break
                tool_name = self._tool_names[idx]
                if allowed_names and tool_name not in allowed_names:
                    continue
                tool = self._tool_map.get(tool_name)
                if tool:
                    results.append((tool, score))
                if len(results) >= top_k:
                    break

            ms = round((time.perf_counter() - t0) * 1000, 2)
            LOGGER.debug("semantic_tool_search_complete", query=clean_query[:40], top_k=len(results), ms=ms)
            return results

        except Exception as exc:
            LOGGER.warning("semantic_tool_search_failed", error=str(exc))
            return self._fallback_keyword_search(query, top_k, allowed_names)

    def _fallback_keyword_search(
        self,
        query: str,
        top_k: int = 5,
        allowed_names: set[str] | None = None,
    ) -> list[Tuple[BaseTool, float]]:
        q_tokens = set(query.lower().split())
        scored: list[Tuple[BaseTool, float]] = []

        for name, tool in self._tool_map.items():
            if allowed_names and name not in allowed_names:
                continue
            desc = (name + " " + (tool.description or "") + " " + TOOL_SEMANTIC_DESCRIPTIONS.get(name, "")).lower()
            overlap = sum(1 for tok in q_tokens if tok in desc)
            if overlap > 0:
                score = overlap / max(len(q_tokens), 1)
                scored.append((tool, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def get_semantic_tools_for_query(
    query: str,
    top_k: int = 4,
    allowed_names: set[str] | None = None,
) -> list[BaseTool]:
    """Helper function to return list of top matching tools for a prompt."""
    index = SemanticToolIndex.get_instance()
    results = index.search(query, top_k=top_k, allowed_names=allowed_names)
    return [t for t, _score in results]
