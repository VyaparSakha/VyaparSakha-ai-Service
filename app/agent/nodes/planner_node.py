"""
app/agent/nodes/planner_node.py
===============================
Planner node — decides what steps are required to fulfill a complex goal.
It generates a structured plan without executing any tools.
"""

from __future__ import annotations

import asyncio
import json
import time
import structlog
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from app.agent.state import VyaparAgentState
from app.lib.llm import get_planner_llm, get_reasoning_llm, get_small_llm
from app.lib.cloudflare_ai import is_cloudflare_configured, getChatModel

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.planner")

_PLANNER_CAPABILITIES = """- Inventory tools: check stock levels, low stock alerts, stockout risk, dead stock, and product details.
- Sales & Profit tools: sales summary (revenue & profit), top selling products, sales trends, profit margin analysis, and sales anomalies.
- Forecasting & Restock tools: demand forecasts, restock priorities, and restock budget.
- Supplier & Buyer tools: supplier dues, buyer dues, pending payments, and purchase orders.
- Store Insights tools: store overview, daily action checklist, and business health metrics."""

_PLANNER_PROMPT = """You are the Vyapar Sakha Planner.
Analyze the user's complex request and build an actionable execution plan of 2 to 4 high-level steps.
Do NOT execute tools. Only recommend steps using available system capabilities:

Available capabilities:
{capabilities}

EXAMPLE:
User Request: "Audit my store inventory and prepare a restock recommendation."
Output:
{{
  "goal": "Perform inventory audit and generate prioritized restock recommendation",
  "steps": [
    "1. Gather low stock products and dead stock alerts using inventory tools.",
    "2. Fetch restocking priorities and demand forecast metrics.",
    "3. Synthesize findings into a structured report with risk badges and action items."
  ]
}}

Output your response strictly as JSON with no extra markdown formatting:
{{
  "goal": "Clear summary of the user's goal",
  "steps": [
    "Step 1...",
    "Step 2..."
  ]
}}
"""


async def planner_node(state: VyaparAgentState) -> Dict[str, Any]:
    prompt = state.get("user_prompt", "")
    if isinstance(prompt, str) and not prompt.strip():
        LOGGER.warning("planner_node_empty_prompt")
        raise ValueError("User message cannot be empty.")
    prompt = str(prompt).strip()
    store_id = state.get("store_id", "unknown")
    t0 = time.perf_counter()

    LOGGER.info("planner_node_start", store_id=store_id, prompt_len=len(prompt))

    if is_cloudflare_configured():
        llm = getChatModel()
    else:
        llm = get_planner_llm() or get_reasoning_llm() or get_small_llm()

    if not llm:
        LOGGER.error("planner_node_no_llm")
        return {
            "plan": {
                "goal": prompt,
                "steps": ["Analyze the request and use available tools to answer."],
            }
        }

    sys_content = _PLANNER_PROMPT.format(capabilities=_PLANNER_CAPABILITIES)

    messages = [
        SystemMessage(content=sys_content),
        HumanMessage(content=f"Create a plan for this request: {prompt}"),
    ]

    try:
        response = await asyncio.wait_for(
            llm.ainvoke(messages, config={"tags": ["hide_stream"]}),
            timeout=12.0,
        )
        content = response.content.strip()

        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        plan_data = json.loads(content.strip())
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        LOGGER.info(
            "planner_node_success",
            steps=len(plan_data.get("steps", [])),
            latency_ms=elapsed_ms,
        )

        return {"plan": plan_data}
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        LOGGER.warning("planner_node_fallback", error=str(exc)[:150], latency_ms=elapsed_ms)
        return {
            "plan": {
                "goal": prompt,
                "steps": ["Analyze the request and use available tools to answer."],
            }
        }
