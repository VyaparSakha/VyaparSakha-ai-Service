"""
app/agent/nodes/subgraph_router.py
===================================
The subgraph_router node sits between the tool node and observe node.

When the LLM calls one of the three subgraph stub tools
(invoke_morning_briefing, invoke_deep_inventory_audit,
invoke_smart_restock_order), the tool returns a dict with
``{"__subgraph__": "<name>", ...params}``.

This node detects that marker in the most recent ToolMessage,
runs the appropriate compiled subgraph, and writes the result
to ``VyaparAgentState.subgraph_result``.

If no subgraph was triggered this turn, the node is a no-op
pass-through (returns an empty patch).

SSE streaming notes
-------------------
Because the subgraph is invoked via ``subgraph.ainvoke()``, the
outer ``astream_events`` call on the main graph will emit:

  on_chain_start  (name = "morning_briefing" | "deep_inventory" | "smart_restock")
  on_chain_start  (name = "fetch_kpis" | "fetch_overview" | "fetch_priorities" ...)
  on_chain_end    (name = ...)
  on_chain_end    (name = "morning_briefing" | ...)

These events are intercepted by the SSE handler in store_ai_routes.py
and emitted as ``event: subgraph_start``, ``event: subgraph_step``,
and ``event: subgraph_end`` to the frontend.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import structlog
from langchain_core.messages import ToolMessage

from app.agent.subgraphs.morning_briefing.graph import morning_briefing_graph
from app.agent.subgraphs.deep_inventory.graph import deep_inventory_graph
from app.agent.subgraphs.smart_restock.graph import smart_restock_graph
from app.agent.subgraphs.web_research.graph import web_research_graph
from app.agent.subgraphs.document_generation.graph import document_generation_graph
from app.agent.subgraphs.email_composer.graph import email_composer_graph

from app.agent.state import make_tool_result

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.nodes.subgraph_router")


# Subgraph name → compiled graph + output key mapping
_SUBGRAPH_REGISTRY: dict[str, dict[str, Any]] = {
    "morning_briefing": {
        "graph": morning_briefing_graph,
        "output_key": "briefing_output",
    },
    "deep_inventory": {
        "graph": deep_inventory_graph,
        "output_key": "audit_output",
    },
    "smart_restock": {
        "graph": smart_restock_graph,
        "output_key": "restock_output",
    },
    "web_research": {
        "graph": web_research_graph,
        "output_key": "synthesis",
    },
    "document_generation": {
        "graph": document_generation_graph,
        "output_key": "file_metadata",
    },
    "email_composer": {
        "graph": email_composer_graph,
        "output_key": "email_output",
    },
}

_SUBGRAPH_TOOL_NAMES = {
    "invoke_morning_briefing",
    "invoke_deep_inventory_audit",
    "invoke_smart_restock_order",
    "invoke_document_generation",
    "invoke_email_composer",
    "web_research",
}



def _extract_subgraph_marker(messages: list) -> tuple[dict | None, str | None, str | None]:
    """
    Scan the recent ToolMessages for a subgraph marker dict.
    Returns (marker_dict, tool_message_id, tool_call_id) or (None, None, None).
    """
    for msg in reversed(messages):
        # Stop scanning when we reach a non-ToolMessage (end of this loop's tool block)
        if not getattr(msg, "type", "") == "tool" and not isinstance(msg, ToolMessage):
            break
            
        content = msg.content
        # ToolMessage content may be a string or list
        if isinstance(content, str):
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "__subgraph__" in data:
                    return data, msg.id, getattr(msg, "tool_call_id", None)
            except (json.JSONDecodeError, ValueError):
                pass
        elif isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict) and "__subgraph__" in chunk:
                    return chunk, msg.id, getattr(msg, "tool_call_id", None)
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    try:
                        data = json.loads(chunk.get("text", ""))
                        if isinstance(data, dict) and "__subgraph__" in data:
                            return data, msg.id, getattr(msg, "tool_call_id", None)
                    except (json.JSONDecodeError, ValueError):
                        pass
    return None, None, None


async def subgraph_router(state: dict) -> dict:
    """
    Detect and execute a subgraph if the last tool call was a subgraph stub.
    Returns a state patch with `subgraph_result` populated and the ToolMessage
    overwritten with the actual output.
    """
    messages = state.get("messages", [])
    store_ctx = state.get("store_context", {})
    user_ctx = state.get("user_context", {})

    marker, tool_msg_id, tool_call_id = _extract_subgraph_marker(messages)
    if not marker or not tool_msg_id:
        # No subgraph triggered — pass through
        return {}

    subgraph_name = marker.get("__subgraph__", "")
    entry = _SUBGRAPH_REGISTRY.get(subgraph_name)
    if not entry:
        LOGGER.warning("subgraph_router.unknown_subgraph", name=subgraph_name)
        return {}

    LOGGER.info("subgraph_router.dispatch", subgraph=subgraph_name)

    # Build input state from marker params + store context
    subgraph_input: dict[str, Any] = {
        "store_id": marker.get("store_id", state.get("store_id", "")),
        "user_id": state.get("user_id", ""),
        "currency": store_ctx.get("currency", "INR"),
        "low_stock_threshold": store_ctx.get("low_stock_threshold", 10),
        "lead_time_days": store_ctx.get("lead_time_days", 3),
        "owner_name": user_ctx.get("name", ""),
    }


    # Merge any extra params from the marker (e.g. expiry_alert_days, include_yellow)
    for key, val in marker.items():
        if key not in ("__subgraph__",) and key not in subgraph_input:
            subgraph_input[key] = val

    try:
        compiled_graph = entry["graph"]
        output_key = entry["output_key"]

        result_state = await compiled_graph.ainvoke(subgraph_input)

        if subgraph_name == "document_generation":
            file_meta = result_state.get("file_metadata", {})
            subgraph_output = {
                "success": result_state.get("status") == "success",
                "file": file_meta,
                "document_type": result_state.get("document_type", "excel"),
                "attempts": result_state.get("retry_count", 0) + 1,
                "validation": result_state.get("validation_result", {}),
                "error": result_state.get("error"),
                "preview": result_state.get("preview_data") or file_meta.get("preview", {}),
            }
        else:
            subgraph_output = result_state.get(output_key, {})

        LOGGER.info(
            "subgraph_router.complete",
            subgraph=subgraph_name,
            output_keys=list(subgraph_output.keys()) if isinstance(subgraph_output, dict) else [],
        )


        # Overwrite the stub ToolMessage with the actual output so the LLM can read it
        replacement_tool_msg = ToolMessage(
            content=json.dumps(subgraph_output, default=str),
            tool_call_id=tool_call_id or tool_msg_id,
            id=tool_msg_id,
            name=f"invoke_{subgraph_name}",
        )

        subgraph_tool_result = make_tool_result(
            call_id=tool_call_id or tool_msg_id,
            tool_name=f"invoke_{subgraph_name}",
            data=subgraph_output,
            loop_index=state.get("loop_count", 0),
            error=subgraph_output.get("error") if isinstance(subgraph_output, dict) else None,
        )

        return {
            "messages": [replacement_tool_msg],
            "tool_results": [subgraph_tool_result],
            "subgraph_result": {
                "subgraph_name": subgraph_name,
                "output": subgraph_output,
                "executed_at": datetime.utcnow().isoformat(),
            },
            "active_subgraph": "",  # Clear after execution
        }


    except Exception as exc:
        LOGGER.error(
            "subgraph_router.error",
            subgraph=subgraph_name,
            error=str(exc),
            exc_info=True,
        )
        error_msg = ToolMessage(
            content=f"Error executing subgraph {subgraph_name}: {str(exc)}",
            tool_call_id=tool_call_id or tool_msg_id,
            id=tool_msg_id,
            name=f"invoke_{subgraph_name}",
        )
        return {
            "messages": [error_msg],
            "subgraph_result": {
                "subgraph_name": subgraph_name,
                "output": {},
                "error": str(exc),
                "executed_at": datetime.utcnow().isoformat(),
            },
            "active_subgraph": "",
        }
