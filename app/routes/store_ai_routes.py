"""
app/routes/store_ai_routes.py
==============================
FastAPI routes for all AI features scoped to a specific store.

Endpoints:
    GET  /{store_id}/forecast       — demand forecast
    GET  /{store_id}/restock        — restock plan
    GET  /{store_id}/insights       — store insights
    GET  /{store_id}/summary        — store summary
    GET  /{store_id}/product/{id}   — per-product insight
    POST /{store_id}/copilot        — agent-powered Copilot (non-streaming)
    POST /{store_id}/copilot/stream — agent-powered Copilot (streaming SSE)
"""

import json
import time
import asyncio
import uuid
import structlog
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Any, Optional

from langgraph.errors import GraphInterrupt
from app.services.usage_collector import (
    UsageCollector,
    BillingUsageCallback,
    report_usage_to_express,
)

from app.services.aggregation_service import (
    get_forecast_for_store,
    get_restock_for_store,
    get_insights_for_store,
)
from app.services.chat_history_service import (
    get_or_create_chat_session,
    add_user_message,
    add_assistant_message,
    list_chat_sessions,
    get_chat_history,
    soft_delete_chat_session,
    ensure_chat_title,
)
from app.models.chat_history import ChatSessionModel, ChatMessageModel

router = APIRouter(tags=["store_ai"])

LOGGER = structlog.get_logger("vyaparsathi.ai.copilot_stream")


# ---------------------------------------------------------------------------
# Mutation tool → (refresh_section, navigate_route) mapping
# ---------------------------------------------------------------------------
# When the agent calls one of these write/delete tools the SSE stream will
# automatically emit:
#   1. event: refresh   { section }  — UI re-fetches that data section
#   2. event: navigate  { route }    — UI navigates to the relevant page
#      (navigate is optional; set to None to skip)
# Routes use :storeId as a placeholder resolved on the client.
# ---------------------------------------------------------------------------

MUTATION_TOOL_MAP: dict[str, dict] = {
    # Products / Inventory
    "tool_create_product":   {"section": "products",   "route": "/storeDashboard/:storeId/overview"},
    "tool_update_product":   {"section": "products",   "route": None},
    "tool_delete_product":   {"section": "products",   "route": None},
    "tool_adjust_stock":     {"section": "inventory",  "route": None},
    # Purchases
    "tool_create_purchase":  {"section": "purchases",  "route": "/storeDashboard/:storeId/purchases"},
    "tool_update_purchase":  {"section": "purchases",  "route": None},
    "tool_delete_purchase":  {"section": "purchases",  "route": None},
    "tool_receive_purchase": {"section": "purchases",  "route": None},
    # Sellers
    "tool_create_seller":    {"section": "sellers",    "route": "/storeDashboard/:storeId/sellers"},
    "tool_update_seller":    {"section": "sellers",    "route": None},
    "tool_delete_seller":    {"section": "sellers",    "route": None},
    # Buyers
    "tool_create_buyer":     {"section": "buyers",     "route": "/storeDashboard/:storeId/buyers"},
    "tool_update_buyer":     {"section": "buyers",     "route": None},
    "tool_delete_buyer":     {"section": "buyers",     "route": None},
    # Expenses
    "tool_create_expense":   {"section": "expenses",   "route": None},
    "tool_update_expense":   {"section": "expenses",   "route": None},
    "tool_delete_expense":   {"section": "expenses",   "route": None},
}


# ---------------------------------------------------------------------------
# Navigation intent keyword map
# ---------------------------------------------------------------------------
# Simple deterministic routing: if the user's message matches any keywords
# we immediately emit a navigate SSE event and a friendly reply,
# skipping the full LLM graph. This is fast, reliable, and zero-latency.
# Keywords are matched case-insensitively on the full lowercased message.
# ---------------------------------------------------------------------------

# Each entry: list of keyword groups (any match = positive)
# Value: (route_template, human_label)
_NAV_ROUTES = [
    (["analytics", "sales chart", "analytics page", "analytics section"],
     "/storeDashboard/:storeId/analytics", "Analytics"),
    (["seller", "sellers", "supplier", "suppliers", "vendor", "vendors"],
     "/storeDashboard/:storeId/sellers", "Sellers"),
    (["buyer", "buyers", "customer", "customers"],
     "/storeDashboard/:storeId/buyers", "Buyers"),
    (["purchase", "purchases", "purchase section", "purchase page", "purchase history"],
     "/storeDashboard/:storeId/purchases", "Purchases"),
    (["billing", "invoice", "bill", "pos", "point of sale"],
     "/storeDashboard/:storeId/billing", "Billing"),
    (["billing history", "bill history", "invoice history"],
     "/storeDashboard/:storeId/billing-history", "Billing History"),
    (["overview", "home", "dashboard home", "store home", "inventory", "product list", "products"],
     "/storeDashboard/:storeId/overview", "Overview"),
    (["ai dashboard", "ai page", "copilot page", "ai section"],
     "/storeDashboard/:storeId/ai-dashboard", "AI Dashboard"),
    (["automation", "automations", "automated"],
     "/storeDashboard/:storeId/automations", "Automations"),
    (["staff", "employees", "team", "workers"],
     "/storeDashboard/:storeId/staff", "Staff"),
    (["settings", "store settings", "setting"],
     "/storeDashboard/:storeId/settings", "Settings"),
]

_NAV_TRIGGER_VERBS = (
    "open", "go to", "navigate to", "take me to", "show", "show me",
    "switch to", "visit", "load", "bring up", "get to", "move to",
    "le chalo", "dikha", "dikhao", "ja", "jao",
)


def _detect_navigation_intent(message: str):
    """
    Returns (route, label) if the message is a clear navigation request,
    otherwise returns None.
    """
    lowered = message.lower().strip()
    has_nav_verb = any(verb in lowered for verb in _NAV_TRIGGER_VERBS)
    if not has_nav_verb:
        return None
    for keywords, route, label in _NAV_ROUTES:
        if any(kw in lowered for kw in keywords):
            return route, label
    return None


def _json_default(obj: Any) -> Any:
    """JSON serializer for objects not handled by the default encoder."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def _flatten_text(value) -> str:
    """
    Recursively coerce a value into a plain string.

    Some LLM providers stream ``chunk.content`` as a list of content
    blocks (e.g. ``[{"type": "text", "text": "..."}]``). Passing such a
    nested value directly to ``str.join`` raises::

        TypeError: sequence item 0: expected str instance, list found

    This helper walks nested lists/dicts and flattens them into a single
    string so every downstream ``.join`` call is safe.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "".join(_flatten_text(v) for v in value)
    if isinstance(value, dict):
        if "text" in value:
            return _flatten_text(value["text"])
        return json.dumps(value, default=_json_default)
    if value is None:
        return ""
    return str(value)


# ---------------------------------------------------------------------------
# Shared response model
# ---------------------------------------------------------------------------

class ApiResponse(BaseModel):
    data: Any
    message: str
    statusCode: int = 200


# ---------------------------------------------------------------------------
# Shared event streaming helpers — raw tuples and SSE formatter
# ---------------------------------------------------------------------------


async def _stream_graph_events_raw(
    graph,
    graph_input: Any,
    config: dict,
    *,
    tools_used_set: set,
    result: dict,
    usage_collector: Optional[UsageCollector] = None,
):
    """
    Async generator that processes LangGraph ``astream_events`` and yields
    (event_name, data_dict) tuples.

    Used by SSE endpoints (/copilot/stream, /clarify) and WebSocket endpoints
    (/ws/copilot, /ws/chat, /{store_id}/copilot/ws).
    """
    assistant_text_parts: list[str] = []
    interrupt_payloads: list = []
    visuals_collected: list = []

    # Track critic-reflection state to suppress intermediate think responses.
    in_critic_retry: bool = False
    pending_text_parts: list[str] = []

    try:
        LOGGER.info(
            "graph_stream_start",
            input_type=type(graph_input).__name__,
        )
        async for event in graph.astream_events(
            graph_input, config=config, version="v2"
        ):
            kind = event["event"]
            name = event["name"]
            data = event.get("data", {}) or {}

            LOGGER.debug(
                "graph_event_received",
                kind=kind,
                name=name,
            )

            # --- Model usage tracking (deduplicated by run_id) ---
            if kind == "on_chat_model_end":
                if usage_collector is not None:
                    try:
                        run_id = event.get("run_id")
                        output = data.get("output")
                        metadata = event.get("metadata", {}) or {}
                        usage_collector.add_model_usage(
                            run_id=str(run_id) if run_id else None,
                            output=output,
                            metadata=metadata,
                        )
                        # Incremental stage deduction & balance exhaustion check
                        node_name = metadata.get("langgraph_node") or name or "model_turn"
                        stage_res = await usage_collector.deduct_stage_usage_with_express(stage_name=node_name)
                        if stage_res and stage_res.get("isExhausted"):
                            LOGGER.warning(
                                "stream_credit_exhausted_halting_orchestration",
                                execution_id=usage_collector.execution_id,
                                node=node_name,
                            )
                            limit_msg = (
                                "\n\n⚠️ **AI Credit Limit Reached**: Your AI credits have been exhausted. "
                                "Further processing was stopped to prevent extra charges. "
                                "Please recharge your wallet in Settings to continue."
                            )
                            yield "token", {"text": limit_msg}
                            yield "error", {"code": "CREDIT_LIMIT_REACHED", "message": "AI credit limit reached."}
                            result["error"] = "CREDIT_LIMIT_REACHED"
                            assistant_text_parts.append(limit_msg)
                            break
                    except Exception as u_err:
                        LOGGER.warning("model_usage_capture_failed", error=str(u_err))

            # --- Interrupt detection ---
            chunk = data.get("chunk")
            if isinstance(chunk, dict) and "__interrupt__" in chunk:
                for intr in chunk["__interrupt__"]:
                    interrupt_payloads.append(intr)
                continue

            # --- Track tool usage ---
            if kind == "on_tool_end":
                tools_used_set.add(name)

            # --- Critic-reflection cycle tracking ---
            if kind == "on_chain_start" and name == "reflection":
                in_critic_retry = True
                pending_text_parts = []
                yield "token", {"text": "\n_[Improving answer...]_\n"}
                continue

            if kind == "on_chain_start" and name == "think" and in_critic_retry:
                pending_text_parts = []
                continue

            if kind == "on_chain_end" and name == "critic":
                output = data.get("output", {}) or {}
                critic_status = output.get("goal_status", "")
                if critic_status in ("complete", "review") and in_critic_retry:
                    if pending_text_parts:
                        flushed = "".join(_flatten_text(p) for p in pending_text_parts)
                        yield "token", {"text": flushed}
                        assistant_text_parts.extend(pending_text_parts)
                        pending_text_parts = []
                    if critic_status == "complete":
                        in_critic_retry = False
                continue

            # --- Stream LLM tokens ---
            if kind == "on_chat_model_stream":
                tags = event.get("tags", [])
                metadata = event.get("metadata", {}) or {}
                node_name = metadata.get("langgraph_node")

                if "hide_stream" in tags or (node_name and node_name not in {"think", "respond"}):
                    continue

                msg_chunk = event["data"]["chunk"]
                if msg_chunk.content:
                    text = _flatten_text(msg_chunk.content)
                    if in_critic_retry:
                        pending_text_parts.append(text)
                    else:
                        assistant_text_parts.append(text)
                        yield "token", {"text": text}

            # --- Tool call events ---
            elif kind == "on_tool_start":
                tool_input = data.get("input", {}) or {}
                tool_payload = {
                    "id": f"tool-{name}-{len(tools_used_set)}",
                    "name": name,
                    "args": tool_input,
                    "status": "running",
                }
                yield "tool_call", tool_payload

            elif kind == "on_tool_end":
                tool_output = data.get("output", {})
                raw_val = getattr(tool_output, "artifact", None) or getattr(tool_output, "content", tool_output)
                if hasattr(raw_val, "model_dump"):
                    tool_result = raw_val.model_dump()
                elif hasattr(raw_val, "dict") and callable(raw_val.dict):
                    tool_result = raw_val.dict()
                else:
                    tool_result = raw_val
                tool_payload = {
                    "id": f"tool-{name}-{len(tools_used_set)}",
                    "name": name,
                    "result": tool_result,
                    "status": "completed",
                }
                yield "tool_call", tool_payload

                try:
                    from app.agent.tools.visual_builders import build_visual_from_tool
                    visual_spec = build_visual_from_tool(name, tool_result) or build_visual_from_tool(name, tool_output)
                    if visual_spec:
                        visuals_collected.append(visual_spec)
                        yield "visual", visual_spec
                except Exception as viz_err:
                    LOGGER.warning("visual_spec_build_failed", tool=name, error=str(viz_err))

                if name in MUTATION_TOOL_MAP:
                    mutation_meta = MUTATION_TOOL_MAP[name]
                    yield "refresh", {"section": mutation_meta["section"]}
                    if mutation_meta.get("route"):
                        yield "navigate", {"route": mutation_meta["route"]}

                if name == "tool_navigate_page":
                    tool_input = data.get("input", {})
                    page = tool_input.get("page")
                    if page:
                        yield "navigate", {"page": page}

                if name == "tool_add_billing_item":
                    tool_result = data.get("output", {})
                    tool_input = data.get("input", {})
                    if isinstance(tool_result, dict) and tool_result.get("success"):
                        yield "billing_add", {
                            "barcode": tool_result.get("barcode"),
                            "quantity": tool_input.get("quantity", 1),
                        }

                if name == "tool_generate_bill":
                    tool_input = data.get("input", {})
                    yield "billing_generate", {
                        "paymentMethod": tool_input.get("payment_method", "cash"),
                    }

            # --- Memory query notifications ---
            elif kind == "on_chain_start" and name == "memory_query":
                yield "token", {"text": "\n_[Loading long-term memory...]_\n"}

            elif kind == "on_chain_end" and name == "memory_query":
                yield "token", {"text": "\n_[Memory loaded.]_\n"}

            # --- Planning node notifications ---
            elif kind == "on_chain_start" and name == "planner":
                yield "token", {"text": "\n_[Formulating execution plan...]_\n"}

            elif kind == "on_chain_end" and name == "planner":
                plan_output = data.get("output", {})
                if isinstance(plan_output, dict) and "plan" in plan_output:
                    plan_data = plan_output["plan"]
                    if isinstance(plan_data, dict):
                        goal_text = plan_data.get("goal", "")
                        steps_list = plan_data.get("steps", [])

                        steps_dict = {}
                        if isinstance(steps_list, list):
                            for idx, step_item in enumerate(steps_list):
                                clean_step = str(step_item).lstrip("0123456789. ")
                                steps_dict[f"Step {idx + 1}"] = clean_step
                        elif isinstance(steps_list, dict):
                            steps_dict = steps_list
                        else:
                            steps_dict = {"Plan": str(steps_list)}

                        plan_tool_payload = {
                            "id": "tool-planner-plan",
                            "name": "Goal & Strategy Plan",
                            "args": {"Goal": goal_text} if goal_text else {},
                            "result": steps_dict,
                            "status": "completed",
                        }
                        tools_used_set.add("planner")
                        yield "tool_call", plan_tool_payload
                        yield "token", {"text": "\n_[Plan ready.]_\n"}

            # --- Subgraph events ---
            elif kind == "on_chain_start" and name in {"morning_briefing", "deep_inventory", "smart_restock", "web_research", "document_generation"}:
                yield "subgraph_start", {"name": name}

            elif kind == "on_chain_start" and name in {
                "fetch_kpis", "fetch_alerts", "synthesize",
                "fetch_overview", "fetch_risks", "fetch_restock",
                "fetch_priorities", "fetch_supplier_context", "build_order",
                "plan_research", "search_web", "select_sources", "fetch_sources",
                "evaluate_evidence", "refine_query",
                "prepare_document", "build_document_spec", "generate_document",
                "validate_document", "repair_document", "upload_file", "create_result"
            }:
                yield "subgraph_step", {"name": name}

            elif kind == "on_chain_end" and name in {"morning_briefing", "deep_inventory", "smart_restock", "web_research", "document_generation"}:
                yield "subgraph_end", {"name": name}
                if name == "document_generation":
                    try:
                        chain_out = data.get("output", {})
                        from app.agent.tools.visual_builders import build_visual_from_tool
                        visual_spec = build_visual_from_tool("invoke_document_generation", chain_out)
                        if visual_spec and not any(v.get("type") == "document_preview" and v.get("filename") == visual_spec.get("filename") for v in visuals_collected):
                            visuals_collected.append(visual_spec)
                            yield "visual", visual_spec
                            tool_payload = {
                                "id": f"tool-invoke_document_generation-{len(tools_used_set)}",
                                "name": "invoke_document_generation",
                                "result": chain_out,
                                "status": "completed",
                            }
                            yield "tool_call", tool_payload
                    except Exception as viz_err:
                        LOGGER.warning("doc_gen_visual_spec_failed", error=str(viz_err))

            elif kind == "on_chain_end" and name == "subgraph_router":
                try:
                    router_output = data.get("output", {})
                    sub_res = router_output.get("subgraph_result", {})
                    if sub_res.get("subgraph_name") == "document_generation":
                        doc_output = sub_res.get("output", {})
                        from app.agent.tools.visual_builders import build_visual_from_tool
                        visual_spec = build_visual_from_tool("invoke_document_generation", doc_output)
                        if visual_spec and not any(v.get("type") == "document_preview" and v.get("filename") == visual_spec.get("filename") for v in visuals_collected):
                            visuals_collected.append(visual_spec)
                            yield "visual", visual_spec
                            tool_payload = {
                                "id": f"tool-invoke_document_generation-{len(tools_used_set)}",
                                "name": "invoke_document_generation",
                                "result": doc_output,
                                "status": "completed",
                            }
                            yield "tool_call", tool_payload
                except Exception as r_err:
                    LOGGER.warning("subgraph_router_stream_visual_failed", error=str(r_err))

        if pending_text_parts:
            flushed = "".join(_flatten_text(p) for p in pending_text_parts)
            yield "token", {"text": flushed}
            assistant_text_parts.extend(pending_text_parts)
            pending_text_parts = []

        result["full_response"] = "".join(
            _flatten_text(p) for p in assistant_text_parts
        )
        result["interrupt_payloads"] = interrupt_payloads
        result["visuals"] = visuals_collected

        LOGGER.info(
            "graph_stream_completed",
            full_response_len=len(result["full_response"]),
            interrupt_count=len(interrupt_payloads),
        )

    except GraphInterrupt as exc:
        LOGGER.warning(
            "graph_stream_interrupted",
            exc_info=True,
        )
        interrupts = exc.args[0] if exc.args else []
        for intr in interrupts:
            interrupt_payloads.append(intr)
        result["interrupt_payloads"] = interrupt_payloads
        result["visuals"] = visuals_collected
        result["full_response"] = "".join(
            _flatten_text(p) for p in assistant_text_parts
        )
    except (Exception, asyncio.CancelledError) as exc:
        LOGGER.error(
            "graph_stream_error",
            error=str(exc),
            exc_info=True,
        )
        result["error"] = str(exc)
        result["interrupt_payloads"] = []
        result["full_response"] = "".join(
            _flatten_text(p) for p in assistant_text_parts
        )
        yield "error", {"message": str(exc)}


async def _stream_graph_events(
    graph,
    graph_input: Any,
    config: dict,
    *,
    tools_used_set: set,
    result: dict,
    usage_collector: Optional[UsageCollector] = None,
):
    """
    Async generator that processes LangGraph events and yields SSE-formatted strings.
    Maintained for backward compatibility.
    """
    async for event_name, data in _stream_graph_events_raw(
        graph,
        graph_input,
        config,
        tools_used_set=tools_used_set,
        result=result,
        usage_collector=usage_collector,
    ):
        yield f"event: {event_name}\ndata: {json.dumps(data, default=_json_default)}\n\n"


def _extract_interrupt_question(interrupt_payloads: list) -> str:
    """Extract the clarification question from a list of Interrupt objects."""
    if not interrupt_payloads:
        return ""
    for intr in interrupt_payloads:
        val = intr.value
        if isinstance(val, dict):
            q = val.get("question", "")
            if q:
                return q
        else:
            text = str(val)
            if text:
                return text
    return ""


# ---------------------------------------------------------------------------
# Existing analytics endpoints (unchanged)
# ---------------------------------------------------------------------------

@router.get("/{store_id}/forecast", response_model=ApiResponse)
async def get_forecast(store_id: str):
    data = await get_forecast_for_store(store_id)
    return ApiResponse(data=data, message="Forecast generated successfully")


@router.get("/{store_id}/restock", response_model=ApiResponse)
async def get_restock(store_id: str):
    data = await get_restock_for_store(store_id)
    return ApiResponse(data=data, message="Restock plan generated successfully")


@router.get("/{store_id}/insights", response_model=ApiResponse)
async def get_insights(store_id: str):
    data = await get_insights_for_store(store_id)
    return ApiResponse(data=data, message="Insights generated successfully")


@router.get("/{store_id}/summary", response_model=ApiResponse)
async def get_summary(store_id: str):
    data = {"title": "Store summary", "summary": "Store summary data placeholder."}
    return ApiResponse(data=data, message="Store summary generated successfully")


@router.get("/{store_id}/product/{product_id}", response_model=ApiResponse)
async def get_product_insight(store_id: str, product_id: str):
    data = {"productId": product_id, "insight": "No specific insight available in local fallback."}
    return ApiResponse(data=data, message="Product insight generated successfully")


# ---------------------------------------------------------------------------
# Chat history endpoints
# ---------------------------------------------------------------------------

class ChatSessionItem(BaseModel):
    chat_id: str
    title: str
    message_count: int = 0
    created_at: str
    updated_at: str


class ChatSessionListResponse(BaseModel):
    chats: list[ChatSessionItem]
    total: int = 0


class ChatMessageItem(BaseModel):
    role: str
    content: str
    timestamp: str
    meta: dict[str, Any] = Field(default_factory=dict)


class ChatHistoryResponse(BaseModel):
    chat_id: str
    messages: list[ChatMessageItem]
    store_id: str | None = None


@router.get("/{store_id}/chats", response_model=ApiResponse)
async def list_chats(store_id: str, request: Request):
    """
    List all active chat sessions for the current user + store.
    """
    user_id = request.headers.get("x-user-id", "default_user")
    sessions = await list_chat_sessions(
        user_id=user_id,
        store_id=store_id,
        limit=100,
    )

    chats = [
        ChatSessionItem(
            chat_id=s.chat_id,
            title=s.title or "New Chat",
            message_count=s.message_count,
            created_at=s.created_at.isoformat(),
            updated_at=s.updated_at.isoformat(),
        )
        for s in sessions
    ]

    return ApiResponse(
        data=ChatSessionListResponse(chats=chats, total=len(chats)).model_dump(),
        message="Chat sessions retrieved successfully",
    )


@router.get("/memory/status", response_model=ApiResponse)
async def get_memory_status():
    """
    Diagnostic endpoint — returns mem0 configuration and health status.
    """
    from app.agent.memory import get_memory_status as _get_status

    status = _get_status()
    return ApiResponse(
        data=status,
        message="Memory status retrieved",
    )


@router.get("/{store_id}/chats/{chat_id}", response_model=ApiResponse)
async def get_chat(store_id: str, chat_id: str, request: Request):
    """
    Retrieve the full message history for a specific chat.
    """
    user_id = request.headers.get("x-user-id", "default_user")
    messages = await get_chat_history(chat_id, limit=200)

    # Verify the chat belongs to this user/store by checking the first message
    # or by looking up the chat session. We use get_chat_history which doesn't
    # enforce ownership, so do an explicit check here.
    from app.services.chat_history_service import get_chat_session
    session = await get_chat_session(chat_id, user_id=user_id)
    if not session:
        return ApiResponse(
            data=None,
            message="Chat not found",
            statusCode=404,
        )

    items = [
        ChatMessageItem(
            role=m.role,
            content=m.content,
            timestamp=m.timestamp.isoformat(),
            meta=m.metadata or {},
        )
        for m in messages
    ]

    return ApiResponse(
        data=ChatHistoryResponse(
            chat_id=chat_id,
            messages=items,
            store_id=session.store_id,
        ).model_dump(),
        message="Chat history retrieved successfully",
    )


@router.delete("/{store_id}/chats/{chat_id}", response_model=ApiResponse)
async def delete_chat(store_id: str, chat_id: str, request: Request):
    """
    Soft-delete a chat session.
    """
    user_id = request.headers.get("x-user-id", "default_user")
    from app.services.chat_history_service import get_chat_session
    session = await get_chat_session(chat_id, user_id=user_id)
    if not session or session.store_id != store_id:
        return ApiResponse(
            data=None,
            message="Chat not found",
            statusCode=404,
        )

    ok = await soft_delete_chat_session(chat_id)
    return ApiResponse(
        data={"deleted": ok},
        message="Chat deleted successfully" if ok else "Failed to delete chat",
    )


# ---------------------------------------------------------------------------
# Copilot — agent-powered endpoints
# ---------------------------------------------------------------------------

class CopilotStreamPayload(BaseModel):
    message: str
    chat_id: str | None = None
    # Legacy field kept for backwards compatibility; ignored if chat_id is provided.
    session_id: str | None = None
    language: str | None = "en"
    execution_id: str | None = None


async def _stream_copilot_events_core(
    *,
    store_id: str,
    payload: CopilotStreamPayload,
    user_id: str = "default_user",
    is_owner: bool = True,
    user_role: str = "Owner",
    permissions: Optional[list[str]] = None,
    execution_id: Optional[str] = None,
):
    """
    Core async generator that processes copilot execution and yields
    (event_name, data_dict) tuples.
    Shared by SSE endpoint (POST /{store_id}/copilot/stream) and WebSocket
    endpoints (/ws/copilot, /ws/chat, /{store_id}/copilot/ws).
    """
    from app.agent import build_graph, get_checkpointer, make_initial_state
    from app.agent.nodes.think_node import _is_conversation_meaningful

    exec_id = (
        payload.execution_id
        or execution_id
        or f"exec_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
    )
    usage_collector = UsageCollector(
        execution_id=exec_id,
        user_id=user_id,
        store_id=store_id,
        feature="copilot",
    )

    chat_session = await get_or_create_chat_session(
        user_id=user_id,
        store_id=store_id,
        chat_id=payload.chat_id,
    )

    chat_id = chat_session.chat_id
    thread_id = chat_session.thread_id

    await add_user_message(chat_id, payload.message)
    tools_used_set: set[str] = set()

    try:
        nav_result = _detect_navigation_intent(payload.message)
        if nav_result:
            nav_route, nav_label = nav_result
            reply = f"Sure! Navigating you to the **{nav_label}** page now. 🚀"
            yield "token", {"text": reply}
            yield "navigate", {"route": nav_route}

            await add_assistant_message(chat_id, reply, metadata={"intent": "navigate", "route": nav_route})
            chat_title = await ensure_chat_title(chat_id, payload.message)
            yield "session", {
                "chatId": chat_id,
                "threadId": thread_id,
                "title": chat_title,
                "needsClarification": False,
            }
            usage_summary = usage_collector.get_summary(execution_status="completed")
            yield "usage", usage_summary
            asyncio.create_task(report_usage_to_express(usage_summary))
            yield "done", {}
            return

        yield "session", {
            "chatId": chat_id,
            "threadId": thread_id,
            "title": chat_session.title or "New Chat",
            "needsClarification": False,
        }

        checkpointer = get_checkpointer()
        graph = build_graph(checkpointer=checkpointer)

        initial_state = make_initial_state(
            user_id=user_id,
            store_id=store_id,
            user_prompt=payload.message,
            thread_id=thread_id,
            user_role=user_role,
            is_owner=is_owner,
            permissions=permissions or [],
        )
        if payload.language:
            initial_state["user_preferences"] = {
                **(initial_state.get("user_preferences") or {}),
                "language": payload.language,
            }

        billing_cb = BillingUsageCallback(usage_collector)
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 100,
            "callbacks": [billing_cb],
        }

        result: dict = {}

        async for event_name, data in _stream_graph_events_raw(
            graph,
            initial_state,
            config,
            tools_used_set=tools_used_set,
            result=result,
            usage_collector=usage_collector,
        ):
            yield event_name, data

        interrupt_payloads = result.get("interrupt_payloads", [])
        if interrupt_payloads:
            question = _extract_interrupt_question(interrupt_payloads)
            if not question:
                question = (
                    "I need some additional information to give you "
                    "an accurate answer. Could you help me with a few "
                    "details?"
                )

            LOGGER.info(
                "copilot_stream_clarification_needed",
                store_id=store_id,
                chat_id=chat_id,
                prompt=question[:200],
            )
            yield "clarification", {
                "question": question,
                "chatId": chat_id,
                "threadId": thread_id,
            }

            await add_assistant_message(
                chat_id,
                f"[CLARIFICATION] {question}",
                metadata={
                    "thread_id": thread_id,
                    "needs_clarification": True,
                },
            )

            chat_title = await ensure_chat_title(chat_id, payload.message)
            yield "session", {
                "chatId": chat_id,
                "threadId": thread_id,
                "title": chat_title,
                "needsClarification": True,
            }

            usage_summary = usage_collector.get_summary(execution_status="interrupted")
            yield "usage", usage_summary
            asyncio.create_task(report_usage_to_express(usage_summary))
            yield "done", {}
            return

        full_response = result.get("full_response", "")
        grader_denied = False
        if not full_response and not result.get("error"):
            try:
                final_state = await graph.aget_state(config)
                values = final_state.values or {}
                grader_denied = bool(values.get("grader_denied"))
                state_answer = values.get("final_answer") or ""
                if state_answer:
                    yield "token", {"text": state_answer}
                    full_response = state_answer
                    LOGGER.info(
                        "copilot_stream_fast_path_emitted",
                        store_id=store_id,
                        chat_id=chat_id,
                        intent=values.get("intent", "unknown"),
                        grader_denied=grader_denied,
                    )
            except Exception as exc:
                LOGGER.warning(
                    "copilot_stream_final_state_read_failed",
                    store_id=store_id,
                    chat_id=chat_id,
                    error=str(exc),
                )

        if result.get("error"):
            if result.get("error") == "CREDIT_LIMIT_REACHED":
                limit_full_response = result.get("full_response") or ""
                if limit_full_response:
                    await add_assistant_message(
                        chat_id,
                        limit_full_response,
                        metadata={"thread_id": thread_id, "stopped_reason": "CREDIT_LIMIT_REACHED"},
                    )
            usage_summary = usage_collector.get_summary(
                execution_status="exhausted" if result.get("error") == "CREDIT_LIMIT_REACHED" else "error"
            )
            yield "usage", usage_summary
            asyncio.create_task(report_usage_to_express(usage_summary))
            yield "done", {}
            return

        if full_response:
            await add_assistant_message(
                chat_id,
                full_response,
                metadata={
                    "thread_id": thread_id,
                    "grader_denied": grader_denied,
                    "visuals": result.get("visuals", []),
                },
            )

        chat_title = await ensure_chat_title(chat_id, payload.message)
        yield "session", {
            "chatId": chat_id,
            "threadId": thread_id,
            "title": chat_title,
            "needsClarification": False,
        }

        usage_summary = usage_collector.get_summary(execution_status="completed")
        yield "usage", usage_summary
        asyncio.create_task(report_usage_to_express(usage_summary))
        yield "done", {}

        should_persist = (
            False
            if grader_denied
            else _is_conversation_meaningful(
                user_prompt=payload.message,
                final_answer=full_response,
                tools_used=list(tools_used_set),
                loop_count=0,
            )
        )
        if should_persist and full_response:
            messages = [
                {"role": "user", "content": payload.message},
                {"role": "assistant", "content": full_response},
            ]
            asyncio.create_task(
                _persist_memory_background(
                    user_id=user_id,
                    store_id=store_id,
                    messages=messages,
                )
            )

    except (Exception, asyncio.CancelledError) as e:
        yield "error", {"message": str(e)}
        try:
            usage_summary = usage_collector.get_summary(execution_status="error")
            yield "usage", usage_summary
            asyncio.create_task(report_usage_to_express(usage_summary))
        except Exception:
            pass
        yield "done", {}


@router.post("/{store_id}/copilot/stream")
async def get_copilot_stream(store_id: str, payload: CopilotStreamPayload, request: Request):
    """
    LangGraph agent Copilot endpoint (streaming SSE).
    """
    user_id = request.headers.get("x-user-id", "default_user")
    is_owner_header = request.headers.get("x-is-owner", "true")
    is_owner = is_owner_header.lower() == "true"
    user_role = request.headers.get("x-user-role", "Owner" if is_owner else "Employee")
    permissions_raw = request.headers.get("x-user-permissions", "")
    permissions = [p.strip() for p in permissions_raw.split(",") if p.strip()]

    execution_id = (
        payload.execution_id
        or request.headers.get("x-execution-id")
        or f"exec_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
    )

    async def event_generator():
        async for event_name, data in _stream_copilot_events_core(
            store_id=store_id,
            payload=payload,
            user_id=user_id,
            is_owner=is_owner,
            user_role=user_role,
            permissions=permissions,
            execution_id=execution_id,
        ):
            yield f"event: {event_name}\ndata: {json.dumps(data, default=_json_default)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Clarify endpoint — user responds to a clarification question
# ---------------------------------------------------------------------------


class ClarifyRequest(BaseModel):
    chat_id: str
    answer: str
    execution_id: str | None = None


async def _stream_clarify_events_core(
    *,
    store_id: str,
    payload: ClarifyRequest,
    user_id: str = "default_user",
    execution_id: Optional[str] = None,
):
    """
    Core async generator that processes clarification resumes and yields
    (event_name, data_dict) tuples.
    Shared by SSE endpoint (POST /{store_id}/clarify) and WebSocket
    endpoints (/ws/copilot, /ws/chat, /{store_id}/copilot/ws).
    """
    from app.agent import build_graph, get_checkpointer
    from langgraph.types import Command
    from app.agent.nodes.think_node import _is_conversation_meaningful

    exec_id = (
        payload.execution_id
        or execution_id
        or f"exec_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
    )
    usage_collector = UsageCollector(
        execution_id=exec_id,
        user_id=user_id,
        store_id=store_id,
        feature="copilot",
    )

    LOGGER.info(
        "clarify_request_received",
        store_id=store_id,
        chat_id=payload.chat_id,
        answer_preview=payload.answer[:100] if payload.answer else "",
        user_id=user_id,
        execution_id=exec_id,
    )

    chat_session = await get_or_create_chat_session(
        user_id=user_id,
        store_id=store_id,
        chat_id=payload.chat_id,
    )

    chat_id = chat_session.chat_id
    thread_id = chat_session.thread_id

    await add_user_message(chat_id, payload.answer)

    graph = build_graph(checkpointer=get_checkpointer())
    billing_cb = BillingUsageCallback(usage_collector)
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 100,
        "callbacks": [billing_cb],
    }

    tools_used_set: set[str] = set()
    result: dict = {}

    try:
        try:
            current_state = await graph.aget_state(config)
            if not current_state.next:
                LOGGER.warning(
                    "clarify_no_pending_interrupt",
                    store_id=store_id,
                    chat_id=chat_id,
                    thread_id=thread_id,
                )
                yield "error", {"message": "No pending clarification to resume. Please start a new conversation."}
                yield "done", {}
                return
        except Exception as state_exc:
            LOGGER.warning(
                "clarify_state_check_failed",
                chat_id=chat_id,
                thread_id=thread_id,
                error=str(state_exc),
                exc_info=True,
            )
            yield "error", {"message": "Could not load graph state for resume. Please start a new conversation."}
            yield "done", {}
            return

        LOGGER.info(
            "clarify_resume_start",
            chat_id=chat_id,
            thread_id=thread_id,
            answer_preview=payload.answer[:100],
        )

        async for event_name, data in _stream_graph_events_raw(
            graph,
            Command(resume=payload.answer),
            config,
            tools_used_set=tools_used_set,
            result=result,
            usage_collector=usage_collector,
        ):
            yield event_name, data

        interrupt_payloads = result.get("interrupt_payloads", [])
        if interrupt_payloads:
            question = _extract_interrupt_question(interrupt_payloads)
            if not question:
                question = "I need more information."

            yield "clarification", {
                "question": question,
                "chatId": chat_id,
                "threadId": thread_id,
            }

            await add_assistant_message(
                chat_id,
                f"[CLARIFICATION] {question}",
                metadata={"thread_id": thread_id, "needs_clarification": True},
            )

            chat_title = await ensure_chat_title(chat_id, payload.answer)
            yield "session", {
                "chatId": chat_id,
                "threadId": thread_id,
                "title": chat_title,
                "needsClarification": True,
            }

            usage_summary = usage_collector.get_summary(execution_status="interrupted")
            yield "usage", usage_summary
            asyncio.create_task(report_usage_to_express(usage_summary))
            yield "done", {}
            return

        full_response = result.get("full_response", "")
        grader_denied = False
        if not full_response and not result.get("error"):
            try:
                final_state = await graph.aget_state(config)
                values = final_state.values or {}
                grader_denied = bool(values.get("grader_denied"))
                state_answer = values.get("final_answer") or ""
                if state_answer:
                    yield "token", {"text": state_answer}
                    full_response = state_answer
            except Exception:
                pass

        if result.get("error"):
            if result.get("error") == "CREDIT_LIMIT_REACHED":
                limit_full_response = result.get("full_response") or ""
                if limit_full_response:
                    await add_assistant_message(
                        chat_id,
                        limit_full_response,
                        metadata={"thread_id": thread_id, "stopped_reason": "CREDIT_LIMIT_REACHED"},
                    )
            usage_summary = usage_collector.get_summary(
                execution_status="exhausted" if result.get("error") == "CREDIT_LIMIT_REACHED" else "error"
            )
            yield "usage", usage_summary
            asyncio.create_task(report_usage_to_express(usage_summary))
            yield "done", {}
            return

        if full_response:
            await add_assistant_message(
                chat_id,
                full_response,
                metadata={"thread_id": thread_id, "visuals": result.get("visuals", [])},
            )

        chat_title = await ensure_chat_title(chat_id, payload.answer)
        yield "session", {
            "chatId": chat_id,
            "threadId": thread_id,
            "title": chat_title,
            "needsClarification": False,
        }

        usage_summary = usage_collector.get_summary(execution_status="completed")
        yield "usage", usage_summary
        asyncio.create_task(report_usage_to_express(usage_summary))
        yield "done", {}

        should_persist = _is_conversation_meaningful(
            user_prompt=payload.answer,
            final_answer=full_response,
            tools_used=list(tools_used_set),
            loop_count=0,
        )
        if should_persist and full_response:
            messages = [
                {"role": "user", "content": payload.answer},
                {"role": "assistant", "content": full_response},
            ]
            asyncio.create_task(
                _persist_memory_background(
                    user_id=user_id,
                    store_id=store_id,
                    messages=messages,
                )
            )

    except (Exception, asyncio.CancelledError) as e:
        LOGGER.error(
            "clarify_stream_error",
            chat_id=chat_id,
            thread_id=thread_id,
            error=str(e),
            exc_info=True,
        )
        yield "error", {"message": str(e)}
        try:
            usage_summary = usage_collector.get_summary(execution_status="error")
            yield "usage", usage_summary
            asyncio.create_task(report_usage_to_express(usage_summary))
        except Exception:
            pass
        yield "done", {}


@router.post("/{store_id}/clarify")
async def clarify(
    store_id: str,
    request: Request,
    payload: ClarifyRequest,
):
    """
    User responds to a clarification question from the agent (streaming SSE).
    """
    user_id = request.headers.get("x-user-id", "default_user")
    execution_id = (
        payload.execution_id
        or request.headers.get("x-execution-id")
        or f"exec_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
    )

    async def event_generator():
        async for event_name, data in _stream_clarify_events_core(
            store_id=store_id,
            payload=payload,
            user_id=user_id,
            execution_id=execution_id,
        ):
            yield f"event: {event_name}\ndata: {json.dumps(data, default=_json_default)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# WebSocket Copilot & Chat Endpoints (Zero Proxy Buffering)
# ---------------------------------------------------------------------------

async def _handle_copilot_websocket_session(
    websocket: WebSocket,
    default_store_id: Optional[str] = None,
    default_user_id: Optional[str] = None,
    default_is_owner: Optional[bool] = None,
    default_user_role: Optional[str] = None,
    default_permissions: Optional[list[str]] = None,
):
    """
    Session handler for Copilot WebSockets. Streams tokens and events
    sub-millisecond without proxy chunk buffering. Supports interactive
    turn-taking (message, clarify, ping).
    """
    await websocket.accept()
    LOGGER.info(
        "copilot_websocket_connected",
        store_id=default_store_id,
        user_id=default_user_id,
    )
    try:
        while True:
            try:
                data = await websocket.receive_json()
            except WebSocketDisconnect:
                LOGGER.info("copilot_websocket_client_disconnected", store_id=default_store_id)
                break
            except Exception as e:
                LOGGER.warning("copilot_websocket_receive_error", error=str(e))
                break

            action = data.get("action") or ("clarify" if "answer" in data else "message")
            if action == "ping":
                await websocket.send_json({"event": "pong", "data": {}})
                continue

            store_id = data.get("store_id") or default_store_id or ""
            user_id = data.get("user_id") or default_user_id or "default_user"
            is_owner = (
                data.get("is_owner")
                if data.get("is_owner") is not None
                else (default_is_owner if default_is_owner is not None else True)
            )
            user_role = data.get("user_role") or default_user_role or ("Owner" if is_owner else "Employee")
            perms_raw = data.get("permissions") or default_permissions or []
            if isinstance(perms_raw, str):
                perms = [p.strip() for p in perms_raw.split(",") if p.strip()]
            else:
                perms = list(perms_raw)

            execution_id = data.get("execution_id") or f"exec_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"

            if action == "clarify":
                clarify_payload = ClarifyRequest(
                    chat_id=data.get("chat_id") or "",
                    answer=data.get("answer") or "",
                    execution_id=execution_id,
                )
                async for event_name, event_data in _stream_clarify_events_core(
                    store_id=store_id,
                    payload=clarify_payload,
                    user_id=user_id,
                    execution_id=execution_id,
                ):
                    await websocket.send_json({"event": event_name, "data": event_data})
            else:
                stream_payload = CopilotStreamPayload(
                    message=data.get("message") or "",
                    chat_id=data.get("chat_id"),
                    session_id=data.get("session_id"),
                    language=data.get("language") or "en",
                    execution_id=execution_id,
                )
                async for event_name, event_data in _stream_copilot_events_core(
                    store_id=store_id,
                    payload=stream_payload,
                    user_id=user_id,
                    is_owner=is_owner,
                    user_role=user_role,
                    permissions=perms,
                    execution_id=execution_id,
                ):
                    await websocket.send_json({"event": event_name, "data": event_data})

    except WebSocketDisconnect:
        LOGGER.info("copilot_websocket_disconnected", store_id=default_store_id)
    except Exception as exc:
        LOGGER.error("copilot_websocket_error", error=str(exc), exc_info=True)
        try:
            await websocket.send_json({"event": "error", "data": {"message": str(exc)}})
            await websocket.send_json({"event": "done", "data": {}})
        except Exception:
            pass


@router.websocket("/ws/copilot")
async def copilot_websocket_endpoint(
    websocket: WebSocket,
    store_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    is_owner: Optional[bool] = Query(None),
    user_role: Optional[str] = Query(None),
    permissions: Optional[str] = Query(None),
):
    perms_list = [p.strip() for p in permissions.split(",") if p.strip()] if permissions else None
    await _handle_copilot_websocket_session(
        websocket=websocket,
        default_store_id=store_id,
        default_user_id=user_id,
        default_is_owner=is_owner,
        default_user_role=user_role,
        default_permissions=perms_list,
    )


@router.websocket("/ws/chat")
async def chat_websocket_endpoint(
    websocket: WebSocket,
    store_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    is_owner: Optional[bool] = Query(None),
    user_role: Optional[str] = Query(None),
    permissions: Optional[str] = Query(None),
):
    perms_list = [p.strip() for p in permissions.split(",") if p.strip()] if permissions else None
    await _handle_copilot_websocket_session(
        websocket=websocket,
        default_store_id=store_id,
        default_user_id=user_id,
        default_is_owner=is_owner,
        default_user_role=user_role,
        default_permissions=perms_list,
    )


@router.websocket("/{store_id}/copilot/ws")
async def store_copilot_websocket_endpoint(
    websocket: WebSocket,
    store_id: str,
    user_id: Optional[str] = Query(None),
    is_owner: Optional[bool] = Query(None),
    user_role: Optional[str] = Query(None),
    permissions: Optional[str] = Query(None),
):
    perms_list = [p.strip() for p in permissions.split(",") if p.strip()] if permissions else None
    await _handle_copilot_websocket_session(
        websocket=websocket,
        default_store_id=store_id,
        default_user_id=user_id,
        default_is_owner=is_owner,
        default_user_role=user_role,
        default_permissions=perms_list,
    )


# ---------------------------------------------------------------------------
# Background memory persistence
# ---------------------------------------------------------------------------

async def _persist_memory_background(
    *,
    user_id: str,
    store_id: str,
    messages: list[dict],
) -> None:
    """
    Persist conversation to Pinecone long-term memory in the background.

    Runs after the SSE stream has completed and the client has received
    the response. Failures are logged but never affect the client.
    """
    from app.agent.memory import process_and_persist_memory
    import structlog

    logger = structlog.get_logger("vyaparsathi.ai.memory.background")

    try:
        res = await process_and_persist_memory(
            user_id=user_id,
            store_id=store_id,
            messages=messages,
            store_ids=[store_id],
        )
        logger.info(
            "background_memory_persist_complete",
            user_id=user_id,
            store_id=store_id,
            user_ok=res.get("user_ok", True),
            store_ok=res.get("store_ok", True),
            multi_store_ok=res.get("multi_store_ok", True),
        )
    except Exception as exc:
        logger.warning(
            "background_memory_persist_failed",
            user_id=user_id,
            store_id=store_id,
            error=str(exc),
            exc_info=True,
        )
