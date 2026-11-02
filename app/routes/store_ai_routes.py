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
import asyncio
import uuid
import structlog
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Any

from langgraph.errors import GraphInterrupt

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
# Shared SSE streaming helper — used by both /copilot/stream and /clarify
# ---------------------------------------------------------------------------


async def _stream_graph_events(
    graph,
    graph_input: Any,
    config: dict,
    *,
    tools_used_set: set,
    result: dict,
):
    """
    Async generator that processes LangGraph ``astream_events`` and yields
    SSE-formatted strings.

    Shared by the ``/copilot/stream`` (new run) and ``/clarify`` (resume)
    endpoints so inter­rupt detection and event handling stay consistent.

    Yields SSE event strings (``event: token\\ndata: ...\\n\\n`` etc.).

    After the generator completes, the mutable ``result`` dict is updated
    with:
        full_response       — accumulated assistant text (may be empty)
        interrupt_payloads  — list of langgraph Interrupt objects (empty
                              if the graph ran to completion)
        error               — set to the error string if an exception
                              was raised inside the generator
    """
    assistant_text_parts: list[str] = []
    interrupt_payloads: list = []
    visuals_collected: list = []

    # Track critic-reflection state to suppress intermediate think responses.
    # When critic FAILS → reflection → think retries, the intermediate think
    # tokens should NOT be streamed since they were rejected. Only the final
    # think pass (after critic PASS or graph end) is shown to the user.
    in_critic_retry: bool = False   # True while inside a reflection-retry cycle
    pending_text_parts: list[str] = []  # Buffer for tokens during retry

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

            # --- Interrupt detection ---
            # When langgraph.types.interrupt() fires, the runtime emits
            # an on_chain_stream event whose chunk contains a __interrupt__
            # key with the Interrupt objects. These are the payloads we
            # passed to interrupt().
            chunk = data.get("chunk")
            if isinstance(chunk, dict) and "__interrupt__" in chunk:
                for intr in chunk["__interrupt__"]:
                    interrupt_payloads.append(intr)
                # The stream will end shortly after this event — no
                # further processing needed for this event.
                continue

            # --- Track tool usage ---
            if kind == "on_tool_end":
                tools_used_set.add(name)

            # --- Critic-reflection cycle tracking ---
            # When reflection node starts, we're in a retry cycle.
            # Discard any already-buffered pending tokens (the rejected answer)
            # and mark that subsequent think tokens should be buffered, not streamed.
            if kind == "on_chain_start" and name == "reflection":
                in_critic_retry = True
                pending_text_parts = []  # Discard rejected intermediate answer
                # Emit a UI hint so the user sees something is happening
                yield f"event: token\ndata: {json.dumps({'text': '\n_[Improving answer...]_\n'})}\n\n"
                continue

            # When a new think node starts after reflection, clear the pending buffer
            if kind == "on_chain_start" and name == "think" and in_critic_retry:
                pending_text_parts = []
                continue

            # When critic node ends with PASS (goal_status == complete), exit retry mode
            # and flush the pending buffer to the stream.
            if kind == "on_chain_end" and name == "critic":
                output = data.get("output", {}) or {}
                critic_status = output.get("goal_status", "")
                if critic_status in ("complete", "review") and in_critic_retry:
                    # Critic passed — flush buffered tokens to the real stream
                    if pending_text_parts:
                        flushed = "".join(_flatten_text(p) for p in pending_text_parts)
                        # Send as a single replacement token so UI can replace buffered content
                        yield f"event: token\ndata: {json.dumps({'text': flushed})}\n\n"
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

                # Suppress tokens from non-conversational nodes or hide_stream
                if "hide_stream" in tags or (node_name and node_name not in {"think", "respond"}):
                    continue

                msg_chunk = event["data"]["chunk"]
                if msg_chunk.content:
                    text = _flatten_text(msg_chunk.content)
                    if in_critic_retry:
                        # Buffer tokens during retry — don't stream intermediate rejected answers
                        pending_text_parts.append(text)
                    else:
                        assistant_text_parts.append(text)
                        yield f"event: token\ndata: {json.dumps({'text': text})}\n\n"

            # --- Tool call events ---
            elif kind == "on_tool_start":
                tool_input = data.get("input", {}) or {}
                tool_payload = json.dumps(
                    {
                        "id": f"tool-{name}-{len(tools_used_set)}",
                        "name": name,
                        "args": tool_input,
                        "status": "running",
                    },
                    default=_json_default,
                )
                yield f"event: tool_call\ndata: {tool_payload}\n\n"

            elif kind == "on_tool_end":
                tool_output = data.get("output", {})
                raw_val = getattr(tool_output, "artifact", None) or getattr(tool_output, "content", tool_output)
                if hasattr(raw_val, "model_dump"):
                    tool_result = raw_val.model_dump()
                elif hasattr(raw_val, "dict") and callable(raw_val.dict):
                    tool_result = raw_val.dict()
                else:
                    tool_result = raw_val
                tool_payload = json.dumps(
                    {
                        "id": f"tool-{name}-{len(tools_used_set)}",
                        "name": name,
                        "result": tool_result,
                        "status": "completed",
                    },
                    default=_json_default,
                )
                yield f"event: tool_call\ndata: {tool_payload}\n\n"

                # --- Emit visual spec for analytical tools (charts, tables, KPI cards) ---
                try:
                    from app.agent.tools.visual_builders import build_visual_from_tool
                    visual_spec = build_visual_from_tool(name, tool_result) or build_visual_from_tool(name, tool_output)
                    if visual_spec:
                        visuals_collected.append(visual_spec)
                        yield f"event: visual\ndata: {json.dumps(visual_spec, default=_json_default)}\n\n"
                except Exception as viz_err:
                    LOGGER.warning("visual_spec_build_failed", tool=name, error=str(viz_err))

                # --- Emit refresh / navigate for mutation tools ---
                if name in MUTATION_TOOL_MAP:
                    mutation_meta = MUTATION_TOOL_MAP[name]
                    refresh_payload = json.dumps({"section": mutation_meta["section"]}, default=_json_default)
                    yield f"event: refresh\ndata: {refresh_payload}\n\n"
                    if mutation_meta.get("route"):
                        nav_payload = json.dumps({"route": mutation_meta["route"]}, default=_json_default)
                        yield f"event: navigate\ndata: {nav_payload}\n\n"

                # --- Handle explicit navigation tool ---
                if name == "tool_navigate_page":
                    # For LangGraph, `data.get("input")` is the input dictionary passed to the tool
                    tool_input = data.get("input", {})
                    page = tool_input.get("page")
                    if page:
                        nav_payload = json.dumps({"page": page}, default=_json_default)
                        yield f"event: navigate\ndata: {nav_payload}\n\n"

                # --- Handle Billing Tools ---
                if name == "tool_add_billing_item":
                    # We need the barcode from the result
                    tool_result = data.get("output", {})
                    tool_input = data.get("input", {})
                    if isinstance(tool_result, dict) and tool_result.get("success"):
                        add_payload = json.dumps({
                            "barcode": tool_result.get("barcode"),
                            "quantity": tool_input.get("quantity", 1)
                        }, default=_json_default)
                        yield f"event: billing_add\ndata: {add_payload}\n\n"

                if name == "tool_generate_bill":
                    tool_input = data.get("input", {})
                    gen_payload = json.dumps({
                        "paymentMethod": tool_input.get("payment_method", "cash")
                    }, default=_json_default)
                    yield f"event: billing_generate\ndata: {gen_payload}\n\n"

            # --- Memory query notifications ---
            elif kind == "on_chain_start" and name == "memory_query":
                yield f"event: token\ndata: {json.dumps({'text': '\n_[Loading long-term memory...]_\n'})}\n\n"

            elif kind == "on_chain_end" and name == "memory_query":
                yield f"event: token\ndata: {json.dumps({'text': '\n_[Memory loaded.]_\n'})}\n\n"

            # --- Planning node notifications & structured tool toggler ---
            elif kind == "on_chain_start" and name == "planner":
                yield f"event: token\ndata: {json.dumps({'text': '\n_[Formulating execution plan...]_\n'})}\n\n"

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

                        plan_tool_payload = json.dumps(
                            {
                                "id": "tool-planner-plan",
                                "name": "Goal & Strategy Plan",
                                "args": {"Goal": goal_text} if goal_text else {},
                                "result": steps_dict,
                                "status": "completed",
                            },
                            default=_json_default,
                        )
                        tools_used_set.add("planner")
                        yield f"event: tool_call\ndata: {plan_tool_payload}\n\n"
                        yield f"event: token\ndata: {json.dumps({'text': '\n_[Plan ready.]_\n'})}\n\n"

            # --- Subgraph events ---
            elif kind == "on_chain_start" and name in {"morning_briefing", "deep_inventory", "smart_restock", "web_research", "document_generation"}:
                yield f"event: subgraph_start\ndata: {json.dumps({'name': name})}\n\n"
            
            elif kind == "on_chain_start" and name in {
                "fetch_kpis", "fetch_alerts", "synthesize", 
                "fetch_overview", "fetch_risks", "fetch_restock", 
                "fetch_priorities", "fetch_supplier_context", "build_order",
                "plan_research", "search_web", "select_sources", "fetch_sources",
                "evaluate_evidence", "refine_query",
                "prepare_document", "build_document_spec", "generate_document",
                "validate_document", "repair_document", "upload_file", "create_result"
            }:
                yield f"event: subgraph_step\ndata: {json.dumps({'name': name})}\n\n"
                
            elif kind == "on_chain_end" and name in {"morning_briefing", "deep_inventory", "smart_restock", "web_research", "document_generation"}:
                yield f"event: subgraph_end\ndata: {json.dumps({'name': name})}\n\n"


        # Flush any buffered retry tokens if not already flushed
        if pending_text_parts:
            flushed = "".join(_flatten_text(p) for p in pending_text_parts)
            yield f"event: token\ndata: {json.dumps({'text': flushed})}\n\n"
            assistant_text_parts.extend(pending_text_parts)
            pending_text_parts = []

        # --- Post-stream: populate result dict ---
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
        # Defensive: astream_events should surface interrupts via
        # __interrupt__ chunks, but if GraphInterrupt leaks to the
        # caller, we can still extract the interrupt values from
        # the exception's args[0] (a list of Interrupt objects).
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
        yield f"event: error\ndata: {json.dumps({'message': str(exc)}, default=_json_default)}\n\n"


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


@router.post("/{store_id}/copilot/stream")
async def get_copilot_stream(store_id: str, payload: CopilotStreamPayload, request: Request):
    """
    LangGraph agent Copilot endpoint (streaming SSE).

    Each logical conversation is identified by a ``chat_id``. Starting a new
    chat (no ``chat_id`` or an unknown one) creates a fresh LangGraph
    ``thread_id`` so the agent gets a clean context window. Chat history is
    persisted in MongoDB (``agent_chats`` + ``agent_chat_messages``).

    Long-term memory (mem0) persistence runs as a background task AFTER the
    SSE stream completes, so it never blocks the response.

    The frontend sends ``chat_id`` (preferred) or the legacy ``session_id``.
    """
    from app.agent import build_graph, get_checkpointer, make_initial_state
    from app.agent.nodes.think_node import _is_conversation_meaningful

    user_id = request.headers.get("x-user-id", "default_user")

    # Resolve the chat session (existing or new)
    chat_session = await get_or_create_chat_session(
        user_id=user_id,
        store_id=store_id,
        chat_id=payload.chat_id,
    )

    chat_id = chat_session.chat_id
    thread_id = chat_session.thread_id

    # Persist the user message
    await add_user_message(chat_id, payload.message)

    # Track tool usage (mutated in place by the shared SSE helper)
    tools_used_set: set[str] = set()

    async def event_generator():
        try:
            # ── Navigation fast-path ──────────────────────────────────────────
            # Detect explicit "go to X page" style requests and immediately
            # emit a navigate event + friendly reply, bypassing the full graph.
            nav_result = _detect_navigation_intent(payload.message)
            if nav_result:
                nav_route, nav_label = nav_result
                reply = f"Sure! Navigating you to the **{nav_label}** page now. 🚀"
                yield f"event: token\ndata: {json.dumps({'text': reply})}\n\n"
                nav_payload = json.dumps({"route": nav_route}, default=_json_default)
                yield f"event: navigate\ndata: {nav_payload}\n\n"

                await add_assistant_message(chat_id, reply, metadata={"intent": "navigate", "route": nav_route})
                chat_title = await ensure_chat_title(chat_id, payload.message)
                session_data = json.dumps({"chatId": chat_id, "threadId": thread_id, "title": chat_title, "needsClarification": False}, default=_json_default)
                yield f"event: session\ndata: {session_data}\n\n"
                yield "event: done\ndata: {}\n\n"
                return
            # -- End navigation fast-path --------------------------------------

            # Immediately emit session metadata with chat_id at the start of stream
            # so the client can transition URL from /chat to /chat/:chatId right away
            initial_session = json.dumps({
                "chatId": chat_id,
                "threadId": thread_id,
                "title": chat_session.title or "New Chat",
                "needsClarification": False,
            }, default=_json_default)
            yield f"event: session\ndata: {initial_session}\n\n"

            checkpointer = get_checkpointer()
            graph = build_graph(checkpointer=checkpointer)

            initial_state = make_initial_state(
                user_id=user_id,
                store_id=store_id,
                user_prompt=payload.message,
                thread_id=thread_id,
            )
            if payload.language:
                initial_state["user_preferences"] = {
                    **(initial_state.get("user_preferences") or {}),
                    "language": payload.language,
                }

            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}

            result: dict = {}

            # Stream graph events via the shared helper — yields SSE
            # strings for tokens, tool calls, memory notifications, and
            # error events. Interrupts are captured in result["interrupt_payloads"].
            async for sse in _stream_graph_events(
                graph,
                initial_state,
                config,
                tools_used_set=tools_used_set,
                result=result,
            ):
                yield sse

            # --- Interrupt handling ---
            # When langgraph.types.interrupt() fires inside the interrupt_node,
            # the stream ends and the interrupt payloads are captured by the
            # helper. We detect this and emit a clarification SSE event.
            #
            # The graph is now PAUSED (not terminated) — the client must
            # call POST /{store_id}/clarify to resume with Command(resume=...).
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

                clarification_event = json.dumps({
                    "question": question,
                    "chatId": chat_id,
                    "threadId": thread_id,
                }, default=_json_default)
                yield f"event: clarification\ndata: {clarification_event}\n\n"

                await add_assistant_message(
                    chat_id,
                    f"[CLARIFICATION] {question}",
                    metadata={
                        "thread_id": thread_id,
                        "needs_clarification": True,
                    },
                )

                session_data = json.dumps({
                    "chatId": chat_id,
                    "threadId": thread_id,
                    "title": await ensure_chat_title(chat_id, payload.message),
                    "needsClarification": True,
                }, default=_json_default)
                yield f"event: session\ndata: {session_data}\n\n"
                yield "event: done\ndata: {}\n\n"
                return

            # --- Normal completion path ---
            full_response = result.get("full_response", "")

            # When no tokens were streamed (grader denial, greeting fast-path,
            # or any other short-circuit node), fall back to final_answer stored
            # in graph state and emit it as token events so the UI shows it.
            grader_denied = False
            if not full_response and not result.get("error"):
                try:
                    final_state = await graph.aget_state(config)
                    values = final_state.values or {}
                    grader_denied = bool(values.get("grader_denied"))
                    state_answer = values.get("final_answer") or ""
                    if state_answer:
                        yield f"event: token\ndata: {json.dumps({'text': state_answer})}\n\n"
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
                yield "event: done\ndata: {}\n\n"
                return

            # Persist the assistant response to chat history
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

            session_data = json.dumps({
                "chatId": chat_id,
                "threadId": thread_id,
                "title": chat_title,
                "needsClarification": False,
            }, default=_json_default)
            yield f"event: session\ndata: {session_data}\n\n"

            yield "event: done\ndata: {}\n\n"

            # After SSE stream is fully sent, persist to mem0 in background
            # so it never blocks the client response. Never persist a
            # refused (potentially harmful) exchange to long-term memory.
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
            error_data = json.dumps({"message": str(e)}, default=_json_default)
            yield f"event: error\ndata: {error_data}\n\n"
            yield "event: done\ndata: {}\n\n"

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


@router.post("/{store_id}/clarify")
async def clarify(
    store_id: str,
    request: Request,
    payload: ClarifyRequest,
):
    """
    User responds to a clarification question from the agent.

    This RESUMES the previously interrupted graph using
    ``Command(resume=<answer>)`` with the same ``thread_id``.
    LangGraph loads the saved checkpoint, re-executes the interrupt
    node from the beginning, and the ``interrupt()`` call returns the
    user's answer. The answer is stored in ``clarification_history``
    and the graph loops back to ``think`` so the LLM can incorporate
    the answer and continue.

    If the graph is not in an interrupted state (e.g. it already
    completed or the checkpoint was lost), an error SSE event is
    emitted and the client should start a new conversation.
    """
    from app.agent import build_graph, get_checkpointer
    from langgraph.types import Command
    from app.agent.nodes.think_node import _is_conversation_meaningful

    user_id = request.headers.get("x-user-id", "default_user")

    LOGGER.info(
        "clarify_request_received",
        store_id=store_id,
        chat_id=payload.chat_id,
        answer_preview=payload.answer[:100] if payload.answer else "",
        user_id=user_id,
    )

    # Resolve the chat session (existing — the graph must already be
    # interrupted on this thread for the resume to work).
    chat_session = await get_or_create_chat_session(
        user_id=user_id,
        store_id=store_id,
        chat_id=payload.chat_id,
    )

    chat_id = chat_session.chat_id
    thread_id = chat_session.thread_id

    LOGGER.info(
        "clarify_session_resolved",
        chat_id=chat_id,
        thread_id=thread_id,
        user_id=user_id,
    )

    # Persist the user's clarification answer to chat history
    await add_user_message(chat_id, payload.answer)

    LOGGER.info("clarify_message_persisted", chat_id=chat_id)

    graph = build_graph(checkpointer=get_checkpointer())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}

    tools_used_set: set[str] = set()

    async def event_generator():
        result: dict = {}

        try:
            LOGGER.info("clarify_guard_check_start", chat_id=chat_id, thread_id=thread_id)
            # ----------------------------------------------------------
            # Guard: verify the graph is actually interrupted before
            # attempting a resume. If state.next is non-empty, the
            # runtime has a pending node to execute (the interrupted
            # node). If it's empty, the graph already completed.
            # ----------------------------------------------------------
            try:
                current_state = await graph.aget_state(config)
                state_next = list(current_state.next) if current_state.next else []
                state_values_keys = list((current_state.values or {}).keys()) if current_state.values else []
                LOGGER.info(
                    "clarify_guard_check_result",
                    chat_id=chat_id,
                    thread_id=thread_id,
                    state_next=state_next,
                    has_values=bool(current_state.values),
                    value_keys=state_values_keys[:10],
                )
                if not current_state.next:
                    LOGGER.warning(
                        "clarify_no_pending_interrupt",
                        store_id=store_id,
                        chat_id=chat_id,
                        thread_id=thread_id,
                    )
                    error_data = json.dumps(
                        {"message": "No pending clarification to resume. Please start a new conversation."},
                        default=_json_default,
                    )
                    yield f"event: error\ndata: {error_data}\n\n"
                    yield "event: done\ndata: {}\n\n"
                    return
            except Exception as state_exc:
                LOGGER.warning(
                    "clarify_state_check_failed",
                    chat_id=chat_id,
                    thread_id=thread_id,
                    error=str(state_exc),
                    exc_info=True,
                )
                error_data = json.dumps(
                    {"message": "Could not load graph state for resume. Please start a new conversation."},
                    default=_json_default,
                )
                yield f"event: error\ndata: {error_data}\n\n"
                yield "event: done\ndata: {}\n\n"
                return

            LOGGER.info(
                "clarify_resume_start",
                chat_id=chat_id,
                thread_id=thread_id,
                answer_preview=payload.answer[:100],
            )

            # ----------------------------------------------------------
            # Resume the interrupted graph.
            #
            # Command(resume=<answer>) is the ONLY Command pattern
            # intended as input to invoke()/astream_events(). The
            # runtime matches the resume value to the pending interrupt()
            # call by index, and interrupt() returns the answer inside
            # the interrupt_node.
            # ----------------------------------------------------------
            async for sse in _stream_graph_events(
                graph,
                Command(resume=payload.answer),
                config,
                tools_used_set=tools_used_set,
                result=result,
            ):
                yield sse

            LOGGER.info(
                "clarify_resume_completed",
                chat_id=chat_id,
                thread_id=thread_id,
                interrupt_payloads=len(result.get("interrupt_payloads", [])),
                has_error=bool(result.get("error")),
            )

            # ----------------------------------------------------------
            # Handle recursive clarification (the LLM may still need
            # more information after incorporating the first answer).
            # ----------------------------------------------------------
            interrupt_payloads = result.get("interrupt_payloads", [])
            if interrupt_payloads:
                question = _extract_interrupt_question(interrupt_payloads)
                if not question:
                    question = "I need more information."

                LOGGER.info(
                    "clarify_recursive_clarification",
                    store_id=store_id,
                    chat_id=chat_id,
                )

                yield f"event: clarification\ndata: {json.dumps({'question': question, 'chatId': chat_id, 'threadId': thread_id}, default=_json_default)}\n\n"

                await add_assistant_message(
                    chat_id,
                    f"[CLARIFICATION] {question}",
                    metadata={"thread_id": thread_id, "needs_clarification": True},
                )

                session_data = json.dumps(
                    {
                        "chatId": chat_id,
                        "threadId": thread_id,
                        "title": await ensure_chat_title(chat_id, payload.answer),
                        "needsClarification": True,
                    },
                    default=_json_default,
                )
                yield f"event: session\ndata: {session_data}\n\n"
                yield "event: done\ndata: {}\n\n"
                return

            # ----------------------------------------------------------
            # Normal completion — the graph finished without further
            # clarification needs.
            # ----------------------------------------------------------
            full_response = result.get("full_response", "")

            # Fast-path fallback (no tokens may have streamed: grader denial, greeting, etc.)
            grader_denied = False
            if not full_response and not result.get("error"):
                try:
                    final_state = await graph.aget_state(config)
                    values = final_state.values or {}
                    grader_denied = bool(values.get("grader_denied"))
                    state_answer = values.get("final_answer") or ""
                    if state_answer:
                        yield f"event: token\ndata: {json.dumps({'text': state_answer})}\n\n"
                        full_response = state_answer
                except Exception:
                    pass

            if result.get("error"):
                yield "event: done\ndata: {}\n\n"
                return

            # Persist the assistant response to chat history
            if full_response:
                await add_assistant_message(
                    chat_id,
                    full_response,
                    metadata={"thread_id": thread_id, "visuals": result.get("visuals", [])},
                )

            # Emit session metadata
            session_data = json.dumps(
                {
                    "chatId": chat_id,
                    "threadId": thread_id,
                    "title": await ensure_chat_title(chat_id, payload.answer),
                },
                default=_json_default,
            )
            yield f"event: session\ndata: {session_data}\n\n"
            yield "event: done\ndata: {}\n\n"

            # Background memory persistence
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
            error_data = json.dumps({"message": str(e)}, default=_json_default)
            yield f"event: error\ndata: {error_data}\n\n"
            yield "event: done\ndata: {}\n\n"

    LOGGER.info("clarify_sse_response_returned", chat_id=chat_id)

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
