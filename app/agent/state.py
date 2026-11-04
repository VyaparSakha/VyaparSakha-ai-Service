"""
app/agent/state.py
==================
VyaparAgentState — the single source of truth for the Vyapar Copilot
LangGraph agent. Every node reads from and writes to this state.

Memory architecture
-------------------
┌─────────────────────────────────────────────────────────────────┐
│  SHORT-TERM  (current request only)                             │
│    messages        → LangGraph add_messages reducer             │
│    tool_results    → operator.add accumulator                   │
│    *_context dicts → each tool writes to its own bucket         │
├─────────────────────────────────────────────────────────────────┤
│  LONG-TERM / USER  (per user_id, via mem0)                      │
│    user_preferences → language, tone, detail level, etc.        │
├─────────────────────────────────────────────────────────────────┤
│  LONG-TERM / STORE (per store_id, isolated via mem0)            │
│    store_knowledge  → product patterns, category notes, etc.    │
├─────────────────────────────────────────────────────────────────┤
│  CROSS-SESSION CHECKPOINT (MongoDB LangGraph Saver)             │
│    Full state snapshot keyed by thread_id = user_id+store_id    │
└─────────────────────────────────────────────────────────────────┘

Loop-control contract
---------------------
The agent loops as long as goal_status != "complete" | "failed"
AND loop_count < max_loops (hard ceiling: 5).

Each loop:
  1. think node   → LLM decides next tool(s) or marks goal complete
  2. tool node    → executes selected tools, writes to *_context buckets
  3. observe node → updates state, increments loop_count, clears
                    tools_called_this_loop
  4. router       → if goal_status == "complete" → respond node → END
                    else if loop_count >= max_loops → respond node (forced)
                    else → think node (next loop)

user_id source
--------------
The Express API gateway injects the authenticated user ID into every
proxied request as the HTTP header ``x-user-id``. FastAPI reads it
via Depends(get_user_id_from_header) and passes it into the initial
state before invoking the graph.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

GoalStatus = Literal["pending", "in_progress", "review", "reflect", "complete", "failed"]
Intent = Literal["live_data", "memory", "conversation_recap", "mixed", "general"]

# ---------------------------------------------------------------------------
# ToolCall — describes a single tool invocation the LLM wants to make
# ---------------------------------------------------------------------------


class ToolCall(TypedDict):
    """A single tool invocation requested by the LLM during the think step."""

    tool_name: str
    """Registered name of the tool (must exist in the tool registry)."""

    arguments: dict[str, Any]
    """Keyword arguments to pass to the tool function."""

    call_id: str
    """Unique ID for this call — used to match results back to calls."""


# ---------------------------------------------------------------------------
# ToolResult — the structured output returned after a tool executes
# ---------------------------------------------------------------------------


class ToolResult(TypedDict):
    """Structured output produced after a tool finishes executing."""

    call_id: str
    """Matches the ToolCall.call_id that triggered this result."""

    tool_name: str
    """Name of the tool that produced this result."""

    success: bool
    """True if the tool executed without error."""

    data: Any
    """The actual payload returned by the tool (dict, list, str, etc.)."""

    error: str | None
    """Human-readable error message if success=False, else None."""

    loop_index: int
    """Which loop iteration this result was produced in (0-based)."""


# ---------------------------------------------------------------------------
# VyaparAgentState — the authoritative state TypedDict
# ---------------------------------------------------------------------------


class VyaparAgentState(TypedDict):
    """
    Complete state for one Vyapar Copilot agent run.

    Fields are grouped into logical sections:
        1. Identity & Scope
        2. Goal Tracking (loop controller)
        3. Short-term Memory: Messages
        4. Tool Execution Tracking
        5. Context Buckets (tool output landing zones)
        6. Long-term Memory References (mem0)
        7. Final Output
    """

    # ──────────────────────────────────────────────────────────────────────
    # 1. Identity & Scope
    # ──────────────────────────────────────────────────────────────────────

    user_id: str
    """
    Authenticated user identifier.

    Source: ``x-user-id`` HTTP header injected by the Express proxy
    after the JWT auth middleware validates the token. This is the
    MongoDB _id of the user document.

    Used as:
      - Namespace key for mem0 user-preference memory
      - Part of the thread_id composite for MongoDB checkpoint
    """

    store_id: str
    """
    The store this conversation is scoped to.

    Source: URL path parameter ``/{store_id}/copilot``.

    Used as:
      - Namespace key for mem0 store-knowledge memory (isolated per store)
      - Part of the thread_id composite for MongoDB checkpoint
      - Passed to every inventory/sales/forecast tool call
    """

    user_role: str
    """
    Role of the user in this store context (e.g. "Owner", "Cashier", "Manager").
    Source: ``x-user-role`` HTTP header injected by Express proxy.
    """

    is_owner: bool
    """
    True if the user is the store owner; False if the user is an employee.
    Source: ``x-is-owner`` HTTP header injected by Express proxy.
    """

    permissions: list[str]
    """
    List of active permission strings assigned to this user for the current store.
    Source: ``x-user-permissions`` HTTP header injected by Express proxy.
    """

    user_context: dict[str, Any]
    """
    Lightweight snapshot of the authenticated user.

    Populated once by ``context_node`` (runs before the grader) via a
    single MongoDB query. Empty dict ``{}`` when the lookup fails —
    the graph continues gracefully without it.

    Keys:
        name  → str — display name, e.g. "Ramesh Kumar"
        email → str — user email (used for personalisation)
    """

    store_context: dict[str, Any]
    """
    Lightweight snapshot of the active store's metadata and settings.

    Populated once by ``context_node`` via a single MongoDB query.
    Empty dict ``{}`` when the lookup fails — the graph continues
    gracefully without it.

    Keys:
        name                → str  — store display name
        business_type       → str  — "retail" | "wholesale" | "both" | ...
        city                → str  — city from store address
        currency            → str  — settings.currency (default "INR")
        low_stock_threshold → int  — settings.lowStockThreshold (default 10)
        lead_time_days      → int  — settings.leadTimeDays (default 3)
    """

    thread_id: str
    """
    LangGraph checkpoint thread identifier.

    Strategy: ``"{user_id}:{store_id}:{chat_id}"`` where ``chat_id`` is a
    UUID identifying one logical chat. Each new chat gets a fresh
    ``thread_id`` so the agent starts with a clean context window.
    Checkpoints are persisted to MongoDB via the LangGraph checkpointer.

    Derived automatically in make_initial_state() when not provided
    explicitly (falls back to ``"{user_id}:{store_id}"`` for one permanent
    session per user/store pair).
    """

    subgraph_result: dict[str, Any]
    """
    Output from the last subgraph invocation. Written by the subgraph_router node.
    Keys: subgraph_name (str), output (dict), executed_at (str)
    """

    active_subgraph: str
    """
    Name of the currently executing subgraph ("morning_briefing" |
    "deep_inventory" | "smart_restock" | ""). Set by think node,
    cleared by subgraph_router after execution.
    """

    # ──────────────────────────────────────────────────────────────────────
    # 2. Goal Tracking — drives the loop
    # ──────────────────────────────────────────────────────────────────────

    user_prompt: str
    """
    The raw, unmodified user question/instruction.
    Never mutated after initial state creation.
    """

    goal: str
    """
    The normalized goal extracted from user_prompt by the think node on
    the first loop. The LLM rephrases the prompt into a concrete,
    tool-oriented goal statement so subsequent loops stay on track.

    Example:
        user_prompt  -> "which products should I restock urgently?"
        goal         -> "identify products with red restock priority and
                         provide specific quantity recommendations"
    """

    goal_status: GoalStatus
    """
    Current completion status of the goal. Controls the loop router.

      pending     -> initial state, think node hasn't run yet
      in_progress -> think node has set the goal, tools are being called
      complete    -> LLM confirms goal is fully answered
      failed      -> max_loops exceeded or unrecoverable error
    """

    loop_count: int
    """
    Number of completed think->tool->observe cycles.
    Incremented by the observe node at the end of each loop.
    """

    max_loops: int
    """
    Hard ceiling on the number of loops. Default: 5.
    If loop_count reaches max_loops the router forces goal_status="failed"
    and sends the agent to the respond node with whatever it has.

    Configured globally via DEFAULT_MAX_LOOPS (not per-request).
    """

    # ──────────────────────────────────────────────────────────────────────
    # 3. Short-term Memory: Conversation Messages
    # ──────────────────────────────────────────────────────────────────────

    messages: Annotated[list[BaseMessage], add_messages]
    """
    The conversation message list for this agent run.

    Uses LangGraph's built-in ``add_messages`` reducer which:
      - Appends new messages instead of replacing the list
      - Deduplicates by message id if the same message is added twice
      - Is safe for parallel node writes

    Contains: HumanMessage (user prompt), AIMessages (think node outputs),
    ToolMessages (tool results formatted for LLM context), SystemMessages
    (injected memory context at loop start).
    """

    # ──────────────────────────────────────────────────────────────────────
    # 4. Tool Execution Tracking
    # ──────────────────────────────────────────────────────────────────────

    pending_tool_calls: list[ToolCall]
    """
    Tool calls the LLM requested in the latest think step.
    Written by: think node.
    Consumed and cleared by: tool node.
    """

    tool_error_counts: dict[str, int]
    """
    Tracks how many times a tool has failed.
    Key: tool_name, Value: failure_count
    Used for circuit breaking to prevent infinite loops.
    """

    candidate_products: list[dict[str, Any]]
    """
    The results of discovery/filtering operations before detailed analysis.
    Stores a compact representation of candidate records (e.g. low stock products).
    """

    discovery_metrics: dict[str, Any]
    """
    Metrics and metadata from the last discovery operation.
    E.g. {"products_scanned": 20000, "candidates_found": 350, "final_candidates": 20}
    """

    tool_results: Annotated[list[ToolResult], operator.add]
    """
    All tool results accumulated across every loop of this run.

    Uses ``operator.add`` as the reducer — each loop appends new
    ToolResult items without clearing previous ones. This gives the
    LLM full visibility into everything gathered so far when it
    reasons in later loops.
    """

    tools_called_this_loop: list[str]
    """
    Tool names called in the CURRENT loop iteration.
    Cleared to [] at the start of each new loop by the observe node.
    Prevents the LLM from calling the same tool twice in one cycle.

    Example: if loop 1 already called "get_inventory_summary",
    the think node prompt will exclude it from available options
    for that loop.
    """

    available_tools: list[str]
    """
    Registry of all tool names the agent can call.
    Set once during graph initialization, never mutated at runtime.

    Current read-only tools:
      - get_inventory_summary
      - get_low_stock_products
      - get_sales_summary
      - get_top_selling_products
      - get_forecast_summary
      - get_restock_priorities
      - get_store_insights
    """

    # ──────────────────────────────────────────────────────────────────────
    # 5. Context Buckets (tool output landing zones)
    # ──────────────────────────────────────────────────────────────────────

    inventory_context: dict[str, Any]
    """
    Data written by inventory tools (get_inventory_summary,
    get_low_stock_products, etc.).

    Structure example:
        {
          "total_products": 42,
          "low_stock": [...],
          "out_of_stock": [...],
          "fetched_at": "2026-09-10T17:00:00"
        }
    """

    sales_context: dict[str, Any]
    """
    Data written by sales tools (get_sales_summary,
    get_top_selling_products, etc.).

    Structure example:
        {
          "total_sales_30d": 152,
          "revenue_30d": 48000.0,
          "top_products": [...],
          "fetched_at": "2026-09-10T17:00:00"
        }
    """

    forecast_context: dict[str, Any]
    """
    Data written by forecast tools (get_forecast_summary, etc.).

    Structure example:
        {
          "horizon_days": 7,
          "items": [...],
          "fetched_at": "2026-09-10T17:00:00"
        }
    """

    restock_context: dict[str, Any]
    """
    Data written by restock tools (get_restock_priorities, etc.).

    Structure example:
        {
          "red_priority": [...],
          "yellow_priority": [...],
          "green_priority": [...],
          "fetched_at": "2026-09-10T17:00:00"
        }
    """

    insights_context: list[dict[str, Any]]
    """
    Data written by the insights tool (get_store_insights).
    A flat list of InsightResult-shaped dicts, e.g.:
        [{"type": "dead_stock", "title": "...", "summary": "...", ...}]
    """

    # ──────────────────────────────────────────────────────────────────────
    # 6. Long-term Memory References (mem0)
    # ──────────────────────────────────────────────────────────────────────

    user_memory_loaded: bool
    """
    Flag: has the mem0 user-preference memory been fetched this session?
    Set to True by the memory_query node after loading.
    Prevents redundant mem0 API calls on every loop.
    """

    store_memory_loaded: bool
    """
    Flag: has the mem0 store-knowledge memory been fetched this session?
    Set to True by the memory_query node after loading.
    """

    memory_query_needed: bool
    """
    Flag set by the think node when it decides it needs to query memory.
    The memory_query node checks this and runs if True, then clears it.
    """

    intent: Intent
    """Request intent selected before the think node chooses tools."""

    intent_reason: str
    """Short internal reason for the selected request intent."""

    domains: list[str]
    """Active business domains identified for this request (e.g. ['sales'], ['inventory']). Used for dynamic tool pruning."""

    requires_planning: bool
    """Flag indicating whether this request is complex and requires a plan."""

    plan: dict[str, Any]
    """Structured plan generated by the planner node."""

    user_preferences: dict[str, Any]
    """
    User's persistent preferences fetched from mem0 (user memory store,
    namespace = user_id).

    Expected keys (set progressively as the user interacts):
        language     -> "hindi" | "english" | "marathi" | ...
        detail_level -> "brief" | "detailed"
        tone         -> "formal" | "casual"
        currency     -> "INR" (default)
    """

    store_knowledge: dict[str, Any]
    """
    Store-specific knowledge fetched from Pinecone (store memory store,
    namespace = store_id — each store is fully isolated).
    """

    multi_store_knowledge: dict[str, Any]
    """
    Multi-store enterprise knowledge fetched from Pinecone (scoped by user_id + store_ids).
    """

    retrieved_memories: list[dict[str, Any]]
    """
    Memories retrieved on-demand by the agent's search_memory tool during a run.
    Distinct from bootstrap memory (user_preferences / store_knowledge) which is
    pre-loaded at graph start. Accumulated across multiple search_memory calls;
    deduplicated by memory_id before injection into the LLM context.
    """

    memory_search_calls: int
    """
    Number of times the agent has invoked the search_memory tool in the current run.
    Used to enforce MAX_MEMORY_SEARCH_CALLS budget and prevent runaway retrieval.
    """

    # ──────────────────────────────────────────────────────────────────────
    # 7. Final Output
    # ──────────────────────────────────────────────────────────────────────

    final_answer: str
    """
    The complete, user-facing response generated by the respond node.
    Empty string until the agent reaches the respond node.
    The API endpoint returns this to the client.
    """

    response_metadata: dict[str, Any]
    """
    Audit trail and debug info attached to the response.

    Keys populated by respond node:
        loops_taken   -> int: how many loops were run
        tools_used    -> list[str]: distinct tool names called
        goal_status   -> final GoalStatus value
        goal          -> the normalized goal string
        memory_loaded -> {"user": bool, "store": bool}
        forced_stop   -> bool: True if max_loops was hit
    """

    error: str | None
    """
    Human-readable error message if something went critically wrong.
    None during normal operation.
    """

    critic_status: Literal["PASS", "FAIL", ""]
    """
    The result of the critic node evaluation.
    PASS: the answer is good.
    FAIL: the answer is incomplete or missing information.
    """

    critic_reason: str
    """Reason for the critic's status."""

    critic_missing_information: list[str]
    """Information the critic identified as missing."""

    reflection: str
    """Reflection generated by the agent when the critic fails."""

    reflection_count: int
    """Number of times the reflection node has run in this conversation turn."""

    should_persist_memory: bool
    """
    Flag set by the think node when the conversation contains meaningful
    content worth storing in long-term mem0 memory.

    Set to False for trivial exchanges ("hi", "hello", "ok") to avoid
    polluting memory with noise. Set to True when the user asks a real
    question, provides preferences, or the agent produces a substantive
    response with actual data/insights.

    The grader node also sets this to False when it denies a request, so a
    refused (potentially harmful) exchange is never persisted to mem0.
    """

    grader_denied: bool
    """
    Flag set by the grader (guardrail) node on the first graph step.

    True  -> the user prompt was classified as harmful and the graph
              terminated immediately with a refusal in ``final_answer``.
    False -> the prompt passed the safety check and the graph continues to
              the think node normally.
    """

    grader_reason: str
    """
    Short, human-readable explanation from the safety classifier.

    Populated by the grader node whenever it runs. Empty string when the
    prompt is safe or when classification produced no reason. Useful for
    audit/logging without exposing classifier internals to the user.
    """

    resume_from_clarification: bool
    """
    Flag set to True by the interrupt node when it stores the user's
    answer and returns, so the think node can detect it is continuing
    after a clarification exchange.

    Note: the primary resume mechanism is LangGraph's native
    interrupt()/Command(resume=...) — the grader and intent nodes
    do NOT re-run when the graph is resumed from an interrupt.
    This flag is kept for backward compatibility and as a signal
    to downstream nodes that a clarification exchange occurred.
    """

    needs_clarification: bool
    """
    Flag set by the interrupt node (or think node) when the agent
    cannot proceed confidently and needs input from the user.

    True  → the interrupt node has called langgraph.types.interrupt()
            with a question payload; the graph is PAUSED at the
            interrupt point. The route handler detects this and emits
            a ``clarification`` SSE event. The graph resumes when
            the client calls POST /{store_id}/clarify with the user's
            answer, which invokes ``Command(resume=<answer>)``.
    False → the agent does not need clarification (default).
    """

    clarification_prompt: str
    """
    The specific question the agent wants to ask the user.

    Populated by the think node when it decides clarification is needed
    (carried forward to the interrupt node, which passes it as the
    payload to langgraph.types.interrupt()). On resume, the interrupt
    node clears this to "".
    Empty string when no clarification is needed.
    """

    clarification_history: list[dict[str, Any]]
    """
    All clarification exchanges in the current graph run.

    Each entry is a dict:
        {
            "question": str,   ← what the agent asked
            "answer": str,     ← what the user said (may be empty if pending)
            "reason": str,     ← why the agent needed clarification
        }

    This list is seeded from the graph config (``clarification_history``)
    on each graph invocation so that answers from previous clarification
    turns are visible to the LLM when the graph resumes.
    """


# ---------------------------------------------------------------------------
# AgentConfig — runtime configuration (NOT part of the graph state)
# ---------------------------------------------------------------------------


class AgentConfig(TypedDict, total=False):
    """
    Runtime configuration for the Vyapar Copilot agent.

    Passed as the ``config`` argument to graph.invoke() / graph.astream().
    Not stored in VyaparAgentState — it lives outside the state to keep
    the state focused on data, not configuration.
    """

    max_loops: int
    """Hard loop ceiling. Default: 5. Currently a global server setting."""

    temperature: float
    """LLM temperature for the think node. Default: 0.3."""

    verbose: bool
    """If True, the observe node logs full state diffs to structlog."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_LOOPS: int = 5
"""
Global hard ceiling on agent loops. Default: 3.
Kept low to respect LLM rate limits while still allowing
memory queries + tool calls in a single run.
"""

from app.agent.tools.registry import VYAPAR_TOOLS

AVAILABLE_TOOLS: list[str] = [t.name for t in VYAPAR_TOOLS]
"""
Master registry of read-only tool names available to the agent.
All tools query MongoDB and return data — no write operations.
"""


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_initial_state(
    *,
    user_id: str,
    store_id: str,
    user_prompt: str,
    thread_id: str | None = None,
    user_role: str = "Owner",
    is_owner: bool = True,
    permissions: list[str] | None = None,
    clarification_history: list[dict[str, Any]] | None = None,
    resume_from_clarification: bool = False,
) -> VyaparAgentState:
    """
    Create a clean VyaparAgentState for a new agent run.

    Args:
        user_id:     Authenticated user ID (from x-user-id header).
        store_id:    Store being queried (from URL path param).
        user_prompt: Raw user question/instruction.
        thread_id:   Optional explicit LangGraph checkpoint thread id.
                     If omitted, derived as "{user_id}:{store_id}" — one
                     permanent session per (user, store) pair.
        user_role:   Role name ("Owner", "Cashier", etc.).
        is_owner:    Whether the user is the store owner.
        permissions: List of permission keys granted to this user.

    Returns:
        A fully initialized VyaparAgentState ready for graph.invoke().

    Note:
        When thread_id is provided (the normal case for chat-based
        conversations), it should be formatted as
        "{user_id}:{store_id}:{chat_id}" so each chat gets its own
        isolated checkpoint and context window.
    """
    from langchain_core.messages import HumanMessage

    if not thread_id:
        thread_id = f"{user_id}:{store_id}"

    user_permissions = permissions if permissions is not None else (["*"] if is_owner else [])

    return VyaparAgentState(
        # 1. Identity & RBAC Scope
        user_id=user_id,
        store_id=store_id,
        thread_id=thread_id,
        user_role=user_role,
        is_owner=is_owner,
        permissions=user_permissions,
        # 1b. Runtime context snapshots — populated by context_node
        user_context={},
        store_context={},
        subgraph_result={},
        active_subgraph="",
        # 2. Goal tracking
        user_prompt=user_prompt,
        goal="",
        goal_status="pending",
        loop_count=0,
        max_loops=DEFAULT_MAX_LOOPS,
        # 3. Messages: seed with the user's prompt as a HumanMessage
        messages=[HumanMessage(content=user_prompt)],
        # 4. Tool tracking
        pending_tool_calls=[],
        tool_error_counts={},
        tool_results=[],
        tools_called_this_loop=[],
        available_tools=AVAILABLE_TOOLS,
        candidate_products=[],
        discovery_metrics={},
        # 5. Context buckets — empty until tools populate them
        inventory_context={},
        sales_context={},
        forecast_context={},
        restock_context={},
        insights_context=[],
        # 6. Memory — not loaded yet
        user_memory_loaded=False,
        store_memory_loaded=False,
        multi_store_memory_loaded=False,
        memory_query_needed=False,
        intent="general",
        intent_reason="",
        domains=[],
        requires_planning=False,
        plan={},
        user_preferences={},
        store_knowledge={},
        multi_store_knowledge={},
        retrieved_memories=[],
        memory_search_calls=0,
        # 7. Output — empty until respond node runs
        final_answer="",
        response_metadata={},
        error=None,
        critic_status="",
        critic_reason="",
        critic_missing_information=[],
        reflection="",
        reflection_count=0,
        # 8. Memory persistence flag
        should_persist_memory=False,
        # 9. Guardrail (grader node) — clean defaults
        grader_denied=False,
        grader_reason="",
        # 10. Clarification / interrupt
        needs_clarification=False,
        clarification_prompt="",
        clarification_history=clarification_history or [],
        resume_from_clarification=resume_from_clarification,
    )


def make_tool_result(
    *,
    call_id: str,
    tool_name: str,
    data: Any,
    loop_index: int,
    error: str | None = None,
) -> ToolResult:
    """
    Convenience factory for creating a ToolResult dict.

    Args:
        call_id:    Must match the ToolCall.call_id that triggered this.
        tool_name:  Name of the tool that ran.
        data:       Payload returned by the tool.
        loop_index: Which loop iteration (0-based) produced this result.
        error:      Error message string if the tool failed, else None.

    Returns:
        A fully populated ToolResult TypedDict.
    """
    return ToolResult(
        call_id=call_id,
        tool_name=tool_name,
        success=error is None,
        data=data,
        error=error,
        loop_index=loop_index,
    )
