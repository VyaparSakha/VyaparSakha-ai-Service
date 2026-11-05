"""
app/agent/graph.py
==================

LangGraph graph assembly for the Vyapar Copilot agent.

Graph flow
----------
              ┌──────────────────┐
              │   grader node     │  ← entry point (guardrail)
              │  (safety check)   │      harmful prompt → END (refusal)
              └──────┬───────────┘      safe prompt      → intent
                     │
                     ▼
              ┌──────────────┐
              │  intent node  │
              │  (routing)    │
              └──────┬───────────┘
                     │
                     ▼
              ┌──────────────┐
              │    think node │
              │  (LLM reasoning)  │
              └──────┬───────────┘
                     │
        ┌────────────┼────────────────┐
        │            │                │
    interrupt   memory_query    pending_tool  goal_status
    needed?       calls?         == "complete"
         │              │                │
         ▼              ▼                ▼
  ┌──────────┐  ┌──────────┐  ┌──────────────┐
  │ interrupt│  │ memory   │  │     END      │
  │  node    │  │  query   │  │ (memory write│
  │          │  │          │  │  in background)│
  └─────┬────┘  └────┬─────┘  └──────────────┘
        │             │
        ▼             ▼
      END          think (loop)

Interrupt path:
    think → interrupt → think (loop)
    The interrupt node calls langgraph.types.interrupt() to pause the
    graph. The clarification question is surfaced to the caller via
    __interrupt__ in the stream. When the user responds, the caller
    resumes with Command(resume=<answer>) using the same thread_id.
    The interrupt() call returns the answer, it is stored in
    clarification_history, and the graph loops back to think.

Max loops: 3 (hard ceiling to respect LLM rate limits).
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.agent.state import VyaparAgentState
from app.agent.nodes.think_node import think_node
from app.agent.nodes.tool_node import tool_node
from app.agent.nodes.observe_node import observe_node
from app.agent.nodes.memory_query import memory_query_node
from app.agent.nodes.intent_node import intent_node
from app.agent.nodes.interrupt_node import interrupt_node
from app.agent.nodes.context_node import context_node
from app.agent.nodes.subgraph_router import subgraph_router
from app.agent.nodes.planner_node import planner_node
from app.agent.nodes.critic_node import critic_node
from app.agent.nodes.reflection_node import reflection_node


# ---------------------------------------------------------------------------
# Router functions
# ---------------------------------------------------------------------------

def _route_after_memory_query(state: VyaparAgentState) -> str:
    """
    Conditional edge function called after memory_query node.

    Routes to the planner if the intent node identified the request as
    complex and needing planning (requires_planning is True).
    Otherwise goes straight to the think node.
    """
    if state.get("requires_planning", False) and not state.get("plan"):
        return "planner"
    return "think"


def _route_after_intent(state: VyaparAgentState) -> str:
    """
    Conditional edge function called after intent_node.
    - goal_status == "complete" → END (intent_node already set final_answer, e.g. greeting/harmful)
    - pending_tool_calls set → tool (execute retrieved semantic tools directly for 2-LLM-call fast path)
    - memory_query_needed → memory_query
    - requires_planning → planner
    - default → think
    """
    if state.get("goal_status") == "complete":
        return "__end__"
    if state.get("pending_tool_calls"):
        return "tool"
    if state.get("memory_query_needed", False) and state.get("intent") in {"memory", "mixed"}:
        return "memory_query"
    if state.get("requires_planning", False) and not state.get("plan"):
        return "planner"
    return "think"

def _route_after_grader(state: VyaparAgentState) -> str:
    """
    Conditional edge function called after grader_node (the entry point).

    The grader sets ``grader_denied`` when the user prompt was classified as
    harmful. In that case the graph terminates immediately — the refusal is
    already stored in ``final_answer``. Otherwise the graph proceeds to the
    normal think node.
    """
    if state.get("grader_denied", False):
        return END
    return "intent"


def _route_after_critic(state: VyaparAgentState) -> str:
    """
    Conditional edge function called after critic_node.
    """
    if state.get("goal_status") == "reflect":
        return "reflection"
    return END


def _route_after_reflection(state: VyaparAgentState) -> str:
    """
    Conditional edge function called after reflection_node.
    If reflection capped out (goal_status == 'complete'), terminate graph.
    Otherwise, loop back to think for another try.
    """
    if state.get("goal_status") == "complete":
        return END
    return "think"


def _route_after_think(state: VyaparAgentState) -> str:
    """
    Conditional edge function called after think_node.

    Priority (highest to lowest):
        1. If needs_clarification → go to interrupt node
        2. If memory_query_needed → go to memory_query node
        3. If goal complete → END
        4. If loop_count >= max_loops → END (prevent infinite recursion)
        5. If pending_tool_calls → go to tool node
        6. Otherwise → END
    """
    # Clarification takes highest priority — the agent is uncertain
    if state.get("needs_clarification", False):
        return "interrupt"

    # Memory query requested by the think node
    if state.get("memory_query_needed", False):
        return "memory_query"

    # Goal completed — exit graph. Memory persistence is handled
    # by the route handler as a background task after the SSE stream.
    if state.get("goal_status") == "complete":
        return END

    if state.get("goal_status") == "review":
        return "critic"

    # Max loop guard — enforce hard ceiling to prevent recursion loops
    loop_count = state.get("loop_count", 0)
    max_loops = state.get("max_loops", 5)
    if loop_count >= max_loops:
        return END

    # Agent requested tool calls
    if state.get("pending_tool_calls"):
        return "tool"

    # No tools and goal not explicitly complete
    return END


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """
    Assemble and compile the Vyapar Copilot LangGraph.

    Args:
        checkpointer:
            MongoDB-backed LangGraph checkpointer.
            If None, graph runs without persistence.

    Returns:
        Compiled LangGraph.
    """
    workflow = StateGraph(VyaparAgentState)

    # --------------------------------------------------------------
    # Register nodes
    # --------------------------------------------------------------

    # workflow.add_node("context", context_node)
    # workflow.add_node("grader", grader_node)
    workflow.add_node("intent", intent_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("interrupt", interrupt_node)
    workflow.add_node("think", think_node)
    workflow.add_node("memory_query", memory_query_node)
    workflow.add_node("tool", tool_node)
    workflow.add_node("subgraph_router", subgraph_router)
    workflow.add_node("observe", observe_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("reflection", reflection_node)

    # --------------------------------------------------------------
    # Entry point — intent router runs first
    # (context_node temporarily commented out)
    # --------------------------------------------------------------

    workflow.set_entry_point("intent")

    # workflow.add_edge("context", "intent")

    # --------------------------------------------------------------
    # Conditional edges from grader
    # Denied → END (refusal already in final_answer)
    # Safe  → think
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # Conditional edges from intent
    # intent -> greeting short-circuit (__end__), memory_query, planner, or think
    # --------------------------------------------------------------
    workflow.add_conditional_edges(
        "intent",
        _route_after_intent,
        {
            "__end__": END,
            "tool": "tool",
            "think": "think",
            "memory_query": "memory_query",
            "planner": "planner",
        },
    )

    # memory_query -> planner or think
    workflow.add_conditional_edges(
        "memory_query",
        _route_after_memory_query,
        {
            "planner": "planner",
            "think": "think",
        },
    )

    # planner -> think
    workflow.add_edge("planner", "think")

    # --------------------------------------------------------------
    # Conditional edges from think
    # interrupt has highest priority (checked in router)
    # --------------------------------------------------------------

    workflow.add_conditional_edges(
        "think",
        _route_after_think,
        {
            "interrupt": "interrupt",
            "memory_query": "memory_query",
            "tool": "tool",
            "critic": "critic",
            "__end__": END,
        },
    )

    # --------------------------------------------------------------
    # Conditional edges from critic
    # --------------------------------------------------------------

    workflow.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "reflection": "reflection",
            "__end__": END,
        },
    )

    # --------------------------------------------------------------
    # Conditional edges from reflection
    # --------------------------------------------------------------

    workflow.add_conditional_edges(
        "reflection",
        _route_after_reflection,
        {
            "think": "think",
            "__end__": END,
        },
    )

    # --------------------------------------------------------------
    # interrupt → think (graph pauses at interrupt() inside the node;
    # when resumed via Command(resume=...), the interrupt node stores
    # the user's answer in clarification_history and the graph loops
    # back to think so the LLM can incorporate the answer and continue)
    # --------------------------------------------------------------

    workflow.add_edge("interrupt", "think")

    # --------------------------------------------------------------
    # tool → observe → subgraph_router → think (agent loop)
    # --------------------------------------------------------------

    workflow.add_edge("tool", "observe")
    workflow.add_edge("observe", "subgraph_router")
    workflow.add_edge("subgraph_router", "think")

    # --------------------------------------------------------------
    # Compile graph
    # --------------------------------------------------------------

    return workflow.compile(
        checkpointer=checkpointer
    )
