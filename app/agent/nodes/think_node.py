"""
app/agent/nodes/think_node.py
==============================
Think node — the LLM decides which tools to call or declares the goal complete.
"""

from __future__ import annotations

import time
from typing import Dict, Any

import structlog

from app.agent.state import VyaparAgentState, ToolCall
from app.lib.llm import get_llm, get_small_llm
from app.lib.cloudflare_ai import (
    is_cloudflare_configured,
    getChatModel,
    estimate_message_tokens,
    compress_context_if_needed,
    MAX_CONTEXT_TOKENS,
)
from app.lib.summarizer import summarize
from app.agent.tools.registry import VYAPAR_TOOLS, get_tools_for_user
from app.agent.tools.clusters import get_pruned_tools_for_user, detect_domains
from app.agent.prompts.system_prompts import (
    build_agent_base,
    MEMORY_WARNING,
    build_tool_synthesis,
    FIRST_LOOP,
    build_available_tools,
)
from app.agent.prompts.summarizer_prompts import SUMMARIZER_HISTORY_INSTRUCTION
from app.agent.utils import latest_human_prompt
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.think")

_STALE_GUARD_REFUSAL_MARKERS = (
    "that's outside what vyapar sakha handles",
    "i'm sorry, but i can't help with that request",
)


def _is_stale_guard_refusal(message) -> bool:
    """Exclude refusals from prior turns when building the current LLM prompt."""
    if getattr(message, "type", None) != "ai":
        return False
    content = getattr(message, "content", "")
    if isinstance(content, list):
        content = " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return any(marker in str(content).lower() for marker in _STALE_GUARD_REFUSAL_MARKERS)


async def think_node(state: VyaparAgentState) -> Dict[str, Any]:
    loop = state.get("loop_count", 0)
    store_id = state.get("store_id", "unknown")
    user_id = state.get("user_id", "unknown")

    LOGGER.info(
        "think_node_start",
        loop=loop,
        store_id=store_id,
        user_id=user_id,
        goal=state.get("goal"),
        goal_status=state.get("goal_status"),
        pending_tool_calls_count=len(state.get("pending_tool_calls", [])),
        tool_results_accumulated=len(state.get("tool_results", [])),
    )

    # Check memory status
    user_memory_loaded = state.get("user_memory_loaded", False)
    store_memory_loaded = state.get("store_memory_loaded", False)

    # The intent node decides whether Mem0 is needed. Do not force a memory
    # lookup for live-data or recent-conversation requests.
    if state.get("memory_query_needed", False):
        max_loops = state.get("max_loops", 3)
        is_final_loop = loop >= max_loops
        if not is_final_loop:
            LOGGER.info(
                "think_node_skip_memory_not_loaded",
                loop=loop,
                store_id=store_id,
                reason="routing to memory_query node before LLM invocation",
            )
            return {"memory_query_needed": True}

    user_prompt = state.get("user_prompt", "")
    raw_msgs = state.get("messages", [])
    if isinstance(user_prompt, str) and not user_prompt.strip() and not raw_msgs:
        raise ValueError("User message cannot be empty.")

    if is_cloudflare_configured():
        llm = getChatModel()
    else:
        llm = get_llm()

    if not llm:
        LOGGER.error("think_node_no_llm", store_id=store_id)
        return {
            "error": "LLM not configured — set at least one of CLOUDFLARE_API_TOKEN, GEMINI_API_KEY, NVIDIA_API_KEY, or GROQ_API_KEY in .env",
            "goal_status": "failed",
        }

    t0 = time.perf_counter()
    
    # Force the LLM to output a direct response (no tools) if we hit the loop ceiling
    is_final_loop = loop >= state.get("max_loops", 3)
    
    # Filter tools dynamically based on active domain(s) and RBAC permissions
    is_owner = state.get("is_owner", True)
    permissions = state.get("permissions", [])
    user_role = state.get("user_role", "Owner" if is_owner else "Employee")
    domains = state.get("domains")
    intent = state.get("intent")

    if intent in ("greeting", "off_topic", "harmful"):
        allowed_tools = []
    else:
        allowed_tools = get_pruned_tools_for_user(
            domains=domains,
            prompt=user_prompt,
            is_owner=is_owner,
            permissions=permissions,
        )

    if is_final_loop or not allowed_tools:
        LOGGER.info(
            "think_node_direct_text_mode",
            loop=loop,
            store_id=store_id,
            is_final=is_final_loop,
            tool_count=len(allowed_tools),
        )
        llm_with_tools = llm  # No tools bound, forces a text response
    else:
        llm_with_tools = llm.bind_tools(allowed_tools)

    # Build system prompt — richer on later loops to guide synthesis
    tools_called = state.get("tools_called_this_loop", [])
    available = [t.name for t in allowed_tools if t.name not in tools_called]
    results_so_far = state.get("tool_results", [])
    max_loops = state.get("max_loops", 3)

    # Extract/normalize goal on first loop
    current_goal = state.get("goal", "")
    goal_status = state.get("goal_status", "pending")
    if not current_goal and goal_status == "pending":
        current_goal = f"Answer the user's question: {state.get('user_prompt', '')}"
        goal_status = "in_progress"

    sys_content = build_agent_base(
        store_id=store_id,
        user_prompt=state.get("user_prompt", ""),
        current_goal=current_goal,
        loop=loop,
        max_loops=max_loops,
        user_context=state.get("user_context", {}),
        store_context=state.get("store_context", {}),
        user_role=user_role,
        is_owner=is_owner,
        permissions=permissions,
    )

    # Inject memory context if available
    user_prefs = state.get("user_preferences", {})
    store_knowledge = state.get("store_knowledge", {})
    multi_store_knowledge = state.get("multi_store_knowledge", {})

    user_mem_summary = user_prefs.get("summary", "") if isinstance(user_prefs, dict) else ""
    store_mem_summary = store_knowledge.get("summary", "") if isinstance(store_knowledge, dict) else ""
    multi_store_mem_summary = multi_store_knowledge.get("summary", "") if isinstance(multi_store_knowledge, dict) else ""

    if user_mem_summary or store_mem_summary or multi_store_mem_summary:
        sys_content += "Bootstrap memory context (stable, pre-loaded facts):\n"
        if user_mem_summary:
            sys_content += f"<user_preferences>\n{user_mem_summary}\n</user_preferences>\n\n"
        if store_mem_summary:
            sys_content += f"<store_knowledge>\n{store_mem_summary}\n</store_knowledge>\n\n"
        if multi_store_mem_summary:
            sys_content += f"<multi_store_knowledge>\n{multi_store_mem_summary}\n</multi_store_knowledge>\n\n"
        # CRITICAL: memory is preferences / store knowledge ONLY.
        # It is NOT a substitute for live tool data. Any question about
        # inventory, sales, forecasts, restock, or insights MUST call the
        # relevant tools — memory may be stale and will not have current numbers.
        sys_content += MEMORY_WARNING

    # Inject on-demand retrieved memories from search_memory tool calls
    from app.agent.tools.memory.search import MAX_MEMORY_SEARCH_CALLS
    retrieved_memories = state.get("retrieved_memories", [])
    memory_search_calls = state.get("memory_search_calls", 0)
    if retrieved_memories:
        # Deduplicate by memory_id before injecting
        seen_mids: set[str] = set()
        unique_retrieved = []
        for rm in retrieved_memories:
            mid = rm.get("memory_id", "")
            if mid and mid not in seen_mids:
                seen_mids.add(mid)
                unique_retrieved.append(rm)
            elif not mid:
                unique_retrieved.append(rm)
        if unique_retrieved:
            sys_content += "On-demand retrieved memories (from search_memory tool):\n<retrieved_memories>\n"
            for rm in unique_retrieved:
                mem_type = rm.get('memory_type', 'unknown')
                event_time = rm.get('event_time', '')
                time_str = f" | {event_time}" if event_time else ""
                content = rm.get('content', '')
                sys_content += f"[{mem_type}{time_str}] {content}\n"
            sys_content += "</retrieved_memories>\n\n"

    # Search budget hint
    remaining_searches = MAX_MEMORY_SEARCH_CALLS - memory_search_calls
    if not is_final_loop and remaining_searches > 0:
        sys_content += (
            f"You have {remaining_searches} search_memory call(s) remaining. "
            "Use search_memory only when you need historical decisions, past events, or "
            "long-term patterns not available in the bootstrap memory above or from live tools.\n\n"
        )
    elif not is_final_loop and remaining_searches <= 0:
        sys_content += (
            "search_memory budget exhausted — do not call search_memory again. "
            "Reason from available context and respond directly.\n\n"
        )

    plan = state.get("plan")
    if plan and isinstance(plan, dict):
        sys_content += "\nExecution Plan:\n"
        if "goal" in plan:
            sys_content += f"Goal: {plan['goal']}\n"
        if "steps" in plan:
            for idx, step in enumerate(plan["steps"]):
                sys_content += f"{idx + 1}. {step}\n"
        sys_content += "Follow this plan using the available tools.\n\n"

    if results_so_far:
        sys_content += build_tool_synthesis(
            results_so_far=results_so_far,
            inventory_context=state.get("inventory_context", {}),
            sales_context=state.get("sales_context", {}),
            forecast_context=state.get("forecast_context", {}),
            insights_context=state.get("insights_context", []),
        )
    else:
        sys_content += FIRST_LOOP

    if available and not is_final_loop:
        sys_content += build_available_tools(available)

        # Instruct the LLM about the ask_for_clarification tool
        # — when to use it and why it exists.
        if "ask_for_clarification" in available:
            sys_content += (
                "IMPORTANT — You have a special tool available called "
                "`ask_for_clarification`. Use this tool when you are "
                "uncertain about what the user wants, when the question "
                "is too vague to answer with available tools, when you "
                "need specific information the user hasn't provided, or "
                "when you cannot answer confidently. Call it with a "
                "`reason` explaining why you need clarification. Do NOT "
                "try to guess or give a vague answer — ask the user "
                "instead.\n\n"
            )

    sys_content += (
        "SELF-CORRECTION RULES:\n"
        "- If a tool returns an error (e.g. Validation Error), DO NOT immediately retry with the exact same parameters.\n"
        "- Analyze the error message. If it says a field is missing, find that information in your context, then retry.\n"
        "- If the error persists, use a different tool to achieve your goal or ask the user for clarification.\n\n"
    )

    # Inject clarification history if present — tells the LLM
    # that a previous clarification was answered so it can
    # incorporate that context into its reasoning.
    clarification_history = state.get("clarification_history", [])
    if clarification_history:
        sys_content += "\n\n--- Previous Clarifications ---\n"
        for entry in clarification_history[-3:]:
            question = entry.get("question", "")
            answer = entry.get("answer", "")
            sys_content += (
                f"User was asked: \"{question}\"\n"
                f"User answered: \"{answer}\"\n\n"
            )
        sys_content += "---\n\n"

    raw_messages = [
        message
        for message in state.get("messages", [])
        if not _is_stale_guard_refusal(message)
    ]
    # Sliding window: keep the last 3 messages verbatim, compress everything
    # older than that into a compact summary that gets appended to the
    # system prompt. This prevents the LLM context window from growing
    # without bound across loops while preserving the recent conversation
    # flow the LLM needs for continuity.
    #
    # NOTE: the summary is merged into the system prompt rather than
    # emitted as a separate SystemMessage — Gemini only allows a single
    # system instruction at position 0.
    # Sliding window: keep the last 6 messages verbatim (3 full user+assistant turns)
    # for better continuity. Everything older is compressed into a compact summary.
    t_window_start = time.perf_counter()
    window_summary, recent_messages = await _sliding_window(raw_messages, keep_last=6)
    t_window_ms = round((time.perf_counter() - t_window_start) * 1000, 1)

    if t_window_ms > 200:
        LOGGER.warning(
            "think_node_sliding_window_slow",
            loop=loop,
            window_ms=t_window_ms,
            raw_message_count=len(raw_messages),
        )
    else:
        LOGGER.debug("think_node_sliding_window", loop=loop, window_ms=t_window_ms)

    if window_summary:
        sys_content = sys_content + "\n\n" + window_summary

    sys_msg = SystemMessage(content=sys_content)
    messages = [sys_msg] + recent_messages
    if not any(getattr(m, "type", None) == "human" for m in recent_messages):
        user_prompt = state.get("user_prompt", "") or current_goal or "Please assist me."
        messages.append(HumanMessage(content=user_prompt))

    # Calculate / estimate token size for 10k–20k token context management
    estimated_tokens = estimate_message_tokens(messages)
    LOGGER.info(
        "think_node_token_estimate",
        loop=loop,
        estimated_tokens=estimated_tokens,
        message_count=len(messages),
    )
    if estimated_tokens > MAX_CONTEXT_TOKENS:
        messages = await compress_context_if_needed(messages, max_tokens=MAX_CONTEXT_TOKENS)

    LOGGER.debug(
        "think_node_invoking_llm",
        loop=loop,
        message_count=len(messages),
        available_tools=available,
        is_final_loop=is_final_loop
    )

    try:
        response = await llm_with_tools.ainvoke(
            messages,
            config={"configurable": {"user_id": user_id, "store_id": store_id}},
        )
    except Exception as exc:
        LOGGER.warning("think_node_llm_with_tools_failed_trying_fallback", loop=loop, error=str(exc)[:200])
        try:
            # Fallback 1: Direct invocation without tool binding (in case tool schema hit provider token limits)
            LOGGER.info("think_node_fallback_direct_llm", loop=loop)
            response = await llm.ainvoke(
                messages,
                config={"configurable": {"user_id": user_id, "store_id": store_id}},
            )
        except Exception as exc2:
            LOGGER.warning("think_node_direct_llm_failed_trying_small_llm", loop=loop, error=str(exc2)[:200])
            try:
                # Fallback 2: Try fast small LLM chain (Groq / Gemini Lite / NVIDIA)
                small_llm = get_small_llm()
                if small_llm:
                    LOGGER.info("think_node_fallback_small_llm", loop=loop)
                    response = await small_llm.ainvoke(
                        messages,
                        config={"configurable": {"user_id": user_id, "store_id": store_id}},
                    )
                else:
                    raise exc2
            except Exception as exc3:
                LOGGER.error("think_node_all_fallbacks_failed", loop=loop, error=str(exc3), exc_info=True)
                fallback_msg = (
                    "Namaste! I am currently experiencing high load with AI service providers. "
                    "Please try your question again in a few moments."
                )
                return {
                    "error": str(exc3),
                    "goal_status": "complete",
                    "final_answer": fallback_msg,
                    "messages": [AIMessage(content=fallback_msg)],
                }

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    if hasattr(response, "tool_calls") and response.tool_calls:
        # Check circuit breaker before committing to tool calls
        error_counts = state.get("tool_error_counts", {})
        broken_tools = [tc["name"] for tc in response.tool_calls if error_counts.get(tc["name"], 0) >= 3]
        if broken_tools:
            LOGGER.warning("think_node_circuit_breaker_triggered", tools=broken_tools)
            return {
                "error": f"Circuit breaker triggered: Tools {broken_tools} failed repeatedly.",
                "goal_status": "failed",
                "final_answer": "I'm having trouble executing some operations due to repeated system errors. Please verify the information or contact support.",
            }

        pending_calls = [
            ToolCall(
                tool_name=tc["name"],
                arguments=tc["args"],
                call_id=tc["id"],
            )
            for tc in response.tool_calls
        ]
        tool_names = [c["tool_name"] for c in pending_calls]

        # If the LLM wants to ask for clarification, route directly
        # to the interrupt node instead of executing a tool call.
        if any(c["tool_name"] == "ask_for_clarification" for c in pending_calls):
            reason = ""
            for c in pending_calls:
                if c["tool_name"] == "ask_for_clarification":
                    reason = c["arguments"].get("reason", "")
                    break
            LOGGER.info(
                "think_node_clarification_tool_called",
                loop=loop,
                tools_requested=tool_names,
                reason=reason[:200] if reason else None,
            )
            return {
                "needs_clarification": True,
                "clarification_prompt": reason,
                "goal_status": "in_progress",
                "goal": current_goal,
                "messages": [response],
            }

        LOGGER.info(
            "think_node_decided_tools",
            loop=loop,
            tools_requested=tool_names,
            llm_latency_ms=elapsed_ms,
        )

        return {
            "pending_tool_calls": pending_calls,
            "tools_called_this_loop": tool_names,
            "messages": [response],
            "goal": current_goal,
            "goal_status": goal_status,
        }
    else:
        LOGGER.info(
            "think_node_decided_complete",
            loop=loop,
            llm_latency_ms=elapsed_ms,
        )
        final_text = _flatten_text(response.content) if hasattr(response, "content") else _flatten_text(response)
        tools_used = list({
            r.get("tool_name") or r.get("tool")
            for r in state.get("tool_results", [])
            if isinstance(r, dict) and (r.get("tool_name") or r.get("tool"))
        })

        # Determine if this conversation is worth persisting to mem0.
        # Skip trivial exchanges ("hi", "hello", "ok", etc.) to avoid
        # polluting long-term memory with noise.
        should_persist = _is_conversation_meaningful(
            user_prompt=state.get("user_prompt", ""),
            final_answer=final_text,
            tools_used=tools_used,
            loop_count=loop,
        )

        # Check if clarification is needed before completing.
        # This happens when:
        #   - Final loop reached with no tools called (LLM uncertain)
        #   - Goal status is "failed" (max loops exceeded)
        #   - Agent tried tools but got no useful results
        clarification_decision = _should_ask_clarification(
            loop=loop,
            max_loops=state.get("max_loops", 3),
            tools_used=tools_used,
            tool_results=state.get("tool_results", []),
            goal_status=goal_status,
        )

        if clarification_decision["needs_clarification"]:
            LOGGER.info(
                "think_node_clarification_needed",
                loop=loop,
                reason=clarification_decision["reason"],
            )
            # Use the decision's question if available; fall back to the
            # reason so the interrupt node has a concrete prompt to use
            # as the interrupt payload (avoids an extra LLM call that would
            # also re-run on resume).
            prompt = clarification_decision.get("question") or clarification_decision.get("reason", "")
            return {
                "needs_clarification": True,
                "clarification_prompt": prompt,
                "goal_status": "in_progress",
                "goal": current_goal,
                "messages": [response],
                "final_answer": final_text,
                "should_persist_memory": should_persist,
            }

        # Route to critic for complex tasks, or tasks that used multiple tools/loops
        is_complex = state.get("requires_planning", False) or len(tools_used) > 1 or loop > 1
        
        # If the user explicitly provided a complex query or the agent had to use tools, review it
        # (We skip review for 0-tool conversations or 1-tool simple lookups to save latency)
        next_status = "review" if is_complex and not is_final_loop else "complete"

        return {
            "goal_status": next_status,
            "goal": current_goal,
            "messages": [response],
            "final_answer": final_text,
            "should_persist_memory": should_persist,
            "response_metadata": {
                "loops_taken": loop,
                "tools_used": tools_used,
                "goal_status": next_status,
                "goal": current_goal,
                "forced_stop": is_final_loop,
                "memory_loaded": {
                    "user": state.get("user_memory_loaded", False),
                    "store": state.get("store_memory_loaded", False),
                },
            }
        }


# ---------------------------------------------------------------------------
# Conversation importance classifier
# ---------------------------------------------------------------------------

# Trivial greetings / acknowledgements that should NOT be persisted
_TRIVIAL_PATTERNS = [
    "hi", "hello", "hey", "hi there", "hello there",
    "ok", "okay", "yes", "no", "yeah", "nope",
    "thanks", "thank you", "thank", "thx",
    "good morning", "good afternoon", "good evening",
    "how are you", "are you there", "you there",
    "help", "help me",
]

_MIN_MEANINGFUL_LENGTH = 15
# Minimum characters for a response to be considered substantive


def _is_conversation_meaningful(
    *,
    user_prompt: str,
    final_answer: str,
    tools_used: list[str],
    loop_count: int,
) -> bool:
    """
    Determine whether a conversation contains enough substance to persist
    to long-term mem0 memory.

    Returns False for trivial exchanges ("hi", "hello", "ok") to avoid
    polluting memory with noise.

    Returns True when:
        - The agent used tools (real data was gathered)
        - The user prompt is substantive (not a greeting)
        - The final answer is substantive (long enough, not just "sure")
    """
    # If tools were used, the conversation has real data — always persist
    if tools_used:
        return True

    # Multiple loops means the agent did real reasoning work
    if loop_count > 0:
        return True

    # Check if user prompt is trivial
    prompt_lower = _flatten_text(user_prompt).strip().lower()
    if prompt_lower in _TRIVIAL_PATTERNS:
        return False

    # Short prompts (< 15 chars) that aren't greetings are likely trivial
    if len(prompt_lower) < _MIN_MEANINGFUL_LENGTH:
        return False

    # Check if the final answer is substantive
    answer_lower = _flatten_text(final_answer).strip().lower()
    if len(answer_lower) < _MIN_MEANINGFUL_LENGTH:
        return False

    # If the answer is just a greeting back, skip
    if answer_lower in _TRIVIAL_PATTERNS:
        return False

    return True


def _should_ask_clarification(
    *,
    loop: int,
    max_loops: int,
    tools_used: list[str],
    tool_results: list[dict],
    goal_status: str,
) -> dict:
    """
    Determine whether the agent should ask the user for clarification.

    Returns a dict:
        {
            "needs_clarification": bool,
            "reason": str,
            "question": str | None,
        }

    Triggers when:
        - Final loop reached with no useful tools called (LLM stuck)
        - Goal status is "failed" (max loops exceeded)
        - All tool calls failed
    """
    failed_results = [
        r for r in tool_results if not r.get("success", False)
    ]
    all_failed = (
        bool(tool_results) and len(failed_results) == len(tool_results)
    )

    if goal_status == "failed":
        return {
            "needs_clarification": True,
            "reason": "max loops exceeded without completing the goal",
            "question": None,
        }

    if loop >= max_loops and not tools_used:
        return {
            "needs_clarification": True,
            "reason": "reached loop ceiling without calling any tools — "
            "query may be ambiguous or out of scope",
            "question": None,
        }

    if all_failed:
        return {
            "needs_clarification": True,
            "reason": "all tool calls failed — may need user guidance "
            "to resolve the issue",
            "question": None,
        }

    return {
        "needs_clarification": False,
        "reason": "",
        "question": None,
    }


# ---------------------------------------------------------------------------
# Sliding-window context compressor
# ---------------------------------------------------------------------------

def _message_text(msg) -> str:
    """Extract a compact text representation of a BaseMessage."""
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text", "")
                parts.append(_flatten_text(text))
            else:
                parts.append(_flatten_text(block))
        content = " ".join(parts)
    return _flatten_text(content)


def _flatten_text(value) -> str:
    """
    Recursively coerce a value into a plain string.

    Some LLM providers nest text as a list of strings or dicts (e.g.
    ``content=[{"type": "text", "text": ["a", "b"]}]``). Passing such a
    nested value directly to ``str.join`` raises::

        TypeError: sequence item 0: expected str instance, list found

    This helper walks nested lists/dicts and flattens them into a single
    string, so every ``.join`` call in this module always receives strings.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten_text(v) for v in value)
    if isinstance(value, dict):
        # Prefer a "text" key if present, else stringify the whole dict.
        if "text" in value:
            return _flatten_text(value["text"])
        return str(value)
    if value is None:
        return ""
    return str(value)


def _is_window_start_invalid(msg) -> bool:
    """
    Return True if ``msg`` cannot validly start the recent window.

    Gemini requires the conversation after the system instruction to begin
    with a user turn or an AI turn that does NOT contain tool_calls.
    Specifically:
      - A ToolMessage must follow an AI tool_call turn — it cannot start
        the conversation.
      - An AIMessage WITH tool_calls must follow a user turn — it cannot
        start the conversation either.

    So a window starting with either of those is invalid and the split
    point must be walked backward.
    """
    msg_type = getattr(msg, "type", "") or ""
    if msg_type == "tool":
        return True
    if msg_type == "ai" and getattr(msg, "tool_calls", None):
        return True
    return False


def _owns_tool_response(prev_msg, cur_msg) -> bool:
    """
    Return True if ``prev_msg`` is an AIMessage with tool_calls and
    ``cur_msg`` is the ToolMessage response for one of those calls.

    When True, splitting between them orphans the ToolMessage — Gemini
    requires "function response turn comes immediately after a function
    call turn", so the pair must stay together in the recent window.
    """
    if getattr(prev_msg, "type", "") != "ai":
        return False
    tool_calls = getattr(prev_msg, "tool_calls", None) or []
    if not tool_calls:
        return False
    if getattr(cur_msg, "type", "") != "tool":
        return False
    call_id = getattr(cur_msg, "tool_call_id", None)
    return any(tc.get("id") == call_id for tc in tool_calls)


async def _sliding_window(messages, *, keep_last: int = 3):
    """
    Compress the message list into a sliding window.

    Strategy:
      - Keep the last ``keep_last`` messages verbatim (preserves recent
        conversation flow the LLM needs for continuity).
      - Everything older than that is collapsed into a single compact
        summary string that the CALLER prepends to the system prompt.

    Why not a SystemMessage? Gemini (ChatGoogleGenerativeAI) only allows a
    single system instruction at position 0 — a second SystemMessage later
    in the list raises "Unexpected message with type SystemMessage at the
    position 1". So the summary is returned as plain text and merged into
    the existing system prompt by the caller.

    Gemini turn-ordering constraint: the recent window must start with a
    HumanMessage or an AIMessage WITHOUT tool_calls. An AIMessage with
    tool_calls must follow a user turn, and a ToolMessage must follow an
    AI tool_call turn — otherwise Gemini returns "Please ensure that
    function call turn comes immediately after a user turn or after a
    function response turn." To satisfy this, the window start is walked
    backward until the first message is a valid turn starter.

    Additionally, tool-call/response pairs are kept atomic: if the message
    just before the split is an AIMessage with tool_calls and the first
    recent message is its ToolMessage response, the split is moved back
    one step so the pair stays together (Gemini requires the function
    response to immediately follow the function call turn).

    This is applied only to the *prompt* sent to the LLM — the full
    message list remains in the graph state (and the checkpointer) for
    cross-session persistence.

    Args:
        messages: list[BaseMessage] from the graph state.
        keep_last: Number of recent messages to keep verbatim.

    Returns:
        tuple[str, list[BaseMessage]] — (summary_text, recent_messages).
        summary_text is "" when no compression was needed.
    """
    if not messages:
        return "", []

    if len(messages) <= keep_last:
        return "", list(messages)

    # Compute the split point, then walk it backward so the recent window
    # starts with a valid Gemini turn starter (human or AI without
    # tool_calls). This preserves the required user→tool_call→tool→...
    # alternation across the compression boundary.
    #
    # We also keep tool-call/response pairs atomic: if the message just
    # before the split is an AIMessage with tool_calls and the first
    # recent message is its ToolMessage response, walking back one
    # step keeps the pair together (Gemini requires the function
    # response to immediately follow the function call turn).
    split = len(messages) - keep_last
    while split > 0 and _is_window_start_invalid(messages[split]):
        split -= 1
    while (
        split > 0
        and _owns_tool_response(messages[split - 1], messages[split])
    ):
        split -= 1

    # If walking back consumed the whole list, there's nothing safe to
    # compress — return the full list verbatim.
    if split <= 0:
        return "", list(messages)

    old = messages[:split]
    recent = messages[split:]

    # Build a compact summary of the older messages
    tool_calls_made = []
    raw_summary_parts = []
    for m in old:
        text = _message_text(m)
        role = getattr(m, "type", "message")
        # Truncate long tool payloads — they live in *_context buckets anyway
        if len(text) > 400:
            text = text[:400] + "...[truncated]"
        raw_summary_parts.append(text)
        # Track tool names mentioned in tool messages for the summary
        name = getattr(m, "name", None) or getattr(m, "tool_call_id", None)
        if name and role == "tool" and name not in tool_calls_made:
            tool_calls_made.append(name)

    # Defensive: ensure every part is a plain string before joining.
    # Some message content can be a list/dict (e.g. Gemini multi-part
    # content), which would raise "sequence item 0: expected str instance,
    # list found" inside str.join.
    raw_summary_parts = [_flatten_text(p) for p in raw_summary_parts]
    raw_summary = "\n".join(raw_summary_parts)

    # If the older segment is large, compress it with the small summarizer
    # (GROQ/NVIDIA) so the sliding-window summary stays tiny.
    # Threshold is set high (3000 chars) because calling the summarizer LLM
    # costs ~5-10s and is only worth it for very long conversation histories.
    if len(raw_summary) > 5000:
        compressed = await summarize(
            raw_summary,
            instruction=SUMMARIZER_HISTORY_INSTRUCTION,
            max_tokens=256,
        )
        summary = compressed or raw_summary
    else:
        summary = raw_summary

    header = "Earlier conversation summary (older messages compressed):"
    summary = f"{header}\n{summary}"
    if tool_calls_made:
        summary += f"\nTools already used earlier: {', '.join(tool_calls_made)}"

    return summary, list(recent)
