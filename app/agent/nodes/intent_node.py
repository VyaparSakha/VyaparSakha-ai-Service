"""Intent routing node for choosing live tools, chat history, or Mem0."""

from __future__ import annotations

import asyncio
import random
import uuid
from typing import Any, Dict

import structlog
from pydantic import BaseModel, Field

from app.agent.prompts.intent_prompt import INTENT_CLASSIFIER_INSTRUCTION
from app.agent.state import VyaparAgentState
from app.agent.utils import latest_human_prompt
from app.lib.grader import _SEVERE_HARM_KEYWORDS
from app.lib.llm import get_intent_llm, get_small_llm
from app.agent.tools.clusters import detect_domains, _CONVERSATIONAL_PATTERNS
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.intent")

_VALID_INTENTS = {"live_data", "memory", "conversation_recap", "mixed", "general", "greeting", "off_topic", "harmful"}
_MEMORY_REQUIRED_INTENTS = {"memory", "mixed"}

_REFUSAL_HARMFUL = (
    "I'm sorry, but I can't help with that request. "
    "Vyapar Sakha is designed to assist with retail store operations "
    "such as inventory, sales, forecasting, and restocking. "
    "Please let me know if there's something else I can help you with."
)

_REFUSAL_OFF_TOPIC = (
    "That's outside what Vyapar Sakha handles. I can help with your "
    "store's inventory, sales, forecasting, restocking, and business "
    "insights — ask me about any of those."
)

_LIVE_TERMS = (
    "inventory", "stock", "stockout", "out of stock", "sales", "selling", "sell", "sold",
    "revenue", "profit", "loss", "margin", "margins", "earnings", "income", "p&l",
    "forecast", "forecasting", "restock", "restocking", "reorder", "demand",
    "anomaly", "store insight", "store summary", "store overview", "store health",
    "lifetime", "turnover", "orders", "order", "bill", "billing", "invoice",
    "buyer dues", "seller dues", "supplier dues", "buyers owe", "owe sellers",
    "owe suppliers", "receivables", "payables", "outstanding payments",
    "pending payments", "amount i owe", "dues", "debt",
    "calculate", "calculation", "audit", "performance",
    "top selling", "best selling", "fast moving", "slow moving", "dead stock",
    "category", "supplier", "suppliers", "buyer", "buyers", "customer", "customers",
    "today", "yesterday", "this month", "last month", "this year", "last year", "end of this year",
    "growth", "target", "kpi",
)
_MEMORY_TERMS = (
    "remember", "last time", "previous", "earlier", "we decided",
    "our decision", "usually", "prefer", "preference", "my language",
    "my style", "tell me about myself", "about myself", "what do you know about me",
)
_RECAP_TERMS = (
    "what did we discuss", "what were we discussing", "recent chat",
    "recent conversation", "summarize our chat", "summarise our chat",
)


_COMPLEX_TERMS = (
    "recommend", "strategy", "plan", "which products should",
    "what should i", "how should i", "next week", "budget",
    "best products", "analyze", "prioritize"
)

async def _generate_conversational_reply(state: VyaparAgentState, prompt: str) -> str:
    """Generate a warm, natural, personalized greeting using the fast LLM tier (Groq 20B / Gemini Lite)."""
    user_ctx = state.get("user_context") or {}
    store_ctx = state.get("store_context") or {}
    merchant_name = user_ctx.get("name") or "Merchant"
    first_name = merchant_name.split()[0] if merchant_name else "Merchant"
    store_name = store_ctx.get("name") or "Store"
    business_type = store_ctx.get("business_type") or "retail"

    sys_prompt = (
        f"You are Vyapar Sakha (व्यापार सखा), the trusted AI retail co-pilot and fractional CFO for Indian store owners.\n"
        f"Active Merchant: {merchant_name}\n"
        f"Store: {store_name} ({business_type.title()} Merchant)\n\n"
        f"Greet the merchant respectfully and warmly in natural, friendly Hinglish/Hindi (or English matching the query).\n"
        f"Address them respectfully as 'Namaste {first_name} ji!'.\n"
        f"Directly answer their query (e.g. if they asked 'kaise ho', answer how you are doing, ask how their vyapar/store is going, "
        f"and offer help with sales, inventory, or billing).\n"
        f"Keep it concise, natural, and friendly (1-3 sentences max). Do NOT invent database numbers. Do NOT sound robotic."
    )

    llm = get_intent_llm() or get_small_llm()
    if not llm:
        return f"Namaste {first_name} ji! Main theek hoon. Aapke store ke liye aaj kya madad kar sakta hoon?"

    try:
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=prompt),
        ]
        res = await asyncio.wait_for(
            llm.ainvoke(messages, config={"tags": ["hide_stream"]}),
            timeout=5.0,
        )
        content = res.content if hasattr(res, "content") else str(res)
        text = str(content).strip()
        if text:
            return text
        return f"Namaste {first_name} ji! Main theek hoon. Aapka vyapar kaisa chal raha hai? Sales ya inventory me koi madad chahiye toh batayein."
    except Exception as exc:
        LOGGER.warning("intent_greeting_llm_failed", error=str(exc)[:150])
        return f"Namaste {first_name} ji! Main theek hoon. Aapka vyapar kaisa chal raha hai? Sales ya inventory me koi madad chahiye toh batayein."


class ToolDispatchCall(BaseModel):
    tool_name: str = Field(description="Exact tool name from the available candidate tools list")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Dictionary of tool arguments (e.g. {'days_lookback': 30, 'store_id': '...'})")


class IntentClassification(BaseModel):
    intent: str = Field(description="The primary intent label: harmful, off_topic, live_data, memory, conversation_recap, mixed, greeting, or general")
    domains: list[str] = Field(default_factory=list, description="Active business domain(s): sales, inventory, forecast, suppliers_purchases, buyers_credit, finance_expenses, billing, documents, communication, automations, insights_overview, web, or [] for conversational")
    requires_planning: bool = Field(default=False, description="True ONLY if the request is complex, multi-step, or strategic requiring a formal step-by-step plan. False for direct data queries like 'tell me about last 30 day sells' or 'low stock alert'")
    reason: str = Field(description="Short operational rationale for the classification and domain selection")
    tool_calls: list[ToolDispatchCall] = Field(default_factory=list, description="If intent is live_data, list 1 to 3 candidate tools to call with accurate arguments. Empty list for conversational, greeting, memory, or general queries.")


def _requires_planning(prompt: str) -> bool:
    lowered = (prompt or "").lower()
    return any(term in lowered for term in _COMPLEX_TERMS)


def _deterministic_intent(prompt: str) -> str | None:
    lowered = (prompt or "").lower().strip()
    if any(kw in lowered for kw in _SEVERE_HARM_KEYWORDS):
        return "harmful"

    has_dues_request = (
        ("buyer" in lowered or "buyers" in lowered)
        and ("seller" in lowered or "sellers" in lowered)
        and ("owe" in lowered or "own" in lowered)
    )
    has_live = any(term in lowered for term in _LIVE_TERMS) or has_dues_request
    has_memory = any(term in lowered for term in _MEMORY_TERMS)
    has_recap = any(term in lowered for term in _RECAP_TERMS)

    if has_recap:
        return "conversation_recap"
    if has_live and has_memory:
        return "mixed"
    if has_live:
        return "live_data"
    if has_memory:
        return "memory"
        
    return None


async def intent_node(state: VyaparAgentState) -> Dict[str, Any]:
    prompt = latest_human_prompt(
        state.get("messages", []),
        state.get("user_prompt", ""),
    ).strip()
    lowered = prompt.lower()

    # 1. Resuming after a clarification: reuse determined intent directly
    if state.get("resume_from_clarification", False):
        existing_intent = state.get("intent", "general") or "general"
        existing_reason = state.get("intent_reason", "") or "resumed from clarification"
        LOGGER.info(
            "intent_node_resume_skip",
            intent=existing_intent,
            reason=existing_reason,
        )
        needs_memory = existing_intent in _MEMORY_REQUIRED_INTENTS
        domains = state.get("domains") or detect_domains(prompt)
        requires_planning = state.get("requires_planning", False)
        return {
            "user_prompt": prompt,
            "intent": existing_intent,
            "intent_reason": existing_reason,
            "domains": domains,
            "memory_query_needed": needs_memory,
            "user_memory_loaded": not needs_memory,
            "store_memory_loaded": not needs_memory,
            "multi_store_memory_loaded": not needs_memory,
            "requires_planning": requires_planning,
        }

    # 2. Fast-path: Severe harm screening
    if any(kw in lowered for kw in _SEVERE_HARM_KEYWORDS):
        LOGGER.warning("intent_severe_harm_refusal", prompt=prompt[:50])
        return {
            "user_prompt": prompt,
            "intent": "harmful",
            "intent_reason": "severe harm keyword detected",
            "domains": [],
            "memory_query_needed": False,
            "user_memory_loaded": True,
            "store_memory_loaded": True,
            "multi_store_memory_loaded": True,
            "requires_planning": False,
            "grader_denied": True,
            "grader_reason": "severe harm keyword detected",
            "goal_status": "complete",
            "goal": "refuse harmful request",
            "final_answer": _REFUSAL_HARMFUL,
            "should_persist_memory": False,
            "messages": [AIMessage(content=_REFUSAL_HARMFUL)],
            "response_metadata": {"intent": "harmful", "fast_path": True},
        }

    # 3. Fast-path: Pure conversational greeting / pleasantry answered directly via fast LLM
    has_business_terms = any(kw in lowered for kw in ("stock", "sale", "sell", "due", "khata", "bill", "report", "order", "product", "inventory"))
    if _CONVERSATIONAL_PATTERNS.search(prompt) and not has_business_terms:
        LOGGER.info("intent_conversational_llm_start", prompt=prompt[:50])
        reply = await _generate_conversational_reply(state, prompt)
        LOGGER.info("intent_conversational_llm_answered", reply_len=len(reply))
        return {
            "user_prompt": prompt,
            "intent": "greeting",
            "intent_reason": "conversational greeting answered directly by LLM",
            "domains": [],
            "memory_query_needed": False,
            "user_memory_loaded": True,
            "store_memory_loaded": True,
            "multi_store_memory_loaded": True,
            "requires_planning": False,
            "goal_status": "complete",
            "goal": "greet user",
            "final_answer": reply,
            "should_persist_memory": False,
            "messages": [AIMessage(content=reply)],
            "response_metadata": {"intent": "greeting", "fast_path": True, "generated_by": "intent_llm"},
        }

    # 4. Fast semantic tool retrieval for candidate tools (in-memory FastEmbed vector search, < 15ms)
    allowed_semantic_tools = []
    tools_desc_text = ""
    is_owner = state.get("is_owner", True)
    permissions = state.get("permissions", [])
    store_id = state.get("store_id") or ""

    try:
        from app.agent.tools.semantic_index import get_semantic_tools_for_query
        from app.agent.tools.registry import is_tool_allowed_for_user

        matched_tools = get_semantic_tools_for_query(prompt, top_k=4)
        allowed_semantic_tools = [
            t for t in matched_tools
            if is_tool_allowed_for_user(t.name, is_owner=is_owner, permissions=permissions)
        ]
        if allowed_semantic_tools:
            tool_lines = []
            for t in allowed_semantic_tools:
                arg_sigs = []
                for arg_name, arg_info in (t.args or {}).items():
                    arg_type = arg_info.get("type", "any") if isinstance(arg_info, dict) else "any"
                    arg_sigs.append(f"{arg_name}: {arg_type}")
                sig = f"{t.name}({', '.join(arg_sigs)})"
                desc = t.description.strip().split("\n")[0] if t.description else ""
                tool_lines.append(f"- {sig}: {desc}")
            tools_desc_text = "\nAvailable Candidate Tools for this query:\n" + "\n".join(tool_lines) + "\n"
    except Exception as exc:
        LOGGER.warning("intent_semantic_candidate_retrieval_failed", error=str(exc)[:150])

    # 5. Single Unified Multi-Tier LLM Call (Intent Classification + Semantic Tool Dispatch in 1 Call)
    intent = None
    reason = ""
    domains: list[str] = []
    requires_planning = False
    dispatched_tool_calls: list[ToolDispatchCall] = []

    llm = get_intent_llm() or get_small_llm()
    if llm and prompt:
        try:
            # Build context from last 6 messages (3 full user+assistant turns)
            history_context = ""
            messages = state.get("messages", [])
            if len(messages) > 1:
                recent_msgs = [m for m in messages[-7:-1] if hasattr(m, "type") and m.type in ("human", "ai")]
                history_lines = []
                for m in recent_msgs:
                    role = "User" if m.type == "human" else "Assistant"
                    content = m.content if isinstance(m.content, str) else str(m.content)
                    history_lines.append(f"{role}: {content[:300]}")
                if history_lines:
                    history_context = "Recent Conversation (last 3 turns):\n" + "\n".join(history_lines) + "\n\n"

            classifier = llm.with_structured_output(IntentClassification)
            user_msg_content = (
                f"{history_context}"
                f"Active Store ID: '{store_id}'\n"
                f"{tools_desc_text}\n"
                f"Latest User Request: {prompt}"
            )
            messages_payload = [
                {"role": "system", "content": INTENT_CLASSIFIER_INSTRUCTION},
                {"role": "user", "content": user_msg_content},
            ]
            response = await asyncio.wait_for(
                classifier.ainvoke(messages_payload, config={"tags": ["hide_stream"]}),
                timeout=12.0,
            )

            if response.intent in _VALID_INTENTS:
                intent = response.intent
                reason = response.reason
                domains = getattr(response, "domains", []) or []
                requires_planning = bool(getattr(response, "requires_planning", False))
                dispatched_tool_calls = getattr(response, "tool_calls", []) or []
            else:
                intent = "general"
                reason = "unrecognized intent from LLM"
        except asyncio.TimeoutError:
            LOGGER.warning("intent_classification_timeout", error="LLM timeout after 12s")
        except Exception as e:
            LOGGER.warning("intent_classification_failed", error=str(e)[:200])

    # 6. Deterministic fallback if LLM classification did not yield an intent
    if not intent:
        intent = _deterministic_intent(prompt) or "general"
        reason = "deterministic fallback routing"
        requires_planning = _requires_planning(prompt)

    # 7. Ensure domain list is populated via pattern detection if empty
    if not domains:
        domains = [] if intent in {"greeting", "harmful", "off_topic"} else detect_domains(prompt)

    intent = intent or "general"
    needs_memory = intent in _MEMORY_REQUIRED_INTENTS

    LOGGER.info(
        "intent_classified",
        intent=intent,
        reason=reason,
        domains=domains,
        needs_memory=needs_memory,
        prompt_len=len(prompt),
        requires_planning=requires_planning,
        tool_call_count=len(dispatched_tool_calls),
    )

    # 8. Greeting / Harmful / Off-topic handling
    if intent == "greeting":
        LOGGER.info("intent_greeting_llm_start", prompt=prompt[:50])
        reply = await _generate_conversational_reply(state, prompt)
        LOGGER.info("intent_greeting_llm_answered", reply_len=len(reply))
        return {
            "user_prompt": prompt,
            "intent": intent,
            "intent_reason": reason,
            "domains": [],
            "memory_query_needed": False,
            "user_memory_loaded": True,
            "store_memory_loaded": True,
            "multi_store_memory_loaded": True,
            "requires_planning": False,
            "goal_status": "complete",
            "goal": "greet user",
            "final_answer": reply,
            "should_persist_memory": False,
            "messages": [AIMessage(content=reply)],
            "response_metadata": {"intent": "greeting", "fast_path": True, "generated_by": "intent_llm"},
        }

    if intent in {"harmful", "off_topic"}:
        reply = _REFUSAL_HARMFUL if intent == "harmful" else _REFUSAL_OFF_TOPIC
        LOGGER.info("intent_refusal_fast_path", intent=intent, reason=reason)
        return {
            "user_prompt": prompt,
            "intent": intent,
            "intent_reason": reason,
            "domains": [],
            "memory_query_needed": False,
            "user_memory_loaded": True,
            "store_memory_loaded": True,
            "multi_store_memory_loaded": True,
            "requires_planning": False,
            "grader_denied": True,
            "grader_reason": reason,
            "goal_status": "complete",
            "goal": f"refuse {intent} request",
            "final_answer": reply,
            "should_persist_memory": False,
            "messages": [AIMessage(content=reply)],
            "response_metadata": {"intent": intent, "fast_path": True, "grader_verdict": intent},
        }

    # 9. Fast-path: Pre-dispatch tool calls from the unified classification in ONE call
    allowed_tool_names = {t.name for t in allowed_semantic_tools}
    if intent == "live_data" and not requires_planning and not needs_memory and dispatched_tool_calls:
        pending_calls = []
        for tc in dispatched_tool_calls:
            t_name = tc.tool_name if hasattr(tc, "tool_name") else tc.get("tool_name")
            if not t_name:
                continue
            if allowed_tool_names and t_name not in allowed_tool_names:
                continue
            args = dict(tc.arguments if hasattr(tc, "arguments") else tc.get("arguments") or {})
            if store_id:
                args.setdefault("store_id", store_id)
            pending_calls.append({
                "call_id": f"call_{uuid.uuid4().hex[:8]}",
                "tool_name": t_name,
                "arguments": args,
            })

        if pending_calls:
            LOGGER.info(
                "intent_unified_tool_dispatch_success",
                tool_count=len(pending_calls),
                tools=[c["tool_name"] for c in pending_calls],
            )
            return {
                "user_prompt": prompt,
                "intent": "live_data",
                "intent_reason": reason or "semantic tools pre-dispatched in single unified LLM call",
                "domains": domains or ["sales"],
                "pending_tool_calls": pending_calls,
                "memory_query_needed": False,
                "user_memory_loaded": True,
                "store_memory_loaded": True,
                "multi_store_memory_loaded": True,
                "requires_planning": False,
            }

    # 10. Standard active business query
    return {
        "user_prompt": prompt,
        "intent": intent,
        "intent_reason": reason,
        "domains": domains,
        "memory_query_needed": needs_memory,
        "user_memory_loaded": not needs_memory,
        "store_memory_loaded": not needs_memory,
        "multi_store_memory_loaded": not needs_memory,
        "requires_planning": requires_planning,
    }
