"""Intent routing node for choosing live tools, chat history, or Mem0."""

from __future__ import annotations

import asyncio
import random
from typing import Any, Dict

import structlog
from pydantic import BaseModel, Field

from app.agent.prompts.intent_prompt import INTENT_CLASSIFIER_INSTRUCTION
from app.agent.state import VyaparAgentState
from app.agent.utils import latest_human_prompt
from app.lib.grader import _SEVERE_HARM_KEYWORDS
from app.lib.llm import get_intent_llm, get_small_llm
from app.agent.tools.clusters import detect_domains
from langchain_core.messages import AIMessage

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

_GREETING_REPLIES = [
    "Hello! 👋 I'm Vyapar Sakha. How can I help with your store today?",
    "Hi there! 😊 Ready to help with your inventory, sales, or restocking. What do you need?",
    "Hey! Welcome back. Ask me anything about your store — inventory, sales, forecasts, and more.",
    "Hello! How can I assist you with your store operations today?",
    "Hi! 👋 I'm here to help. Ask me about your inventory, sales trends, or store insights.",
]

_FAREWELL_TERMS = {"good night", "goodnight", "bye", "goodbye", "good bye", "see you", "cya", "take care", "ttyl", "gn"}
_THANKS_TERMS = {"thanks", "thank you", "thankyou", "thx", "ty", "great", "awesome", "perfect", "nice", "good", "ok thanks", "okay thanks"}
_OPENER_TERMS = {"hi", "hello", "hey", "hii", "helo", "hola", "good morning", "good evening", "good afternoon", "howdy", "sup", "how are you", "how r u"}

_FAREWELL_REPLIES = [
    "Good night! 🌙 Rest well. I'll be here when you need me.",
    "Good night! 🌙 Take care. Come back anytime your store needs attention.",
    "Bye! 👋 See you next time. Vyapar Sakha is always ready.",
]
_THANKS_REPLIES = [
    "You're welcome! 😊 Let me know if there's anything else I can help with.",
    "Happy to help! 🙌 Anything else about your store?",
    "Glad I could help! Ask me anytime. 😊",
]
_OPENER_REPLIES = [
    "Hello! 👋 I'm Vyapar Sakha. How can I help with your store today?",
    "Hi there! 😊 Ready to help with your inventory, sales, or restocking.",
    "Hey! Ask me anything about your store — inventory, sales, forecasts, and more.",
    "Hello! How can I assist with your store operations today?",
]

def _build_greeting_reply(state: VyaparAgentState, prompt: str) -> str:
    """Build a contextually correct reply by analysing the actual greeting prompt."""
    lowered = (prompt or "").lower().strip().rstrip("!.,?")
    messages = state.get("messages", [])
    prior_msgs = [m for m in messages[:-1] if hasattr(m, "type") and m.type in ("human", "ai")]
    is_returning = bool(prior_msgs)

    # Get user's first name if available for personalization
    user_ctx = state.get("user_context") or {}
    name = (user_ctx.get("name") or "").split()[0] if user_ctx.get("name") else ""
    name_part = f", {name}" if name else ""

    # Farewell
    if any(lowered == t or lowered.startswith(t) for t in _FAREWELL_TERMS):
        reply = random.choice(_FAREWELL_REPLIES)
        if name:
            reply = reply.replace("!", f"{name_part}!", 1)
        return reply

    # Thanks / acknowledgment
    if any(lowered == t or lowered.startswith(t) for t in _THANKS_TERMS):
        return random.choice(_THANKS_REPLIES)

    # Opening greeting — differentiate returning vs new session
    if is_returning:
        return f"Welcome back{name_part}! 😊 How can I help you with your store today?"

    return random.choice(_OPENER_REPLIES)


class IntentClassification(BaseModel):
    intent: str = Field(description="The primary intent label: harmful, off_topic, live_data, memory, conversation_recap, mixed, greeting, or general")
    reason: str = Field(description="Short reason for the classification")
    requires_planning: bool = Field(description="True if the request is complex and needs step-by-step planning or strategy")


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
    
    requires_planning = _requires_planning(prompt)

    # When resuming after a clarification, the intent was
    # already determined — reuse it directly.
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

    deterministic = _deterministic_intent(prompt)
    reason = "deterministic routing"
    intent = deterministic

    if intent is None and prompt:
        llm = get_intent_llm() or get_small_llm()
        if llm:
            try:
                # Build context from last 6 messages (3 full user+assistant turns)
                history_context = ""
                messages = state.get("messages", [])
                if len(messages) > 1:
                    # Get last 7 messages excluding the very latest (which is the current prompt)
                    recent_msgs = [m for m in messages[-7:-1] if hasattr(m, "type") and m.type in ("human", "ai")]
                    history_lines = []
                    for m in recent_msgs:
                        role = "User" if m.type == "human" else "Assistant"
                        content = m.content if isinstance(m.content, str) else str(m.content)
                        history_lines.append(f"{role}: {content[:300]}")
                    if history_lines:
                        history_context = "Recent Conversation (last 3 turns):\n" + "\n".join(history_lines) + "\n\n"
                        
                classifier = llm.with_structured_output(IntentClassification)
                messages = [
                    {"role": "system", "content": INTENT_CLASSIFIER_INSTRUCTION},
                    {"role": "user", "content": f"{history_context}Latest User Request: {prompt}"},
                ]
                response = await asyncio.wait_for(
                    classifier.ainvoke(messages, config={"tags": ["hide_stream"]}),
                    timeout=6.0,
                )
                
                if response.intent in _VALID_INTENTS:
                    intent = response.intent
                    reason = response.reason
                    requires_planning = requires_planning or response.requires_planning
                else:
                    intent = "general"
                    reason = "invalid intent from LLM"
            except asyncio.TimeoutError:
                LOGGER.warning("intent_classification_timeout", error="LLM timeout after 6s")
                intent = "general"
                reason = "LLM classification timeout"
            except Exception as e:
                LOGGER.warning("intent_classification_failed", error=str(e))
                intent = "general"
                reason = "LLM classification failed"

    intent = intent or "general"
    # Only load memory when the intent genuinely requires it.
    # live_data / general / conversation_recap queries don't need historical memory;
    # the actual data comes from live tools. This is the primary knob that prevents
    # the 25-second memory_query delay for ~80% of requests.
    needs_memory = intent in _MEMORY_REQUIRED_INTENTS

    LOGGER.info(
        "intent_classified",
        intent=intent,
        reason=reason,
        needs_memory=needs_memory,
        prompt_len=len(prompt),
        requires_planning=requires_planning,
    )

    # ── Greeting fast-path ─────────────────────────────────────────────────
    # Skip memory_query, think, and all heavy nodes entirely.
    # Generate a contextual reply from checkpointer history right here.
    if intent == "greeting":
        reply = _build_greeting_reply(state, prompt)
        LOGGER.info("intent_greeting_fast_path", reply_len=len(reply))
        return {
            "user_prompt": prompt,
            "intent": intent,
            "intent_reason": reason,
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
            "response_metadata": {"intent": "greeting", "fast_path": True},
        }
    # ── End greeting fast-path ─────────────────────────────────────────────

    if intent in {"harmful", "off_topic"}:
        reply = _REFUSAL_HARMFUL if intent == "harmful" else _REFUSAL_OFF_TOPIC
        LOGGER.info("intent_refusal_fast_path", intent=intent, reason=reason)
        return {
            "user_prompt": prompt,
            "intent": intent,
            "intent_reason": reason,
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

    domains = [] if intent in {"greeting", "harmful", "off_topic"} else detect_domains(prompt)
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
