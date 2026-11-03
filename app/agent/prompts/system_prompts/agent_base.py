"""
app/agent/prompts/system_prompts/agent_base.py
==============================================
Core system prompt builder -- assembles identity, persona, store/owner context,
tool rules, few-shot examples, and current task guidelines.
Rebuilt on every think-node invocation with the latest state.
"""

from __future__ import annotations

from typing import Any
from app.agent.prompts.system_prompts.persona import build_persona_and_context
from app.agent.prompts.system_prompts.tool_rules import TOOL_SELECTION_RULES
from app.agent.prompts.system_prompts.few_shot_examples import FEW_SHOT_TOOL_DECISIONS


def build_agent_base(
    *,
    store_id: str,
    user_prompt: str,
    current_goal: str,
    loop: int,
    max_loops: int,
    user_context: dict[str, Any] | None = None,
    store_context: dict[str, Any] | None = None,
    user_role: str = "Owner",
    is_owner: bool = True,
    permissions: list[str] | None = None,
) -> str:
    """
    Build the complete system base prompt.
    """
    persona_prompt, _meta = build_persona_and_context(
        store_id=store_id,
        user_prompt=user_prompt,
        current_goal=current_goal,
        loop=loop,
        max_loops=max_loops,
        user_context=user_context,
        store_context=store_context,
        user_role=user_role,
        is_owner=is_owner,
        permissions=permissions,
    )

    parts = [
        persona_prompt,
        TOOL_SELECTION_RULES,
        FEW_SHOT_TOOL_DECISIONS,
    ]

    return "\n".join(parts) + "\n"