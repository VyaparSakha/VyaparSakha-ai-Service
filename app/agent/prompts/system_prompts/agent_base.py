"""
app/agent/prompts/system_prompts/agent_base.py
==============================================
Core system prompt builder -- assembles identity, persona, store/owner context,
tool rules, few-shot examples, and current task guidelines.
Rebuilt on every think-node invocation with the latest state.
"""

from __future__ import annotations

from typing import Any
from app.agent.prompts.system_prompts.dynamic_builder import build_dynamic_agent_base


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
    domains: list[str] | None = None,
) -> str:
    """
    Build the complete system base prompt dynamically tailored to active domains.
    """
    if domains is None and user_prompt:
        from app.agent.tools.clusters import detect_domains
        domains = detect_domains(user_prompt)

    return build_dynamic_agent_base(
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
        domains=domains,
    )
