"""
app/agent/prompts/system_prompts/__init__.py
============================================
System prompt fragments injected into the main system message.
"""

from app.agent.prompts.system_prompts.agent_base import build_agent_base
from app.agent.prompts.system_prompts.dynamic_builder import build_dynamic_agent_base
from app.agent.prompts.system_prompts.persona import build_persona_and_context
from app.agent.prompts.system_prompts.tool_rules import TOOL_SELECTION_RULES
from app.agent.prompts.system_prompts.few_shot_examples import FEW_SHOT_TOOL_DECISIONS, FEW_SHOT_SYNTHESIS
from app.agent.prompts.system_prompts.memory_warning import MEMORY_WARNING
from app.agent.prompts.system_prompts.tool_synthesis import build_tool_synthesis
from app.agent.prompts.system_prompts.first_loop import FIRST_LOOP
from app.agent.prompts.system_prompts.available_tools import build_available_tools

__all__ = [
    "build_agent_base",
    "build_dynamic_agent_base",
    "build_persona_and_context",
    "TOOL_SELECTION_RULES",
    "FEW_SHOT_TOOL_DECISIONS",
    "FEW_SHOT_SYNTHESIS",
    "MEMORY_WARNING",
    "build_tool_synthesis",
    "FIRST_LOOP",
    "build_available_tools",
]