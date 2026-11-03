"""
app/agent/prompts/__init__.py
=============================
Prompt registry for the Vyapar Copilot agent.

All LLM-facing prompt strings live here as plain Python modules so they
can be edited, reviewed, and versioned independently of the agent logic.

Layout:

    app/agent/prompts/
    ├── __init__.py
    ├── system_prompts/        # prompts injected into the main system message
    │   ├── __init__.py
    │   ├── agent_base.py      # identity, store context, goal, loop counter
    │   ├── memory_warning.py  # memory-is-not-live-data warning
    │   ├── tool_synthesis.py  # "you have already gathered data..."
    │   ├── first_loop.py      # "use the available tools..."
    │   └── available_tools.py # "available tools this loop: ..."
    └── summarizer_prompts/    # instructions for the small summarizer LLM
        ├── __init__.py
        ├── memory.py          # mem0 long-term memory summarizer
        ├── tool.py            # tool payload summarizer
        ├── history.py         # sliding-window history summarizer
        └── title.py           # short chat-title generation
    └── classifier_prompts/    # instructions for the lightweight guardrail LLM
        ├── __init__.py
        └── harm_check.py      # harm-classification instruction for the grader
"""

from app.agent.prompts.system_prompts import (
    agent_base,
    memory_warning,
    tool_synthesis,
    first_loop,
    available_tools,
)
from app.agent.prompts.summarizer_prompts import (
    memory as summarizer_memory,
    tool as summarizer_tool,
    history as summarizer_history,
    title as summarizer_title,
)
from app.agent.prompts.classifier_prompts import CLASSIFIER_HARM_INSTRUCTION
from app.agent.prompts.voice_prompt import build_voice_system_prompt

__all__ = [
    "agent_base",
    "memory_warning",
    "tool_synthesis",
    "first_loop",
    "available_tools",
    "summarizer_memory",
    "summarizer_tool",
    "summarizer_history",
    "summarizer_title",
    "CLASSIFIER_HARM_INSTRUCTION",
    "build_voice_system_prompt",
]
