"""
app/agent/nodes/reflection_node.py
==================================
Reflection node — generates an improved approach based on the critic's feedback.
"""

from typing import Dict, Any
import structlog

from app.agent.state import VyaparAgentState
from app.lib.llm import get_critic_llm, get_small_llm
from langchain_core.messages import SystemMessage, HumanMessage

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.reflection")

def get_llm():
    return get_critic_llm() or get_small_llm()


async def reflection_node(state: VyaparAgentState) -> Dict[str, Any]:
    # Hard cap: only allow 1 reflection retry. The screenshots showed
    # 2 full critic-reflection cycles (50s total). If the first retry
    # doesn't fix it, more retries rarely help and just waste time.
    reflection_count = state.get("reflection_count", 0)
    if reflection_count >= 1:
        LOGGER.info("reflection_node_max_reached", reflection_count=reflection_count)
        return {"goal_status": "complete"}  # Accept the best answer we have

    # Use evaluation model (Groq 120B/20B) — does not consume Gemini quotas
    llm = get_llm()
    if not llm:
        return {"goal_status": "complete"}
        
    reason = state.get("critic_reason", "")
    missing = state.get("critic_missing_information", [])
    
    LOGGER.info("reflection_node_start", reason=reason)
    
    system_prompt = (
        "You are the reflection component of Vyapar Sakha. "
        "The previous approach failed. The Critic identified what is missing. "
        "Provide an improved execution direction. Which tools should be used next to get the missing information? "
        "Respond with a short, direct paragraph on what to do differently to fix it. "
        "Do not repeat successful work (e.g. if sales is already gathered, only gather forecast)."
    )
    
    recent_errors = [res for res in state.get("tool_results", []) if not res.get("success")]
    error_context = ""
    if recent_errors:
        error_lines = [f"Tool {r['tool_name']} failed: {r['error']}" for r in recent_errors]
        error_context = "\nRecent Tool Failures:\n" + "\n".join(error_lines)
    
    user_msg = (
        f"Critic Reason: {reason}\n"
        f"Missing Information: {', '.join(missing)}\n"
        f"{error_context}\n\n"
        "What should we do differently next? If a tool failed due to missing arguments, suggest fixing the arguments."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg)
    ]
    
    # Hide from SSE stream by stripping callbacks
    try:
        response = await llm.ainvoke(messages, config={"callbacks": []})
        reflection_text = response.content
    except Exception as e:
        LOGGER.error("reflection_node_error", error=str(e))
        return {"goal_status": "complete"} # Fail open
        
    LOGGER.info("reflection_node_result", reflection_len=len(reflection_text))
    
    # Add the reflection back into the conversation history so think_node sees it
    reflection_msg = HumanMessage(content=f"Evaluation failed.\nCritic says: {reason}\nReflection for next step: {reflection_text}")
    
    # Increment loop count so reflection respects the global max_loops limit
    current_loop = state.get("loop_count", 0)
    
    return {
        "goal_status": "in_progress",
        "reflection": reflection_text,
        "reflection_count": reflection_count + 1,
        "messages": [reflection_msg],
        "loop_count": current_loop + 1
    }
