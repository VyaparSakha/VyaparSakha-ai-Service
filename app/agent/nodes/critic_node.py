"""
app/agent/nodes/critic_node.py
==============================
Critic node — evaluates the draft answer against the goal and available data.
"""

from typing import Dict, Any, List
import asyncio
import structlog
from pydantic import BaseModel, Field

from app.agent.state import VyaparAgentState
from app.lib.llm import get_critic_llm, get_small_llm
from langchain_core.messages import SystemMessage, HumanMessage

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.critic")

class CriticResult(BaseModel):
    status: str = Field(description="Must be 'PASS' if the answer is good, or 'FAIL' if it is missing information or contradicts the context.")
    reason: str = Field(description="A short explanation of why it passed or failed.")
    missing_information: List[str] = Field(description="A list of information that is missing from the answer but required by the goal. Empty if PASS.")
    needs_retry: bool = Field(description="True if the agent should retry to get the missing information.")

def get_llm():
    return get_critic_llm() or get_small_llm()


async def critic_node(state: VyaparAgentState) -> Dict[str, Any]:
    # Use dedicated evaluation model (Groq 120B/20B) — leaves Gemini quota for main thinking
    llm = get_llm()
    if not llm:
        return {"goal_status": "complete"}
    
    goal = state.get("goal", "")
    final_answer = state.get("final_answer", "")
    
    # We strip callbacks so this evaluation is hidden from the SSE stream.
    structured_llm = llm.with_structured_output(CriticResult).with_config({"callbacks": []})
    
    system_prompt = (
        "You are an evaluator for Vyapar Copilot.\n"
        "Your job is to check if the drafted 'Final Answer' satisfies the user's 'Goal', "
        "and if it's supported by the retrieved context.\n"
        "Check for:\n"
        "1. Goal completion: Did the agent actually answer the question?\n"
        "2. Data support: Are the claims supported by the available data?\n"
        "3. Tool coverage: Were the necessary tools used?\n"
        "4. Completeness: Is any important part of the question ignored?\n"
        "5. Document downloads: If a document generation tool ran and produced a file/download URL, check that the draft answer includes a download link.\n"
        "Do NOT invent data. Do not execute tools. Just evaluate.\n"
    )

    tool_results = state.get("tool_results", [])

    # Create an abbreviated context summary to avoid blowing up the context window
    context_str = "Available context summary:\n"
    for r in tool_results:
        t_name = r.get("tool_name", "")
        t_success = r.get("success", False)
        context_str += f"- Tool: {t_name} | Success: {t_success}\n"

        data = r.get("data", {})
        if isinstance(data, dict):
            file_info = data.get("file") if isinstance(data.get("file"), dict) else data
            if isinstance(file_info, dict) and file_info.get("download_url"):
                context_str += (
                    f"  -> File Created: {file_info.get('filename')}\n"
                    f"  -> Download URL: {file_info.get('download_url')}\n"
                )

    
    user_msg = (
        f"Goal: {goal}\n\n"
        f"{context_str}\n\n"
        f"Drafted Answer:\n{final_answer}\n"
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg)
    ]
    
    try:
        LOGGER.info("critic_node_evaluating")
        result: CriticResult = await structured_llm.ainvoke(messages)
    except Exception as e:
        LOGGER.error("critic_node_error", error=str(e))
        return {"goal_status": "complete"} # Fail open
        
    status = result.status.upper()
    LOGGER.info("critic_node_result", status=status, reason=result.reason)
    
    reflection_count = state.get("reflection_count", 0)
    if status == "FAIL" and result.needs_retry and reflection_count < 1:
        return {
            "goal_status": "reflect",
            "critic_status": status,
            "critic_reason": result.reason,
            "critic_missing_information": result.missing_information
        }

    if status == "FAIL" and reflection_count >= 1:
        LOGGER.info("critic_node_max_reflection_reached", reflection_count=reflection_count)

    return {
        "goal_status": "complete",
        "critic_status": status,
        "critic_reason": result.reason,
        "critic_missing_information": result.missing_information
    }
