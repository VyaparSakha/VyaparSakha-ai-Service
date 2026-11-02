"""
app/agent/prompts/system_prompts/tool_synthesis.py
===================================================
Prompt fragment injected when the agent has gathered data from tool calls.
Instructs the LLM on structuring its response cleanly with tables, metrics,
and risk badges.
"""

from __future__ import annotations

import json
from app.agent.prompts.system_prompts.few_shot_examples import FEW_SHOT_SYNTHESIS


def build_tool_synthesis(
    *,
    results_so_far: list[dict],
    inventory_context: dict,
    sales_context: dict,
    forecast_context: dict,
    insights_context: list,
    candidate_products: list | None = None,
    discovery_metrics: dict | None = None,
) -> str:
    """
    Build the tool-synthesis prompt fragment.
    """
    candidate_products = candidate_products or []
    discovery_metrics = discovery_metrics or {}
    
    top_candidates = candidate_products[:20] if len(candidate_products) > 20 else candidate_products
    
    candidates_context = ""
    if candidate_products:
        candidates_context = (
            f"<discovery_candidates>\n"
            f"Total Candidates Found: {len(candidate_products)}\n"
            f"Showing Top {len(top_candidates)} Candidates:\n"
            f"{json.dumps(top_candidates, default=str)}\n"
            f"</discovery_candidates>\n\n"
        )
        
    metrics_context = ""
    if discovery_metrics:
        metrics_context = (
            f"<discovery_metrics>\n"
            f"{json.dumps(discovery_metrics, default=str)}\n"
            f"</discovery_metrics>\n\n"
        )

    doc_download_instruction = ""
    for r in results_so_far:
        data = r.get("data", {})
        if isinstance(data, dict):
            file_info = data.get("file") if isinstance(data.get("file"), dict) else data
            download_url = file_info.get("download_url")
            filename = file_info.get("filename", "report.xlsx")
            if download_url:
                doc_download_instruction += (
                    f"\n[CRITICAL: DOCUMENT GENERATED & READY FOR DOWNLOAD]\n"
                    f"File Name: {filename}\n"
                    f"Download URL: {download_url}\n"
                    f"You MUST prominently include a Markdown download button/link in your Executive Summary or response:\n"
                    f"📥 **[Download {filename}]({download_url})**\n\n"
                )

    synthesis_instruction = (
        f"DATA SYNTHESIS INSTRUCTIONS ({len(results_so_far)} tool call(s) executed):\n"
        f"{doc_download_instruction}"
        "1. You have sufficient tool data to fulfill the user's request. Respond directly WITHOUT calling more tools unless essential data is missing.\n"
        "2. Ground every claim, number, product name, and metric strictly in the gathered context below (<tool_results_gathered>). Never hallucinate product names or stock quantities. Do NOT copy example names (such as Basmati Rice or Tata Salt) from example templates.\n"
        "3. Format your response into structured sections:\n"
        "   - **Executive Summary**: 2-3 direct sentences highlighting the key trend, risk, or milestone (plus file download link if a document was created).\n"
        "   - **Action Recommendations**: Priority-ordered practical next steps for the store owner.\n"
        "   - Note: The UI automatically renders interactive charts, sortable tables, and KPI cards from tool data. Do NOT repeat every single number or copy giant raw tables into text.\n\n"
        f"{FEW_SHOT_SYNTHESIS}\n\n"
    )

    context_blocks = (
        f"<tool_results_gathered>\n{json.dumps(results_so_far, default=str)}\n</tool_results_gathered>\n\n"
        f"<inventory_data>\n{json.dumps(inventory_context, default=str)}\n</inventory_data>\n\n"
        f"<sales_data>\n{json.dumps(sales_context, default=str)}\n</sales_data>\n\n"
        f"<forecast_data>\n{json.dumps(forecast_context, default=str)}\n</forecast_data>\n\n"
        f"<insights>\n{json.dumps(insights_context, default=str)}\n</insights>\n\n"
        f"{metrics_context}"
        f"{candidates_context}"
    )


    return synthesis_instruction + context_blocks