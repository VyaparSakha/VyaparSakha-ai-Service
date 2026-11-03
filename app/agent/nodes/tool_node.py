"""
app/agent/nodes/tool_node.py
=============================
Tool node — executes tool calls requested by the think node.
"""

from __future__ import annotations

import time
from typing import Dict, Any

import structlog

from app.agent.state import VyaparAgentState, make_tool_result
from app.agent.tools.registry import get_tool_by_name, is_tool_allowed_for_user
from pydantic import ValidationError

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.tool")


async def tool_node(state: VyaparAgentState) -> Dict[str, Any]:
    pending = state.get("pending_tool_calls", [])
    loop = state.get("loop_count", 0)
    store_id = state.get("store_id", "unknown")

    LOGGER.info(
        "tool_node_start",
        loop=loop,
        store_id=store_id,
        tools_to_run=[c["tool_name"] for c in pending],
    )

    results = []
    updates: Dict[str, Any] = {"pending_tool_calls": []}
    error_counts = dict(state.get("tool_error_counts", {}))

    for call in pending:
        tool_name = call["tool_name"]
        tool = get_tool_by_name(tool_name)

        if not tool:
            LOGGER.warning("tool_node_unknown_tool", tool_name=tool_name, loop=loop)
            results.append(
                make_tool_result(
                    call_id=call["call_id"],
                    tool_name=tool_name,
                    data=None,
                    error=f"Tool '{tool_name}' not found in registry",
                    loop_index=loop,
                )
            )
            continue

        # RBAC Check: Ensure the tool execution is permitted for user's role & permissions
        is_owner = state.get("is_owner", True)
        permissions = state.get("permissions", [])
        user_role = state.get("user_role", "Owner" if is_owner else "Employee")

        if not is_tool_allowed_for_user(tool_name, is_owner=is_owner, permissions=permissions):
            LOGGER.warning(
                "tool_node_permission_denied",
                tool_name=tool_name,
                user_role=user_role,
                is_owner=is_owner,
                loop=loop,
            )
            results.append(
                make_tool_result(
                    call_id=call["call_id"],
                    tool_name=tool_name,
                    data=None,
                    error=f"Permission Denied: Your assigned role '{user_role}' does not have permission to execute '{tool_name}'.",
                    loop_index=loop,
                )
            )
            continue

        t0 = time.perf_counter()
        try:
            # Ensure store_id is always passed to every tool
            args = dict(call["arguments"])
            if "store_id" not in args:
                args["store_id"] = store_id

            LOGGER.debug("tool_node_invoking", tool_name=tool_name, args=args, loop=loop)
            # Pass user_id/store_id in configurable so search_memory can
            # resolve identity securely without LLM providing it.
            invoke_config = {
                "configurable": {
                    "user_id": state.get("user_id", ""),
                    "store_id": store_id,
                }
            }
            data = await tool.ainvoke(args, config=invoke_config)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

            raw_data = data.model_dump() if hasattr(data, "model_dump") else data

            LOGGER.info(
                "tool_node_success",
                tool_name=tool_name,
                loop=loop,
                latency_ms=elapsed_ms,
            )

            error_counts[tool_name] = 0

            results.append(
                make_tool_result(
                    call_id=call["call_id"],
                    tool_name=tool_name,
                    data=raw_data,
                    loop_index=loop,
                )
            )

            # Route data into the correct context bucket
            if "inventory" in tool_name or "stock" in tool_name:
                ctx = dict(state.get("inventory_context", {}))
                ctx[tool_name] = raw_data
                updates["inventory_context"] = ctx
            elif "sales" in tool_name or "selling" in tool_name:
                ctx = dict(state.get("sales_context", {}))
                ctx[tool_name] = raw_data
                updates["sales_context"] = ctx
            elif "forecast" in tool_name or "restock" in tool_name:
                ctx = dict(state.get("forecast_context", {}))
                ctx[tool_name] = raw_data
                updates["forecast_context"] = ctx
            elif "insights" in tool_name:
                ctx = list(state.get("insights_context", []))
                if isinstance(raw_data, list):
                    ctx.extend(raw_data)
                else:
                    ctx.append(raw_data)
                updates["insights_context"] = ctx
            elif tool_name == "search_memory":
                # Accumulate retrieved memories; deduplicate by memory_id
                from app.agent.tools.memory.search import MAX_MEMORY_SEARCH_CALLS
                current_retrieved = list(state.get("retrieved_memories", []))
                current_calls = int(state.get("memory_search_calls", 0))

                if current_calls < MAX_MEMORY_SEARCH_CALLS:
                    new_memories = raw_data.get("results", []) if isinstance(raw_data, dict) else []
                    existing_ids = {m.get("memory_id", "") for m in current_retrieved if m.get("memory_id")}
                    for mem in new_memories:
                        mid = mem.get("memory_id", "")
                        if not mid or mid not in existing_ids:
                            current_retrieved.append(mem)
                            if mid:
                                existing_ids.add(mid)
                    updates["retrieved_memories"] = current_retrieved
                    updates["memory_search_calls"] = current_calls + 1
                    LOGGER.info(
                        "memory_search_accumulated",
                        new_count=len(new_memories),
                        total=len(current_retrieved),
                        calls_used=current_calls + 1,
                    )
                else:
                    LOGGER.warning("memory_search_budget_exceeded", calls=current_calls)



            # Extract candidates if this looks like a discovery tool
            if isinstance(raw_data, dict) and "items" in raw_data and isinstance(raw_data["items"], list):
                # Only add if it looks like a list of products
                if len(raw_data["items"]) > 0 and "product_id" in raw_data["items"][0]:
                    candidates = list(state.get("candidate_products", []))
                    candidates.extend(raw_data["items"])
                    updates["candidate_products"] = candidates
                    
                    # Update metrics
                    metrics = dict(state.get("discovery_metrics", {}))
                    metrics[tool_name] = {
                        "candidates_found": len(raw_data["items"]),
                        "count": raw_data.get("count", len(raw_data["items"]))
                    }
                    updates["discovery_metrics"] = metrics


        except ValidationError as ve:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            error_details = []
            for err in ve.errors():
                field = ".".join(str(loc) for loc in err.get("loc", []))
                msg = err.get("msg", "Validation error")
                error_details.append(f"Field '{field}': {msg}")
                
            formatted_error = f"Validation Error in {tool_name}. Please fix the following arguments:\n" + "\n".join(error_details)
            error_counts[tool_name] = error_counts.get(tool_name, 0) + 1
            LOGGER.warning(
                "tool_node_validation_error",
                tool_name=tool_name,
                loop=loop,
                error=formatted_error,
                latency_ms=elapsed_ms,
            )
            results.append(
                make_tool_result(
                    call_id=call["call_id"],
                    tool_name=tool_name,
                    data=None,
                    error=formatted_error,
                    loop_index=loop,
                )
            )

        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            error_counts[tool_name] = error_counts.get(tool_name, 0) + 1
            LOGGER.error(
                "tool_node_error",
                tool_name=tool_name,
                loop=loop,
                error=str(exc),
                latency_ms=elapsed_ms,
                exc_info=True,
            )
            results.append(
                make_tool_result(
                    call_id=call["call_id"],
                    tool_name=tool_name,
                    data=None,
                    error=str(exc),
                    loop_index=loop,
                )
            )

    LOGGER.info(
        "tool_node_complete",
        loop=loop,
        results_count=len(results),
        successes=sum(1 for r in results if r["success"]),
        failures=sum(1 for r in results if not r["success"]),
    )

    updates["tool_results"] = results
    updates["tool_error_counts"] = error_counts
    return updates
