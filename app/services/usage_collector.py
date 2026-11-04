"""
app/services/usage_collector.py
================================
Tracks, normalizes, and aggregates actual AI model usage (tokens, audio duration,
provider breakdown) across LangGraph and voice sessions.

Integrates with LangChain callbacks and astream_events to ensure zero double-counting,
accurate multi-provider tracking (Gemini, Groq, NVIDIA, Ollama), and secure
server-to-server reporting back to the Express billing engine.
"""

from __future__ import annotations

import time
import math
import httpx
import structlog
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from app.config.settings import get_settings

LOGGER = structlog.get_logger("vyaparsathi.ai.usage")


class NormalizedModelUsage:
    """Normalized usage data from a single model call."""

    def __init__(
        self,
        execution_id: str,
        run_id: str,
        user_id: str,
        store_id: str,
        feature: str,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        audio_input_units: float = 0.0,
        audio_output_units: float = 0.0,
        provider_usage_status: str = "verified",
        timestamp: float | None = None,
    ):
        self.execution_id = execution_id
        self.run_id = run_id
        self.user_id = user_id
        self.store_id = store_id
        self.feature = feature
        self.provider = provider
        self.model = model
        self.input_tokens = max(0, int(input_tokens or 0))
        self.output_tokens = max(0, int(output_tokens or 0))
        self.total_tokens = max(0, int(total_tokens or (self.input_tokens + self.output_tokens)))
        self.audio_input_units = max(0.0, float(audio_input_units or 0.0))
        self.audio_output_units = max(0.0, float(audio_output_units or 0.0))
        self.provider_usage_status = provider_usage_status
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> dict:
        return {
            "executionId": self.execution_id,
            "runId": self.run_id,
            "userId": self.user_id,
            "storeId": self.store_id,
            "feature": self.feature,
            "provider": self.provider,
            "model": self.model,
            "inputTokens": self.input_tokens,
            "outputTokens": self.output_tokens,
            "totalTokens": self.total_tokens,
            "audioInputUnits": self.audio_input_units,
            "audioOutputUnits": self.audio_output_units,
            "providerUsageStatus": self.provider_usage_status,
            "timestamp": self.timestamp,
        }


class UsageCollector:
    """
    Aggregates model invocations for one logical executionId.
    Deduplicates by run_id to prevent double-counting between callbacks and event streams.
    """

    def __init__(
        self,
        execution_id: str,
        user_id: str = "default_user",
        store_id: str = "",
        feature: str = "copilot",
    ):
        self.execution_id = execution_id
        self.user_id = user_id
        self.store_id = store_id
        self.feature = feature
        self._runs: Dict[str, NormalizedModelUsage] = {}
        self.total_audio_in_sec: float = 0.0
        self.total_audio_out_sec: float = 0.0
        self._last_deducted_input_tokens: int = 0
        self._last_deducted_output_tokens: int = 0
        self._last_deducted_audio_in: float = 0.0
        self._last_deducted_audio_out: float = 0.0
        self._stage_count: int = 0
        self.is_exhausted: bool = False

    def add_model_usage(
        self,
        run_id: str,
        output: Any,
        metadata: Optional[dict] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Optional[NormalizedModelUsage]:
        """
        Extracts verified token usage from output and adds to ledger.
        Skips if run_id has already been recorded with verified token counts.
        """
        if not run_id:
            return None

        # If already recorded with verified usage, prevent double-counting
        existing = self._runs.get(run_id)
        if existing and existing.provider_usage_status == "verified" and existing.total_tokens > 0:
            return None

        metadata = metadata or {}
        resolved_provider = provider or metadata.get("ls_provider") or metadata.get("provider") or "default"
        resolved_model = model or metadata.get("ls_model_name") or metadata.get("model") or "default"

        in_tokens = 0
        out_tokens = 0
        tot_tokens = 0
        audio_in = 0.0
        audio_out = 0.0
        status = "missing"

        # 1. Handle LangChain LLMResult (output.generations -> ChatGeneration -> message)
        if hasattr(output, "generations") and output.generations:
            for gen_list in output.generations:
                for gen in (gen_list or []):
                    msg = getattr(gen, "message", None)
                    if msg:
                        msg_usage = getattr(msg, "usage_metadata", None)
                        if isinstance(msg_usage, dict) and msg_usage:
                            in_tokens += msg_usage.get("input_tokens") or 0
                            out_tokens += msg_usage.get("output_tokens") or 0
                            tot_tokens += msg_usage.get("total_tokens") or (in_tokens + out_tokens)
                            if in_tokens > 0 or out_tokens > 0:
                                status = "verified"

                        msg_resp = getattr(msg, "response_metadata", {}) or {}
                        if isinstance(msg_resp, dict) and msg_resp:
                            if msg_resp.get("model_name") and resolved_model == "default":
                                resolved_model = msg_resp["model_name"]
                            if msg_resp.get("model_provider") and resolved_provider == "default":
                                resolved_provider = msg_resp["model_provider"]
                            if status != "verified":
                                tok_u = msg_resp.get("token_usage") or msg_resp.get("usage") or {}
                                if isinstance(tok_u, dict):
                                    in_tokens += tok_u.get("prompt_tokens") or tok_u.get("input_tokens") or 0
                                    out_tokens += tok_u.get("completion_tokens") or tok_u.get("output_tokens") or 0
                                    tot_tokens += tok_u.get("total_tokens") or (in_tokens + out_tokens)
                                    if in_tokens > 0 or out_tokens > 0:
                                        status = "verified"

                    gen_info = getattr(gen, "generation_info", {}) or {}
                    if isinstance(gen_info, dict) and gen_info:
                        if gen_info.get("model_name") and resolved_model == "default":
                            resolved_model = gen_info["model_name"]

        # 2. Standard LangChain AIMessage / AIMessageChunk usage_metadata
        usage_meta = getattr(output, "usage_metadata", None)
        if isinstance(usage_meta, dict) and usage_meta:
            in_tokens = max(in_tokens, usage_meta.get("input_tokens") or 0)
            out_tokens = max(out_tokens, usage_meta.get("output_tokens") or 0)
            tot_tokens = max(tot_tokens, usage_meta.get("total_tokens") or (in_tokens + out_tokens))
            if in_tokens > 0 or out_tokens > 0 or tot_tokens > 0:
                status = "verified"

        # 3. Provider response_metadata (Google GenAI, OpenAI, Groq, NVIDIA)
        resp_meta = getattr(output, "response_metadata", {}) or {}
        if isinstance(resp_meta, dict) and resp_meta:
            if resp_meta.get("model_name") and resolved_model == "default":
                resolved_model = resp_meta["model_name"]
            elif resp_meta.get("model") and resolved_model == "default":
                resolved_model = resp_meta["model"]
            if resp_meta.get("model_provider") and resolved_provider == "default":
                resolved_provider = resp_meta["model_provider"]

            if status != "verified":
                tok_usage = resp_meta.get("token_usage") or resp_meta.get("usage") or {}
                if isinstance(tok_usage, dict):
                    in_tokens = max(in_tokens, tok_usage.get("prompt_tokens") or tok_usage.get("input_tokens") or 0)
                    out_tokens = max(out_tokens, tok_usage.get("completion_tokens") or tok_usage.get("output_tokens") or 0)
                    tot_tokens = max(tot_tokens, tok_usage.get("total_tokens") or (in_tokens + out_tokens))
                    if in_tokens > 0 or out_tokens > 0:
                        status = "verified"

            audio_in = max(audio_in, float(resp_meta.get("input_audio_seconds") or 0.0))
            audio_out = max(audio_out, float(resp_meta.get("output_audio_seconds") or 0.0))

        # 4. Handle LLMResult structure if passed with llm_output
        if hasattr(output, "llm_output") and isinstance(output.llm_output, dict):
            llm_out = output.llm_output
            if llm_out.get("model_name") and resolved_model == "default":
                resolved_model = llm_out["model_name"]
            token_usage = llm_out.get("token_usage") or {}
            if isinstance(token_usage, dict) and status != "verified":
                in_tokens = max(in_tokens, token_usage.get("prompt_tokens") or in_tokens)
                out_tokens = max(out_tokens, token_usage.get("completion_tokens") or out_tokens)
                tot_tokens = max(tot_tokens, token_usage.get("total_tokens") or (in_tokens + out_tokens))
                if in_tokens > 0 or out_tokens > 0:
                    status = "verified"

        # Provider and Model Normalization
        prov_lower = resolved_provider.lower()
        mod_lower = resolved_model.lower()
        if resolved_provider == "default":
            if "gemini" in mod_lower or "google" in mod_lower:
                resolved_provider = "gemini"
            elif "llama" in mod_lower or "groq" in mod_lower or "mixtral" in mod_lower:
                resolved_provider = "groq"
            elif "nvidia" in mod_lower:
                resolved_provider = "nvidia"
            else:
                resolved_provider = "gemini"
        elif "google_genai" in prov_lower:
            resolved_provider = "gemini"

        if resolved_model == "default":
            if "google" in prov_lower or "gemini" in prov_lower:
                resolved_model = "gemini-2.5-flash"

        usage = NormalizedModelUsage(
            execution_id=self.execution_id,
            run_id=run_id,
            user_id=self.user_id,
            store_id=self.store_id,
            feature=self.feature,
            provider=resolved_provider,
            model=resolved_model,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            total_tokens=tot_tokens,
            audio_input_units=audio_in,
            audio_output_units=audio_out,
            provider_usage_status=status,
        )

        self._runs[run_id] = usage
        LOGGER.debug(
            "usage_recorded",
            run_id=run_id,
            provider=resolved_provider,
            model=resolved_model,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            status=status,
        )
        return usage

    def add_voice_usage(self, audio_in_seconds: float, audio_out_seconds: float):
        """Adds audio duration from a live voice session."""
        self.total_audio_in_sec += max(0.0, float(audio_in_seconds or 0.0))
        self.total_audio_out_sec += max(0.0, float(audio_out_seconds or 0.0))

    def get_summary(self, execution_status: str = "completed") -> dict:
        """
        Produces aggregated logical execution usage across all nodes.
        Preserves breakdown by provider & model.
        """
        total_in = 0
        total_out = 0
        total_tok = 0
        total_audio_in = self.total_audio_in_sec
        total_audio_out = self.total_audio_out_sec

        breakdown_map: Dict[str, dict] = {}
        primary_provider = "gemini"
        primary_model = "gemini-2.5-flash"

        for usage in self._runs.values():
            total_in += usage.input_tokens
            total_out += usage.output_tokens
            total_tok += usage.total_tokens
            total_audio_in += usage.audio_input_units
            total_audio_out += usage.audio_output_units

            key = f"{usage.provider}::{usage.model}"
            if key not in breakdown_map:
                breakdown_map[key] = {
                    "provider": usage.provider,
                    "model": usage.model,
                    "inputTokens": 0,
                    "outputTokens": 0,
                    "audioInputUnits": 0.0,
                    "audioOutputUnits": 0.0,
                    "callCount": 0,
                    "providerUsageStatus": usage.provider_usage_status,
                }
            breakdown_map[key]["inputTokens"] += usage.input_tokens
            breakdown_map[key]["outputTokens"] += usage.output_tokens
            breakdown_map[key]["audioInputUnits"] += usage.audio_input_units
            breakdown_map[key]["audioOutputUnits"] += usage.audio_output_units
            breakdown_map[key]["callCount"] += 1
            if usage.provider_usage_status == "verified":
                breakdown_map[key]["providerUsageStatus"] = "verified"

            if usage.input_tokens > 0 or usage.output_tokens > 0:
                primary_provider = usage.provider
                primary_model = usage.model

        # Estimated credit calculation for immediate client display (Express settlement is authoritative)
        prov_l = primary_provider.lower()
        mod_l = primary_model.lower()
        if "gemini" in prov_l or "gemini" in mod_l or "flash" in mod_l:
            in_rate = 0.05
            out_rate = 0.15
            audio_in_rate = 0.05
            audio_out_rate = 0.10
            max_cap = 50
        else:
            in_rate = 2.0
            out_rate = 6.0
            audio_in_rate = 1.0
            audio_out_rate = 2.0
            max_cap = 250

        token_cost = total_in * in_rate + total_out * out_rate
        token_credits = math.ceil(token_cost / 1000) if token_cost > 0 else 0
        audio_credits = math.ceil(total_audio_in * audio_in_rate + total_audio_out * audio_out_rate)
        estimated_credits = token_credits + audio_credits
        if estimated_credits > 0 or total_tok > 0 or total_audio_in > 0:
            estimated_credits = max(1, min(estimated_credits, max_cap))

        return {
            "executionId": self.execution_id,
            "userId": self.user_id,
            "storeId": self.store_id,
            "feature": self.feature,
            "provider": primary_provider,
            "model": primary_model,
            "inputTokens": total_in,
            "outputTokens": total_out,
            "totalTokens": total_tok,
            "audioInputUnits": round(total_audio_in, 2),
            "audioOutputUnits": round(total_audio_out, 2),
            "creditsConsumed": estimated_credits,
            "modelBreakdown": list(breakdown_map.values()),
            "executionStatus": execution_status,
            "timestamp": time.time(),
        }

    def get_stage_delta(self) -> dict:
        """
        Calculates delta token and audio usage since the last stage deduction.
        """
        total_in = 0
        total_out = 0
        for usage in self._runs.values():
            total_in += usage.input_tokens
            total_out += usage.output_tokens

        delta_in = max(0, total_in - self._last_deducted_input_tokens)
        delta_out = max(0, total_out - self._last_deducted_output_tokens)
        delta_audio_in = max(0.0, self.total_audio_in_sec - self._last_deducted_audio_in)
        delta_audio_out = max(0.0, self.total_audio_out_sec - self._last_deducted_audio_out)

        return {
            "total_in": total_in,
            "total_out": total_out,
            "delta_in": delta_in,
            "delta_out": delta_out,
            "delta_audio_in": delta_audio_in,
            "delta_audio_out": delta_audio_out,
            "has_delta": (delta_in > 0 or delta_out > 0 or delta_audio_in > 0 or delta_audio_out > 0),
        }

    async def deduct_stage_usage_with_express(self, stage_name: str = "stage") -> Optional[dict]:
        """
        Deducts credit usage for the incremental delta tokens/audio incurred since the last stage.
        Calls Express `POST /api/ai-billing/internal/report-usage`.
        If the wallet balance is exhausted (<= 0), sets self.is_exhausted = True and returns
        isExhausted: True so the caller can immediately halt further orchestration.
        """
        delta = self.get_stage_delta()
        if not delta["has_delta"]:
            return {"isExhausted": self.is_exhausted, "creditsCharged": 0}

        self._stage_count += 1
        current_stage = f"{stage_name}_{self._stage_count}"

        # Resolve primary model & provider from latest run
        primary_provider = "gemini"
        primary_model = "gemini-2.5-flash"
        if self._runs:
            latest_run = list(self._runs.values())[-1]
            primary_provider = latest_run.provider
            primary_model = latest_run.model

        payload = {
            "executionId": self.execution_id,
            "userId": self.user_id,
            "storeId": self.store_id,
            "feature": self.feature,
            "provider": primary_provider,
            "model": primary_model,
            "inputTokens": delta["delta_in"],
            "outputTokens": delta["delta_out"],
            "audioInputUnits": round(delta["delta_audio_in"], 2),
            "audioOutputUnits": round(delta["delta_audio_out"], 2),
            "executionStatus": "in_progress",
            "metadata": {
                "stageName": current_stage,
                "isStageDeduction": True,
                "cumulativeInputTokens": delta["total_in"],
                "cumulativeOutputTokens": delta["total_out"],
            },
        }

        settings = get_settings()
        endpoint = f"{settings.express_backend_url.rstrip('/')}/api/ai-billing/internal/report-usage"
        headers = {
            "Content-Type": "application/json",
            "x-internal-service-key": settings.internal_service_key,
            "x-service-name": "vyapar-ai-service",
        }

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(endpoint, json=payload, headers=headers)
                if resp.status_code == 200:
                    data = resp.json().get("data", {})
                    # Advance deducted token watermarks
                    self._last_deducted_input_tokens = delta["total_in"]
                    self._last_deducted_output_tokens = delta["total_out"]
                    self._last_deducted_audio_in = self.total_audio_in_sec
                    self._last_deducted_audio_out = self.total_audio_out_sec

                    is_exhausted = bool(data.get("isExhausted"))
                    remaining_balance = data.get("remainingBalance", 0)
                    if is_exhausted or (remaining_balance is not None and remaining_balance <= 0):
                        self.is_exhausted = True

                    LOGGER.info(
                        "stage_usage_deducted_success",
                        execution_id=self.execution_id,
                        stage=current_stage,
                        credits_charged=data.get("creditsCharged"),
                        remaining_balance=remaining_balance,
                        is_exhausted=self.is_exhausted,
                    )
                    return {
                        "success": True,
                        "creditsCharged": data.get("creditsCharged", 0),
                        "remainingBalance": remaining_balance,
                        "isExhausted": self.is_exhausted,
                    }
                else:
                    LOGGER.warning(
                        "stage_usage_deduction_non_200",
                        status_code=resp.status_code,
                        body=resp.text[:300],
                    )
                    return {"success": False, "isExhausted": False, "error": resp.text}
        except Exception as exc:
            LOGGER.error("stage_usage_deduction_failed", error=str(exc))
            return {"success": False, "isExhausted": False, "error": str(exc)}


class BillingUsageCallback(AsyncCallbackHandler):
    """
    LangChain AsyncCallbackHandler that intercepts every LLM invocation across
    LangGraph nodes and passes usage metadata into the execution's UsageCollector.
    """

    def __init__(self, collector: UsageCollector):
        super().__init__()
        self.collector = collector

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        pass

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        try:
            metadata = kwargs.get("metadata") or {}
            self.collector.add_model_usage(
                run_id=str(run_id),
                output=response,
                metadata=metadata,
            )
        except Exception as exc:
            LOGGER.warning("billing_callback_error", error=str(exc))


async def report_usage_to_express(usage_summary: dict) -> dict:
    """
    Sends the normalized usage summary to Express private endpoint
    `POST /api/ai-billing/internal/report-usage`.
    Uses server-to-server internal key for authentication.
    """
    settings = get_settings()
    endpoint = f"{settings.express_backend_url.rstrip('/')}/api/ai-billing/internal/report-usage"

    headers = {
        "Content-Type": "application/json",
        "x-internal-service-key": settings.internal_service_key,
        "x-service-name": "vyapar-ai-service",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(endpoint, json=usage_summary, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                LOGGER.info(
                    "usage_reported_to_express_success",
                    execution_id=usage_summary.get("executionId"),
                    credits_charged=data.get("data", {}).get("creditsCharged"),
                )
                return data
            else:
                LOGGER.warning(
                    "usage_reported_to_express_non_200",
                    status_code=resp.status_code,
                    body=resp.text[:300],
                )
                return {"error": resp.text, "statusCode": resp.status_code}
    except Exception as exc:
        LOGGER.error("usage_reporting_failed", endpoint=endpoint, error=str(exc))
        return {"error": str(exc), "status": "failed"}


async def reserve_credits_with_express(
    execution_id: str,
    store_id: str,
    credits: int = 50,
    feature_name: str = "voice",
    user_id: str = "default_user",
) -> dict:
    """
    Calls Express internal endpoint `POST /api/ai-billing/internal/reserve`
    to reserve credits before starting a voice/agent session.
    """
    settings = get_settings()
    endpoint = f"{settings.express_backend_url.rstrip('/')}/api/ai-billing/internal/reserve"
    headers = {
        "Content-Type": "application/json",
        "x-internal-service-key": settings.internal_service_key,
        "x-service-name": "vyapar-ai-service",
    }
    payload = {
        "executionId": execution_id,
        "storeId": store_id,
        "credits": credits,
        "featureName": feature_name,
        "metadata": {"userId": user_id, "channel": "websocket"},
    }
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.post(endpoint, json=payload, headers=headers)
            if resp.status_code == 200:
                return {"success": True, "data": resp.json()}
            return {"success": False, "status_code": resp.status_code, "error": resp.text}
    except Exception as exc:
        LOGGER.error("express_reservation_request_failed", error=str(exc))
        return {"success": False, "error": str(exc)}


async def release_reservation_with_express(execution_id: str, reason: str = "") -> dict:
    """
    Calls Express internal endpoint `POST /api/ai-billing/internal/release`
    to release unused reserved credits.
    """
    settings = get_settings()
    endpoint = f"{settings.express_backend_url.rstrip('/')}/api/ai-billing/internal/release"
    headers = {
        "Content-Type": "application/json",
        "x-internal-service-key": settings.internal_service_key,
        "x-service-name": "vyapar-ai-service",
    }
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.post(
                endpoint,
                json={"executionId": execution_id, "reason": reason},
                headers=headers,
            )
            return {"success": resp.status_code == 200}
    except Exception as exc:
        LOGGER.warning("express_release_request_failed", error=str(exc))
        return {"success": False, "error": str(exc)}


async def check_balance_with_express(
    store_id: str = "",
    user_id: str = "default_user",
    owner_id: str = "",
) -> dict:
    """
    Calls Express internal endpoint `POST /api/ai-billing/internal/check-balance`
    to check if user/owner has credits available before starting.
    """
    settings = get_settings()
    endpoint = f"{settings.express_backend_url.rstrip('/')}/api/ai-billing/internal/check-balance"
    headers = {
        "Content-Type": "application/json",
        "x-internal-service-key": settings.internal_service_key,
        "x-service-name": "vyapar-ai-service",
    }
    payload = {
        "storeId": store_id,
        "userId": user_id,
        "ownerId": owner_id,
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(endpoint, json=payload, headers=headers)
            if resp.status_code == 200:
                body = resp.json()
                data = body.get("data", {})
                return {
                    "allowed": bool(data.get("allowed")),
                    "balance": data.get("balance", 0),
                    "reason": data.get("reason"),
                }
            return {"allowed": False, "reason": "CHECK_FAILED", "error": resp.text}
    except Exception as exc:
        LOGGER.error("express_balance_check_failed", error=str(exc))
        return {"allowed": False, "reason": "NETWORK_ERROR", "error": str(exc)}
