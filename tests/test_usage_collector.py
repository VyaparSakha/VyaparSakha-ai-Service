"""
tests/test_usage_collector.py
=============================
Tests for AI model usage tracking, deduplication by run_id,
multi-model breakdown aggregation, audio duration metering,
and LangChain callback integration.
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult, Generation

from app.services.usage_collector import (
    UsageCollector,
    NormalizedModelUsage,
    BillingUsageCallback,
)


def test_usage_collector_extracts_from_usage_metadata():
    collector = UsageCollector(
        execution_id="exec_test_001",
        user_id="user_123",
        store_id="store_abc",
        feature="copilot",
    )

    msg = AIMessage(
        content="Here are your sales trends.",
        usage_metadata={
            "input_tokens": 350,
            "output_tokens": 150,
            "total_tokens": 500,
        },
    )

    usage = collector.add_model_usage(
        run_id="run_node_think_1",
        output=msg,
        metadata={"ls_provider": "groq", "ls_model_name": "llama-3.3-70b-versatile"},
    )

    assert usage is not None
    assert usage.input_tokens == 350
    assert usage.output_tokens == 150
    assert usage.total_tokens == 500
    assert usage.provider == "groq"
    assert usage.model == "llama-3.3-70b-versatile"
    assert usage.provider_usage_status == "verified"


def test_usage_collector_deduplicates_by_run_id():
    collector = UsageCollector(
        execution_id="exec_test_002",
        user_id="user_123",
        store_id="store_abc",
        feature="copilot",
    )

    msg = AIMessage(
        content="Test content",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        },
    )

    # First recording succeeds
    u1 = collector.add_model_usage(run_id="duplicate_run_id", output=msg)
    assert u1 is not None

    # Second recording with identical run_id is ignored (prevents double counting)
    u2 = collector.add_model_usage(run_id="duplicate_run_id", output=msg)
    assert u2 is None

    summary = collector.get_summary()
    assert summary["inputTokens"] == 100
    assert summary["outputTokens"] == 50
    assert summary["totalTokens"] == 150


def test_usage_collector_fallback_to_response_metadata():
    collector = UsageCollector(
        execution_id="exec_test_003",
        user_id="user_123",
        store_id="store_abc",
        feature="copilot",
    )

    # AIMessage with token usage inside response_metadata
    msg = AIMessage(
        content="Response using provider metadata",
        response_metadata={
            "token_usage": {
                "prompt_tokens": 420,
                "completion_tokens": 80,
                "total_tokens": 500,
            },
            "model_name": "gemini-2.5-flash",
        },
    )

    usage = collector.add_model_usage(
        run_id="run_response_meta",
        output=msg,
        provider="google",
    )

    assert usage is not None
    assert usage.input_tokens == 420
    assert usage.output_tokens == 80
    assert usage.total_tokens == 500
    assert usage.model == "gemini-2.5-flash"
    assert usage.provider == "google"


def test_usage_collector_multi_model_breakdown():
    collector = UsageCollector(
        execution_id="exec_multi_node_004",
        user_id="user_123",
        store_id="store_abc",
        feature="copilot",
    )

    # Node 1: Classifier using Groq LLaMA
    collector.add_model_usage(
        run_id="run_classify",
        output=AIMessage(content="classified", usage_metadata={"input_tokens": 200, "output_tokens": 50, "total_tokens": 250}),
        metadata={"ls_provider": "groq", "ls_model_name": "llama-3.3-70b-versatile"},
    )

    # Node 2: Think / Synthesis using Gemini
    collector.add_model_usage(
        run_id="run_synthesis",
        output=AIMessage(content="detailed answer", usage_metadata={"input_tokens": 800, "output_tokens": 400, "total_tokens": 1200}),
        metadata={"ls_provider": "google", "ls_model_name": "gemini-2.5-flash"},
    )

    summary = collector.get_summary(execution_status="completed")

    assert summary["inputTokens"] == 1000
    assert summary["outputTokens"] == 450
    assert summary["totalTokens"] == 1450
    assert summary["executionStatus"] == "completed"

    breakdown = summary["modelBreakdown"]
    assert len(breakdown) == 2

    groq_item = next(b for b in breakdown if b["provider"] == "groq")
    assert groq_item["inputTokens"] == 200
    assert groq_item["outputTokens"] == 50
    assert groq_item["callCount"] == 1

    gemini_item = next(b for b in breakdown if b["provider"] == "google")
    assert gemini_item["inputTokens"] == 800
    assert gemini_item["outputTokens"] == 400
    assert gemini_item["callCount"] == 1


def test_usage_collector_voice_duration():
    collector = UsageCollector(
        execution_id="voice_live_005",
        user_id="user_123",
        store_id="store_abc",
        feature="voice",
    )

    # Simulate 3 audio chunks: 5.5s in, 8.2s out
    collector.add_voice_usage(audio_in_seconds=2.5, audio_out_seconds=0.0)
    collector.add_voice_usage(audio_in_seconds=3.0, audio_out_seconds=4.1)
    collector.add_voice_usage(audio_in_seconds=0.0, audio_out_seconds=4.1)

    summary = collector.get_summary()

    assert summary["audioInputUnits"] == 5.5
    assert summary["audioOutputUnits"] == 8.2
    assert summary["feature"] == "voice"
    assert summary["creditsConsumed"] > 0


@pytest.mark.asyncio
async def test_billing_usage_callback_intercepts_llm_end():
    collector = UsageCollector(
        execution_id="exec_callback_006",
        user_id="user_123",
        store_id="store_abc",
        feature="copilot",
    )
    callback = BillingUsageCallback(collector)

    llm_result = LLMResult(
        generations=[[Generation(text="Hello world")]],
        llm_output={
            "token_usage": {
                "prompt_tokens": 150,
                "completion_tokens": 75,
                "total_tokens": 225,
            },
            "model_name": "meta-llama/llama-3.3-70b-instruct",
        },
    )

    await callback.on_llm_end(
        response=llm_result,
        run_id="run_cb_test_uuid",
        metadata={"ls_provider": "groq"},
    )

    summary = collector.get_summary()
    assert summary["inputTokens"] == 150
    assert summary["outputTokens"] == 75
    assert summary["totalTokens"] == 225


@pytest.mark.asyncio
async def test_llm_result_with_chat_generation_google_genai():
    from langchain_core.outputs.chat_generation import ChatGeneration
    collector = UsageCollector(
        execution_id="exec_gemini_007",
        user_id="user_123",
        store_id="store_abc",
        feature="copilot",
    )
    callback = BillingUsageCallback(collector)

    msg = AIMessage(
        content="Here is your report",
        response_metadata={"model_name": "gemini-2.5-flash", "model_provider": "google_genai"},
        usage_metadata={"input_tokens": 1250, "output_tokens": 320, "total_tokens": 1570},
    )
    gen = ChatGeneration(text="Here is your report", message=msg)
    llm_result = LLMResult(generations=[[gen]], llm_output={})

    await callback.on_llm_end(
        response=llm_result,
        run_id="run_gemini_node",
    )

    summary = collector.get_summary()
    assert summary["inputTokens"] == 1250
    assert summary["outputTokens"] == 320
    assert summary["totalTokens"] == 1570
    assert summary["model"] == "gemini-2.5-flash"
    assert summary["provider"] == "gemini"
    assert summary["creditsConsumed"] > 0
    assert summary["modelBreakdown"][0]["providerUsageStatus"] == "verified"


def test_usage_collector_stage_delta_computation():
    collector = UsageCollector(
        execution_id="exec_stage_008",
        user_id="user_123",
        store_id="store_abc",
        feature="copilot",
    )

    # Initial delta is empty
    d0 = collector.get_stage_delta()
    assert d0["has_delta"] is False
    assert d0["delta_in"] == 0
    assert d0["delta_out"] == 0

    # Add stage 1 model usage
    collector.add_model_usage(
        run_id="run_stage_1",
        output=AIMessage(content="Stage 1", usage_metadata={"input_tokens": 500, "output_tokens": 150, "total_tokens": 650}),
    )

    d1 = collector.get_stage_delta()
    assert d1["has_delta"] is True
    assert d1["delta_in"] == 500
    assert d1["delta_out"] == 150

    # Simulate advancing watermark after stage 1 deduction
    collector._last_deducted_input_tokens = d1["total_in"]
    collector._last_deducted_output_tokens = d1["total_out"]

    # Now delta is zero until next stage
    d1_after = collector.get_stage_delta()
    assert d1_after["has_delta"] is False
    assert d1_after["delta_in"] == 0
    assert d1_after["delta_out"] == 0

    # Add stage 2 model usage
    collector.add_model_usage(
        run_id="run_stage_2",
        output=AIMessage(content="Stage 2", usage_metadata={"input_tokens": 300, "output_tokens": 100, "total_tokens": 400}),
    )

    d2 = collector.get_stage_delta()
    assert d2["has_delta"] is True
    assert d2["delta_in"] == 300
    assert d2["delta_out"] == 100
    assert d2["total_in"] == 800
    assert d2["total_out"] == 250


@pytest.mark.asyncio
async def test_deduct_stage_usage_with_express_detects_exhaustion():
    from unittest.mock import patch, AsyncMock
    collector = UsageCollector(
        execution_id="exec_stage_009",
        user_id="user_123",
        store_id="store_abc",
        feature="copilot",
    )

    collector.add_model_usage(
        run_id="run_stage_1",
        output=AIMessage(content="Stage 1", usage_metadata={"input_tokens": 500, "output_tokens": 100, "total_tokens": 600}),
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "success": True,
        "data": {
            "creditsCharged": 1,
            "remainingBalance": 0,
            "isExhausted": True,
        },
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        res = await collector.deduct_stage_usage_with_express(stage_name="think")

        assert res is not None
        assert res["success"] is True
        assert res["isExhausted"] is True
        assert res["remainingBalance"] == 0
        assert collector.is_exhausted is True
        assert collector._last_deducted_input_tokens == 500


@pytest.mark.asyncio
async def test_check_balance_with_express():
    from unittest.mock import patch, AsyncMock
    from app.services.usage_collector import check_balance_with_express

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "success": True,
        "data": {
            "allowed": True,
            "balance": 20,
            "reason": None,
        },
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        check = await check_balance_with_express(store_id="store_123", user_id="user_456")

        assert check["allowed"] is True
        assert check["balance"] == 20
        assert check["reason"] is None


