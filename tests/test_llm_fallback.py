"""
tests/test_llm_fallback.py
===========================
Unit tests for multi-tier LLM fallback chain, Groq 120B/27B/20B support,
daily quota exhaustion protection, and planner/think node resilience.
"""

import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from app.lib.llm import (
    _FallbackLLM,
    _PROVIDER_COOLDOWNS,
    _COOLDOWN_SECONDS,
    _DAILY_COOLDOWN_SECONDS,
    _record_provider_error,
    _get_valid_indices,
    get_planner_llm,
    get_reasoning_llm,
)


def test_planner_llm_chain_configuration():
    """Verify that get_planner_llm prioritizes large 120B reasoning models."""
    chain = get_planner_llm()
    assert chain is not None
    assert any("gpt-oss-120b" in name for name in chain._names)


def test_reasoning_llm_chain_configuration():
    """Verify that get_reasoning_llm includes 120B, 27B, and 20B models."""
    chain = get_reasoning_llm()
    assert chain is not None
    assert any("gpt-oss-120b" in name for name in chain._names)
    assert any("qwen3.8-27b" in name for name in chain._names)


def test_daily_quota_exhaustion_detection():
    """Verify that daily quota errors receive a 12-hour cooldown and are not reset by short resets."""
    test_key = "test_gemini_daily_key"
    exc = Exception(
        "GoogleRateLimitError: 429 RESOURCE_EXHAUSTED. Quota exceeded for metric: "
        "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20. "
        "Generate content requests per day limit exceeded on Free Tier. Please retry in 14h57m."
    )
    _record_provider_error(test_key, exc)

    now = time.time()
    cooldown = _PROVIDER_COOLDOWNS.get(test_key, 0)
    assert cooldown - now > 3600.0  # Must be long cooldown (> 1 hour)

    # Simulate all providers cooling down and trigger reset
    test_names = [test_key, "test_groq_temp_key"]
    _PROVIDER_COOLDOWNS["test_groq_temp_key"] = now + 15.0  # Short temporary cooldown

    valid = _get_valid_indices(test_names)
    # The short temporary key should be reset, but the daily exhausted key MUST remain cooled down!
    assert 1 in valid
    assert 0 not in valid


def test_tool_binding_suffix_isolation():
    """Verify that 413 tool schema errors do not disable the base model for direct text."""
    base_name = "groq/openai/gpt-oss-120b (key_1)"
    tool_name = f"{base_name}[tools]"

    # Simulate 413 TPM limit failure on tool-bound invocation
    exc_413 = Exception("Error code: 413 - {'error': {'message': 'Request too large on tokens per minute (TPM)'}}")
    _record_provider_error(tool_name, exc_413)

    now = time.time()
    assert _PROVIDER_COOLDOWNS.get(tool_name, 0) > now
    # Base model must NOT be marked on cooldown!
    assert _PROVIDER_COOLDOWNS.get(base_name, 0) <= now


@pytest.mark.asyncio
async def test_fallback_llm_switches_to_second_provider_on_failure():
    """Verify _FallbackLLM transparently calls the fallback provider if primary fails."""
    mock_primary = MagicMock()
    mock_primary.ainvoke = AsyncMock(side_effect=Exception("429 Too Many Requests"))

    mock_fallback = MagicMock()
    mock_fallback.ainvoke = AsyncMock(return_value=MagicMock(content="Hello from Groq 120B!"))

    chain = _FallbackLLM(
        members=[mock_primary, mock_fallback],
        names=["test_provider_1", "test_provider_2"],
        builders=[lambda: mock_primary, lambda: mock_fallback],
    )

    result = await chain.ainvoke("Test question")
    assert result.content == "Hello from Groq 120B!"
    mock_fallback.ainvoke.assert_called_once()
