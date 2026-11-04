"""
app/lib/llm.py
==============
Multi-Tier LLM factory with role-based model grouping, automatic fallback,
and multi-key rotation to eliminate rate limit bottlenecks across LangGraph nodes.

Architecture & Tier Strategy
----------------------------
1. Reasoning / Strategic Tier (`get_reasoning_llm` / `get_llm` / `get_planner_llm`):
   - Used by: `think_node`, `planner_node`, and strategic synthesis.
   - Primary: Gemini Flash (`gemini-2.5-flash`, `gemini-2.5-flash-lite`) across all configured keys.
   - Resilient Fallback: Groq 120B (`openai/gpt-oss-120b`), NVIDIA Llama 90B/11B, OpenRouter, Ollama.
   - If Gemini hits 429/TPM limits, switches automatically to Groq 120B / NVIDIA so execution never breaks.

2. Intent Routing Tier (`get_intent_llm`):
   - Used by: `intent_node`.
   - Primary: Gemini Flash Lite (`gemini-2.5-flash-lite`, low-token footprint) or Gemini Flash.
   - Fast Fallback: Groq 20B (`openai/gpt-oss-20b`), Groq Qwen 27B, NVIDIA 11B, OpenRouter.

3. Evaluation & Critic Tier (`get_critic_llm` / `get_evaluation_llm`):
   - Used by: `critic_node`, `reflection_node`.
   - Non-Gemini Primary: Groq 120B (`openai/gpt-oss-120b`), Groq 20B (`openai/gpt-oss-20b`),
     Groq Qwen 27B, NVIDIA 11B, OpenRouter, Ollama.
   - Tail Emergency Fallback ONLY: Gemini Flash Lite (only invoked if all non-Gemini providers fail).

4. Fast Auxiliary & Memory Tier (`get_small_llm` / `get_fast_llm` / `get_memory_llm`):
   - Used by: Memory extraction, retrieval reranking, conflict resolution, summarization, subgraphs.
   - Non-Gemini Primary: Groq 20B (`openai/gpt-oss-20b`), Groq Qwen 27B, NVIDIA 11B, OpenRouter, Ollama.
   - Tail Emergency Fallback ONLY: Gemini Flash Lite.

5. Guardrail & Safeguard Tier (`get_safeguard_llm`):
   - Used by: `grader_node` (safety & harm screening).
   - Dedicated Non-Gemini Primary: Groq Safeguard 20B (`openai/gpt-oss-safeguard-20b`), Groq 20B, NVIDIA Guard.
   - Tail Emergency Fallback ONLY: Gemini Flash Lite.
"""

from __future__ import annotations

import os
import time
from functools import partial
from typing import Any, Callable

import structlog

from app.config.settings import get_settings
from app.lib.gemini_keys import get_provider_keys

LOGGER = structlog.get_logger("vyaparsathi.ai.llm")


def _build_gemini(model: str, *, api_key: str | None = None, **kwargs) -> Any:
    """Build a ChatGoogleGenerativeAI instance for the given model."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = get_settings()
    key = api_key or settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=key,
        temperature=0.3,
        max_retries=0,
    )


def _build_openai(model: str, *, base_url: str, api_key: str, default_headers: dict | None = None) -> Any:
    """Build a ChatOpenAI instance pointed at an OpenAI-compatible endpoint."""
    from langchain_openai import ChatOpenAI

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    if default_headers:
        headers.update(default_headers)

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.3,
        max_retries=0,
        default_headers=headers,
    )


def _build_ollama(model: str, *, base_url: str = "http://localhost:11434/v1", api_key: str = "ollama") -> Any:
    """Build a ChatOpenAI instance targeting a local Ollama server."""
    return _build_openai(model, base_url=base_url, api_key=api_key or "ollama")


# ---------------------------------------------------------------------------
# Provider Definitions by Architectural Group
# ---------------------------------------------------------------------------

_OPENAI_BASE_URL = {
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

# 1. Reasoning & Strategic Thinking (think_node, synthesis)
_REASONING_PROVIDERS = [
    ("gemini", "gemini-2.5-flash", _build_gemini, "GEMINI"),
    ("gemini", "gemini-2.5-flash-lite", _build_gemini, "GEMINI"),
    ("groq", "openai/gpt-oss-120b", partial(_build_openai, base_url=_OPENAI_BASE_URL["groq"]), "GROQ"),
    ("groq", "qwen/qwen3.8-27b", partial(_build_openai, base_url=_OPENAI_BASE_URL["groq"]), "GROQ"),
    ("groq", "openai/gpt-oss-20b", partial(_build_openai, base_url=_OPENAI_BASE_URL["groq"]), "GROQ"),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct:free", partial(_build_openai, base_url=_OPENAI_BASE_URL["openrouter"]), "OPENROUTER"),
    ("ollama", "llama3", _build_ollama, "OLLAMA"),
]

# 1b. Planner Tier (planner_node) - Prioritizes large 120B reasoning models for generating plans
_PLANNER_PROVIDERS = [
    ("groq", "openai/gpt-oss-120b", partial(_build_openai, base_url=_OPENAI_BASE_URL["groq"]), "GROQ"),
    ("gemini", "gemini-2.5-flash", _build_gemini, "GEMINI"),
    ("gemini", "gemini-2.5-flash-lite", _build_gemini, "GEMINI"),
    ("groq", "qwen/qwen3.8-27b", partial(_build_openai, base_url=_OPENAI_BASE_URL["groq"]), "GROQ"),
    ("groq", "openai/gpt-oss-20b", partial(_build_openai, base_url=_OPENAI_BASE_URL["groq"]), "GROQ"),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct:free", partial(_build_openai, base_url=_OPENAI_BASE_URL["openrouter"]), "OPENROUTER"),
    ("ollama", "llama3", _build_ollama, "OLLAMA"),
]

# Legacy alias for backward compatibility
_LARGE_PROVIDERS = _REASONING_PROVIDERS

# 2. Intent Routing (intent_node)
_INTENT_PROVIDERS = [
    ("gemini", "gemini-2.5-flash-lite", _build_gemini, "GEMINI"),
    ("gemini", "gemini-2.5-flash", _build_gemini, "GEMINI"),
    ("groq", "openai/gpt-oss-20b", partial(_build_openai, base_url=_OPENAI_BASE_URL["groq"]), "GROQ"),
    ("groq", "qwen/qwen3.8-27b", partial(_build_openai, base_url=_OPENAI_BASE_URL["groq"]), "GROQ"),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct:free", partial(_build_openai, base_url=_OPENAI_BASE_URL["openrouter"]), "OPENROUTER"),
    ("ollama", "llama3", _build_ollama, "OLLAMA"),
]

# 3. Critic & Reflection Evaluation (critic_node, reflection_node)
# Non-Gemini primary (Groq 120B / 20B / Qwen) to conserve Gemini rate limits
_CRITIC_PROVIDERS = [
    ("groq", "openai/gpt-oss-120b", partial(_build_openai, base_url=_OPENAI_BASE_URL["groq"]), "GROQ"),
    ("groq", "openai/gpt-oss-20b", partial(_build_openai, base_url=_OPENAI_BASE_URL["groq"]), "GROQ"),
    ("groq", "qwen/qwen3.8-27b", partial(_build_openai, base_url=_OPENAI_BASE_URL["groq"]), "GROQ"),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct:free", partial(_build_openai, base_url=_OPENAI_BASE_URL["openrouter"]), "OPENROUTER"),
    ("ollama", "llama3", _build_ollama, "OLLAMA"),
    ("gemini", "gemini-2.5-flash-lite", _build_gemini, "GEMINI"),  # Emergency tail fallback
]

# 4. Fast Auxiliary & Memory (memory extraction, retrieval rerank, conflict resolution, summarizer)
# Non-Gemini primary: Groq 20B (ultra fast), Groq Qwen 27B
_SMALL_PROVIDERS = [
    ("groq", "openai/gpt-oss-20b", partial(_build_openai, base_url=_OPENAI_BASE_URL["groq"]), "GROQ"),
    ("groq", "qwen/qwen3.8-27b", partial(_build_openai, base_url=_OPENAI_BASE_URL["groq"]), "GROQ"),
    ("openrouter", "google/gemma-2-9b-it:free", partial(_build_openai, base_url=_OPENAI_BASE_URL["openrouter"]), "OPENROUTER"),
    ("ollama", "llama3", _build_ollama, "OLLAMA"),
    ("gemini", "gemini-2.5-flash-lite", _build_gemini, "GEMINI"),  # Emergency tail fallback
]

# 5. Guardrail / Safeguard (grader_node)
# Non-Gemini primary: Groq Safeguard 20B, Groq 20B
_SAFEGUARD_PROVIDERS = [
    ("groq", "openai/gpt-oss-safeguard-20b", partial(_build_openai, base_url=_OPENAI_BASE_URL["groq"]), "GROQ"),
    ("groq", "openai/gpt-oss-20b", partial(_build_openai, base_url=_OPENAI_BASE_URL["groq"]), "GROQ"),
    ("openrouter", "google/gemma-2-9b-it:free", partial(_build_openai, base_url=_OPENAI_BASE_URL["openrouter"]), "OPENROUTER"),
    ("gemini", "gemini-2.5-flash-lite", _build_gemini, "GEMINI"),  # Emergency tail fallback
]


# ---------------------------------------------------------------------------
# Fallback Chain with Automatic Cooldown & Daily Quota Protection
# ---------------------------------------------------------------------------

_PROVIDER_COOLDOWNS: dict[str, float] = {}
_COOLDOWN_SECONDS = 45.0
_DAILY_COOLDOWN_SECONDS = 43200.0  # 12 hours for free-tier daily quota exhaustion

_DAILY_QUOTA_KEYWORDS = (
    "per day limit exceeded",
    "daily limit",
    "requests per day",
    "generate_content_free_tier_requests",
    "quota exceeded for metric",
    "limit: 20",
    "please retry in",
)

_RATE_LIMIT_KEYWORDS = (
    "429",
    "resource_exhausted",
    "quota",
    "rate_limit",
    "rate_limit_exceeded",
    "tpm",
    "tokens per minute",
    "too many requests",
    "overloaded",
    "413",
    "request too large",
    "503",
    "service unavailable",
)


def _is_on_cooldown(name: str, now: float) -> bool:
    """Check if a provider or its base model is on cooldown."""
    if _PROVIDER_COOLDOWNS.get(name, 0) > now:
        return True
    base_name = name.split("[")[0]
    if base_name != name and _PROVIDER_COOLDOWNS.get(base_name, 0) > now:
        return True
    return False


def _record_provider_error(name: str, exc: Exception) -> None:
    """Record rate limits and quota exhaustions with proper tiered cooldowns."""
    err_msg = str(exc).lower()
    now = time.time()
    base_name = name.split("[")[0]
    if any(k in err_msg for k in _DAILY_QUOTA_KEYWORDS):
        _PROVIDER_COOLDOWNS[name] = now + _DAILY_COOLDOWN_SECONDS
        _PROVIDER_COOLDOWNS[base_name] = now + _DAILY_COOLDOWN_SECONDS
        LOGGER.warning(
            "llm_provider_daily_exhaustion_detected",
            provider=name,
            cooldown_s=_DAILY_COOLDOWN_SECONDS,
            error=str(exc)[:150],
        )
    elif any(k in err_msg for k in _RATE_LIMIT_KEYWORDS):
        # If it's a 413 or tool schema limit, cool down only the tool-bound member, not base text model!
        _PROVIDER_COOLDOWNS[name] = now + _COOLDOWN_SECONDS
        if "413" not in err_msg and "request too large" not in err_msg:
            _PROVIDER_COOLDOWNS[base_name] = now + _COOLDOWN_SECONDS
        LOGGER.warning(
            "llm_provider_cooldown_set",
            provider=name,
            cooldown_s=_COOLDOWN_SECONDS,
            error=str(exc)[:150],
        )


def _get_valid_indices(names: list[str]) -> list[int]:
    """
    Get active indices not on cooldown. If all are on cooldown,
    reset ONLY short-term cooldowns to avoid hammering daily-exhausted keys.
    """
    now = time.time()
    valid = [idx for idx, name in enumerate(names) if not _is_on_cooldown(name, now)]
    if valid:
        return valid

    LOGGER.warning("llm_all_providers_on_cooldown_checking_resets", providers=names)
    # Reset only short-term cooldowns (< 10 minutes left). NEVER reset daily-exhausted keys!
    for name in names:
        cooldown_until = _PROVIDER_COOLDOWNS.get(name, 0)
        if 0 < cooldown_until - now <= 600.0:
            _PROVIDER_COOLDOWNS.pop(name, None)
        base = name.split("[")[0]
        base_cooldown = _PROVIDER_COOLDOWNS.get(base, 0)
        if 0 < base_cooldown - now <= 600.0:
            _PROVIDER_COOLDOWNS.pop(base, None)

    valid = [idx for idx, name in enumerate(names) if not _is_on_cooldown(name, now)]
    if valid:
        return valid

    # Absolute fallback: if everything is strictly exhausted, reset everything
    LOGGER.error("llm_all_providers_strictly_exhausted_emergency_reset", providers=names)
    for name in names:
        _PROVIDER_COOLDOWNS.pop(name, None)
        _PROVIDER_COOLDOWNS.pop(name.split("[")[0], None)
    return list(range(len(names)))


class _FallbackLLM:
    """
    A Runnable wrapper that tries each provider in order until one succeeds.

    Initializes members lazily: only the primary provider is constructed upfront.
    Fallback members are initialized on-demand if the primary fails.
    Rate-limited providers (429 / RESOURCE_EXHAUSTED / TPM limits) are put on cooldown
    to avoid burning latency on known-exhausted keys.
    """

    def __init__(self, members: list[Any], names: list[str], builders: list[Callable[[], Any]] | None = None):
        self._members = list(members)
        self._names = list(names)
        self._builders = builders or [lambda m=m: m for m in members]

    def _get_member(self, idx: int) -> Any:
        """Lazily initialize the member at index if not already instantiated."""
        if self._members[idx] is None:
            t0 = time.perf_counter()
            self._members[idx] = self._builders[idx]()
            LOGGER.debug(
                "llm_member_lazy_initialized",
                provider=self._names[idx],
                ms=round((time.perf_counter() - t0) * 1000, 1),
            )
        return self._members[idx]

    # -- LangChain Runnable interface ---------------------------------------

    def invoke(self, input, config=None, **kwargs):
        valid_indices = _get_valid_indices(self._names)

        last_error = None
        for idx in valid_indices:
            name = self._names[idx]
            try:
                member = self._get_member(idx)
                LOGGER.debug("llm_fallback_try", provider=name)
                return member.invoke(input, config=config, **kwargs)
            except Exception as exc:
                last_error = exc
                _record_provider_error(name, exc)
                LOGGER.warning(
                    "llm_fallback_member_failed",
                    provider=name,
                    error=str(exc)[:200],
                )

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"All LLM providers ({self._names}) failed to invoke.")

    async def ainvoke(self, input, config=None, **kwargs):
        valid_indices = _get_valid_indices(self._names)

        last_error = None
        for idx in valid_indices:
            name = self._names[idx]
            try:
                member = self._get_member(idx)
                LOGGER.debug("llm_fallback_try", provider=name)
                return await member.ainvoke(input, config=config, **kwargs)
            except Exception as exc:
                last_error = exc
                _record_provider_error(name, exc)
                if "event loop" in str(exc).lower() or "closed" in str(exc).lower():
                    try:
                        fresh_member = self._builders[idx]()
                        self._members[idx] = fresh_member
                        return await fresh_member.ainvoke(input, config=config, **kwargs)
                    except Exception as exc2:
                        last_error = exc2
                LOGGER.warning(
                    "llm_fallback_member_failed",
                    provider=name,
                    error=str(exc)[:200],
                )

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"All LLM providers ({self._names}) failed to invoke.")

    def stream(self, input, config=None, **kwargs):
        valid_indices = _get_valid_indices(self._names)

        last_error = None
        for idx in valid_indices:
            name = self._names[idx]
            try:
                member = self._get_member(idx)
                LOGGER.debug("llm_fallback_try_stream", provider=name)
                for chunk in member.stream(input, config=config, **kwargs):
                    yield chunk
                return
            except Exception as exc:
                last_error = exc
                _record_provider_error(name, exc)
                LOGGER.warning(
                    "llm_fallback_member_failed",
                    provider=name,
                    error=str(exc)[:200],
                )

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"All LLM providers ({self._names}) failed to stream.")

    async def astream(self, input, config=None, **kwargs):
        valid_indices = _get_valid_indices(self._names)

        last_error = None
        for idx in valid_indices:
            name = self._names[idx]
            try:
                member = self._get_member(idx)
                LOGGER.debug("llm_fallback_try_astream", provider=name)
                async for chunk in member.astream(input, config=config, **kwargs):
                    yield chunk
                return
            except Exception as exc:
                last_error = exc
                _record_provider_error(name, exc)
                if "event loop" in str(exc).lower() or "closed" in str(exc).lower():
                    try:
                        fresh_member = self._builders[idx]()
                        self._members[idx] = fresh_member
                        async for chunk in fresh_member.astream(input, config=config, **kwargs):
                            yield chunk
                        return
                    except Exception as exc2:
                        last_error = exc2
                LOGGER.warning(
                    "llm_fallback_member_failed",
                    provider=name,
                    error=str(exc)[:200],
                )

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"All LLM providers ({self._names}) failed to stream.")

    def bind_tools(self, tools, **kwargs):
        """Bind tools lazily to fallback members."""
        new_names = [f"{n}[tools]" for n in self._names]
        new_builders = []
        for b in self._builders:
            def _make_tool_bound_builder(builder=b):
                instance = builder()
                if hasattr(instance, "bind_tools"):
                    try:
                        return instance.bind_tools(tools, **kwargs)
                    except Exception as e:
                        LOGGER.warning("bind_tools_failed_on_member", error=str(e)[:150])
                        return instance
                return instance
            new_builders.append(_make_tool_bound_builder)

        new_members: list[Any] = [None] * len(new_builders)
        if self._members and self._members[0] is not None:
            try:
                new_members[0] = self._members[0].bind_tools(tools, **kwargs)
            except Exception:
                new_members[0] = new_builders[0]()
        elif new_builders:
            new_members[0] = new_builders[0]()
        return _FallbackLLM(new_members, new_names, new_builders)

    def with_structured_output(self, schema: Any, **kwargs) -> _FallbackLLM:
        """Bind structured output schema lazily to fallback members."""
        new_names = [f"{n}[struct]" for n in self._names]
        new_builders = []
        for b in self._builders:
            def _make_structured_builder(builder=b):
                instance = builder()
                if hasattr(instance, "with_structured_output"):
                    try:
                        return instance.with_structured_output(schema, **kwargs)
                    except Exception as e:
                        LOGGER.warning("with_structured_output_failed_on_member", error=str(e)[:150])
                        return instance
                return instance
            new_builders.append(_make_structured_builder)

        new_members: list[Any] = [None] * len(new_builders)
        if self._members and self._members[0] is not None:
            try:
                new_members[0] = self._members[0].with_structured_output(schema, **kwargs)
            except Exception:
                new_members[0] = new_builders[0]()
        elif new_builders:
            new_members[0] = new_builders[0]()
        return _FallbackLLM(new_members, new_names, new_builders)
        for b in self._builders:
            def _make_structured_builder(builder=b):
                instance = builder()
                if hasattr(instance, "with_structured_output"):
                    try:
                        return instance.with_structured_output(schema, **kwargs)
                    except Exception as e:
                        LOGGER.warning("with_structured_output_failed_on_member", error=str(e)[:150])
                        return instance
                return instance
            new_builders.append(_make_structured_builder)

        new_members: list[Any] = [None] * len(new_builders)
        if self._members and self._members[0] is not None:
            try:
                new_members[0] = self._members[0].with_structured_output(schema, **kwargs)
            except Exception:
                new_members[0] = new_builders[0]()
        elif new_builders:
            new_members[0] = new_builders[0]()
        return _FallbackLLM(new_members, new_names, new_builders)

    def with_config(self, config: dict | None = None, **kwargs) -> _FallbackLLM:
        """Apply Runnable config (e.g. callbacks=[]) lazily to fallback members."""
        new_names = list(self._names)
        new_builders = [
            (lambda b=b, c=config, kw=kwargs: b().with_config(c, **kw) if hasattr(b(), "with_config") else b())
            for b in self._builders
        ]
        new_members: list[Any] = [None] * len(new_builders)
        if self._members and self._members[0] is not None:
            m = self._members[0]
            new_members[0] = m.with_config(config, **kwargs) if hasattr(m, "with_config") else m
        elif new_builders:
            new_members[0] = new_builders[0]()
        return _FallbackLLM(new_members, new_names, new_builders)

    def __getattr__(self, item):
        member0 = self._get_member(0)
        return getattr(member0, item)


# ---------------------------------------------------------------------------
# Internal Builder Helper & Module Caches
# ---------------------------------------------------------------------------

_reasoning_llm_cache: _FallbackLLM | None = None
_intent_llm_cache: _FallbackLLM | None = None
_critic_llm_cache: _FallbackLLM | None = None
_small_llm_cache: _FallbackLLM | None = None
_safeguard_llm_cache: _FallbackLLM | None = None


def _build_chain(provider_list: list) -> _FallbackLLM | None:
    names: list[str] = []
    builders: list[Callable[[], Any]] = []

    for provider_name, model, build_fn, provider_prefix in provider_list:
        if provider_name == "ollama":
            settings = get_settings()
            base_url = settings.ollama_base_url or os.getenv("OLLAMA_BASE_URL")
            if not base_url:
                continue
            b_fn = partial(build_fn, model, base_url=base_url)
            builders.append(b_fn)
            names.append(f"ollama/{model}")
            continue

        keys = get_provider_keys(provider_prefix)
        if not keys:
            if provider_prefix == "GROQ" and not getattr(_build_chain, "_groq_warned", False):
                _build_chain._groq_warned = True
                LOGGER.warning(
                    "groq_api_key_missing_in_environment",
                    hint="Set GROQ_API_KEY in Render environment variables to enable GPT-120B and Qwen-27B fallback models!",
                )
            continue

        for idx, key in enumerate(keys, 1):
            if provider_name == "gemini":
                b_fn = partial(_build_gemini, model, api_key=key)
            else:
                b_fn = partial(build_fn, model, api_key=key)

            builders.append(b_fn)
            key_tag = f"key_{idx}"
            names.append(f"{provider_name}/{model} ({key_tag})")

    if not builders:
        return None

    # Lazy members: initialize only the primary member upfront to eliminate startup lag
    members: list[Any] = [None] * len(builders)
    try:
        t0 = time.perf_counter()
        members[0] = builders[0]()
        LOGGER.info("llm_primary_member_ready", provider=names[0], ms=round((time.perf_counter() - t0) * 1000, 1))
    except Exception as exc:
        LOGGER.warning("llm_primary_member_init_failed", provider=names[0], error=str(exc)[:200])

    return _FallbackLLM(members, names, builders)


# ---------------------------------------------------------------------------
# Public Group Factory Functions
# ---------------------------------------------------------------------------

def get_reasoning_llm() -> _FallbackLLM | None:
    """
    Return the fallback LLM chain for complex reasoning, strategic planning,
    and tool synthesis (`think_node`, `planner_node`).

    Prioritizes Gemini Flash models, falling back to Groq 120B, NVIDIA, OpenRouter, and Ollama.
    """
    global _reasoning_llm_cache
    if _reasoning_llm_cache is not None:
        return _reasoning_llm_cache
    chain = _build_chain(_REASONING_PROVIDERS)
    if chain is None:
        LOGGER.error("get_reasoning_llm_no_provider_configured")
        return None
    LOGGER.info("get_reasoning_llm_fallback_chain_built", members=chain._names)
    _reasoning_llm_cache = chain
    return _reasoning_llm_cache


# Backward compatible aliases for the reasoning tier
get_llm = get_reasoning_llm

_planner_llm_cache: _FallbackLLM | None = None


def get_planner_llm() -> _FallbackLLM | None:
    """
    Return the fallback LLM chain for complex planning (`planner_node`).
    Prioritizes large models like Groq 120B (`openai/gpt-oss-120b`) and Gemini Flash
    for fast, high-quality structured execution plans without rate limit failures.
    """
    global _planner_llm_cache
    if _planner_llm_cache is not None:
        return _planner_llm_cache
    chain = _build_chain(_PLANNER_PROVIDERS)
    if chain is None:
        return get_reasoning_llm()
    LOGGER.info("get_planner_llm_fallback_chain_built", members=chain._names)
    _planner_llm_cache = chain
    return _planner_llm_cache


def get_intent_llm() -> _FallbackLLM | None:
    """
    Return the fallback LLM chain for intent classification (`intent_node`).

    Prioritizes Gemini Flash Lite for fast and cost-effective intent routing,
    falling back immediately to Groq 20B / Qwen 27B / NVIDIA without downtime.
    """
    global _intent_llm_cache
    if _intent_llm_cache is not None:
        return _intent_llm_cache
    chain = _build_chain(_INTENT_PROVIDERS)
    if chain is None:
        LOGGER.error("get_intent_llm_no_provider_configured")
        return None
    LOGGER.info("get_intent_llm_fallback_chain_built", members=chain._names)
    _intent_llm_cache = chain
    return _intent_llm_cache


def get_critic_llm() -> _FallbackLLM | None:
    """
    Return the fallback LLM chain for evaluating and critiquing draft answers
    (`critic_node`, `reflection_node`).

    Prioritizes strong non-Gemini models (Groq 120B, Groq 20B, Groq Qwen 27B, NVIDIA)
    to protect Gemini rate limits while preserving high critique quality.
    Gemini is only consulted as an emergency tail fallback if all non-Gemini providers fail.
    """
    global _critic_llm_cache
    if _critic_llm_cache is not None:
        return _critic_llm_cache
    chain = _build_chain(_CRITIC_PROVIDERS)
    if chain is None:
        LOGGER.error("get_critic_llm_no_provider_configured")
        return None
    LOGGER.info("get_critic_llm_fallback_chain_built", members=chain._names)
    _critic_llm_cache = chain
    return _critic_llm_cache


# Backward compatible alias for evaluation
get_evaluation_llm = get_critic_llm


def get_small_llm() -> _FallbackLLM | None:
    """
    Return the fallback LLM chain prioritizing fast, lightweight non-Gemini models.

    Used by memory extraction, memory retrieval reranking, conflict resolution,
    conversation summarization, and light subgraphs.
    Prioritizes Groq 20B / Groq Qwen 27B / NVIDIA Llama 11B, with Gemini Flash Lite
    only as an emergency tail fallback.
    """
    global _small_llm_cache
    if _small_llm_cache is not None:
        return _small_llm_cache
    chain = _build_chain(_SMALL_PROVIDERS)
    if chain is None:
        LOGGER.error("get_small_llm_no_provider_configured")
        return None
    LOGGER.info("get_small_llm_fallback_chain_built", members=chain._names)
    _small_llm_cache = chain
    return _small_llm_cache


# Semantic aliases for the fast/small tier
get_fast_llm = get_small_llm
get_memory_llm = get_small_llm


def get_safeguard_llm() -> _FallbackLLM | None:
    """
    Return the fallback LLM chain for safety and harm screening (`grader_node`).

    Prioritizes Groq Safeguard 20B, Groq 20B, NVIDIA Guard, and OpenRouter,
    with Gemini Flash Lite only as an emergency tail fallback.
    """
    global _safeguard_llm_cache
    if _safeguard_llm_cache is not None:
        return _safeguard_llm_cache
    chain = _build_chain(_SAFEGUARD_PROVIDERS)
    if chain is None:
        LOGGER.error("get_safeguard_llm_no_provider_configured")
        return None
    LOGGER.info("get_safeguard_llm_fallback_chain_built", members=chain._names)
    _safeguard_llm_cache = chain
    return _safeguard_llm_cache


def clear_llm_cache():
    """Clear all cached LLM chain instances (useful for testing or key rotation)."""
    global _reasoning_llm_cache, _intent_llm_cache, _critic_llm_cache, _small_llm_cache, _safeguard_llm_cache
    _reasoning_llm_cache = None
    _intent_llm_cache = None
    _critic_llm_cache = None
    _small_llm_cache = None
    _safeguard_llm_cache = None
    LOGGER.info("llm_cache_cleared")


def get_llm_status() -> dict:
    """Return diagnostic info about which providers and keys are available across groups."""
    groups = {
        "reasoning": _REASONING_PROVIDERS,
        "intent": _INTENT_PROVIDERS,
        "critic": _CRITIC_PROVIDERS,
        "small": _SMALL_PROVIDERS,
        "safeguard": _SAFEGUARD_PROVIDERS,
    }
    status: dict[str, Any] = {"groups": {}}
    all_available = []

    for group_name, providers in groups.items():
        prov_status = []
        for provider_name, model, _build_fn, provider_prefix in providers:
            keys = get_provider_keys(provider_prefix) if provider_name != "ollama" else [os.getenv("OLLAMA_BASE_URL", "")]
            keys = [k for k in keys if k]
            info = {
                "provider": provider_name,
                "model": model,
                "configured": bool(keys),
                "key_count": len(keys),
            }
            prov_status.append(info)
            if group_name == "reasoning":
                all_available.append(info)

        status["groups"][group_name] = {
            "providers": prov_status,
            "active_count": sum(1 for p in prov_status if p["configured"]),
        }

    # Backward compatibility with top-level fields
    status["providers"] = all_available
    status["active_count"] = sum(1 for p in all_available if p["configured"])
    return status


# ---------------------------------------------------------------------------
# Cloudflare Workers AI Factory Re-exports
# ---------------------------------------------------------------------------
from app.lib.cloudflare_ai import (
    getChatModel,
    get_chat_model,
    is_cloudflare_configured,
    validate_cloudflare_config,
    estimate_message_tokens,
)
