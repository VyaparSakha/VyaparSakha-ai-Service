"""
app/lib/cloudflare_ai.py
========================
Model factory and LangGraph/LangChain integration for Cloudflare Workers AI.
Supports:
  - GLM:   `@cf/zai-org/glm-4.7-flash`
  - Gemma: `@cf/google/gemma-4-26b-a4b-it`

Uses Cloudflare's OpenAI-compatible chat completions endpoint via ChatOpenAI:
  Base URL: https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1
  API Key:  CLOUDFLARE_API_TOKEN

Features:
  - Validates credentials and model configurations cleanly.
  - Safe defaults: temperature=0.4, max_tokens=2000, timeout=60s, streaming=True.
  - Automatic single-retry fallback between GLM and Gemma on transient capacity/timeout/availability errors.
  - Strict guard: no fallback for authentication (401/403) or client validation errors.
  - Token estimation and context management for 10,000–20,000 token contexts.
  - Tool preservation: tool_calls and ToolMessages are fully preserved.
  - Compatible with LangGraph nodes (planner_node, think_node) and streaming.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, AsyncIterator, Callable, Iterator, List, Optional, Sequence, Union

import structlog
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI

LOGGER = structlog.get_logger("vyaparsathi.ai.cloudflare")

# ---------------------------------------------------------------------------
# Cloudflare Model Constants
# ---------------------------------------------------------------------------

MODEL_GLM = "@cf/zai-org/glm-4.7-flash"
MODEL_GEMMA = "@cf/google/gemma-4-26b-a4b-it"

VALID_MODEL_IDENTIFIERS = {
    "glm": MODEL_GLM,
    MODEL_GLM: MODEL_GLM,
    "gemma": MODEL_GEMMA,
    MODEL_GEMMA: MODEL_GEMMA,
}

DEFAULT_MODEL = "glm"
DEFAULT_TEMPERATURE = 0.4
DEFAULT_MAX_TOKENS = 2000
DEFAULT_REQUEST_TIMEOUT = 60.0
MAX_CONTEXT_TOKENS = 20000
HIGH_CONTEXT_THRESHOLD = 12000


# ---------------------------------------------------------------------------
# Configuration & Validation
# ---------------------------------------------------------------------------

def resolve_model_name(model_name: Optional[str] = None) -> tuple[str, str]:
    """
    Resolve model shorthand ('glm' or 'gemma') to (shorthand, full_model_id).

    Raises:
        ValueError: If model is not 'glm' or 'gemma'.
    """
    raw = (model_name or os.environ.get("CLOUDFLARE_AI_MODEL") or DEFAULT_MODEL).strip()
    if not raw:
        raw = DEFAULT_MODEL
    key = raw.lower()

    if key in ("glm", MODEL_GLM.lower()):
        return "glm", MODEL_GLM
    elif key in ("gemma", MODEL_GEMMA.lower()):
        return "gemma", MODEL_GEMMA
    else:
        raise ValueError(
            f"Invalid model '{raw}'. Valid options are: 'glm' ({MODEL_GLM}) or 'gemma' ({MODEL_GEMMA})."
        )


def get_alternate_model(current_model_id: str) -> str:
    """Return the fallback counterpart model."""
    if current_model_id == MODEL_GLM:
        return MODEL_GEMMA
    return MODEL_GLM


def is_cloudflare_configured() -> bool:
    """Check if Cloudflare credentials are configured in environment or settings."""
    settings = None
    try:
        from app.config.settings import get_settings
        settings = get_settings()
    except Exception:
        pass

    aid = (
        os.environ.get("CLOUDFLARE_AI_ACCOUNT_ID")
        or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        or getattr(settings, "effective_cloudflare_ai_account_id", None)
        or getattr(settings, "cloudflare_account_id", None)
    )
    tok = (
        os.environ.get("CLOUDFLARE_AI_API_TOKEN")
        or os.environ.get("CLOUDFLARE_API_TOKEN")
        or getattr(settings, "effective_cloudflare_ai_api_token", None)
        or getattr(settings, "cloudflare_api_token", None)
    )
    return bool(aid and str(aid).strip() and tok and str(tok).strip())


def validate_cloudflare_config(
    account_id: Optional[str] = None,
    api_token: Optional[str] = None,
    model_name: Optional[str] = None,
) -> dict[str, Any]:
    """
    Validate that required Cloudflare credentials and model configuration exist.

    Supports CLOUDFLARE_AI_ACCOUNT_ID (dedicated for Workers AI) as well as CLOUDFLARE_ACCOUNT_ID.
    Supports CLOUDFLARE_AI_API_TOKEN as well as CLOUDFLARE_API_TOKEN.

    Raises:
        ValueError: If CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN is missing or model is invalid.
    """
    settings = None
    try:
        from app.config.settings import get_settings
        settings = get_settings()
    except Exception:
        pass

    if account_id is not None:
        aid = str(account_id).strip()
    elif "CLOUDFLARE_AI_ACCOUNT_ID" in os.environ and os.environ.get("CLOUDFLARE_AI_ACCOUNT_ID", "").strip():
        aid = os.environ.get("CLOUDFLARE_AI_ACCOUNT_ID", "").strip()
    elif "CLOUDFLARE_ACCOUNT_ID" in os.environ:
        aid = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    else:
        aid = (
            getattr(settings, "effective_cloudflare_ai_account_id", None)
            or getattr(settings, "cloudflare_account_id", None)
            or ""
        ).strip()

    if not aid:
        raise ValueError("CLOUDFLARE_ACCOUNT_ID is required but missing (set CLOUDFLARE_AI_ACCOUNT_ID or CLOUDFLARE_ACCOUNT_ID).")

    if api_token is not None:
        token = str(api_token).strip()
    elif "CLOUDFLARE_AI_API_TOKEN" in os.environ and os.environ.get("CLOUDFLARE_AI_API_TOKEN", "").strip():
        token = os.environ.get("CLOUDFLARE_AI_API_TOKEN", "").strip()
    elif "CLOUDFLARE_API_TOKEN" in os.environ:
        token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    else:
        token = (
            getattr(settings, "effective_cloudflare_ai_api_token", None)
            or getattr(settings, "cloudflare_api_token", None)
            or ""
        ).strip()

    if not token:
        raise ValueError("CLOUDFLARE_API_TOKEN is required but missing (set CLOUDFLARE_AI_API_TOKEN or CLOUDFLARE_API_TOKEN).")

    model_key, full_model_id = resolve_model_name(model_name)

    base_url = f"https://api.cloudflare.com/client/v4/accounts/{str(aid).strip()}/ai/v1"

    temp_raw = os.environ.get("AI_TEMPERATURE")
    try:
        temp = float(temp_raw) if temp_raw is not None else DEFAULT_TEMPERATURE
    except (ValueError, TypeError):
        temp = DEFAULT_TEMPERATURE

    tokens_raw = os.environ.get("AI_MAX_TOKENS")
    try:
        max_tokens = int(tokens_raw) if tokens_raw is not None else DEFAULT_MAX_TOKENS
    except (ValueError, TypeError):
        max_tokens = DEFAULT_MAX_TOKENS

    fb_raw = os.environ.get("AI_ENABLE_FALLBACK", "true")
    enable_fallback = str(fb_raw).strip().lower() in ("true", "1", "yes")

    return {
        "account_id": str(aid).strip(),
        "api_token": str(token).strip(),
        "model_key": model_key,
        "model": full_model_id,
        "base_url": base_url,
        "temperature": temp,
        "max_tokens": max_tokens,
        "enable_fallback": enable_fallback,
        "timeout": DEFAULT_REQUEST_TIMEOUT,
    }


def validate_messages(messages: Any) -> list[BaseMessage]:
    """
    Validate that the messages input is valid:
    - Rejects empty messages lists.
    - Rejects empty user messages (HumanMessage with whitespace or empty content).

    Raises:
        ValueError: If validation fails.
    """
    if not messages:
        raise ValueError("Messages list cannot be empty.")

    if not isinstance(messages, list):
        msg_list = [messages]
    else:
        msg_list = list(messages)

    for msg in msg_list:
        is_human = isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human"
        if is_human:
            content = getattr(msg, "content", "")
            if isinstance(content, str) and not content.strip():
                raise ValueError("User message cannot be empty.")
            elif isinstance(content, list) and len(content) == 0:
                raise ValueError("User message cannot be empty.")

    return msg_list


# ---------------------------------------------------------------------------
# Token Counting & Context Management (10k–20k Tokens)
# ---------------------------------------------------------------------------

def estimate_message_tokens(messages: Sequence[BaseMessage]) -> int:
    """
    Estimate token size for a list of messages.
    Accounts for message text, tool_calls, tool outputs, and protocol overhead.
    """
    total_chars = 0
    message_count = len(messages)

    for msg in messages:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total_chars += len(str(part.get("text", "")))
                else:
                    total_chars += len(str(part))

        # Tool calls on AIMessage
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            try:
                total_chars += len(json.dumps(tool_calls))
            except Exception:
                total_chars += len(str(tool_calls))

        # Name or role
        if getattr(msg, "name", None):
            total_chars += len(str(msg.name))

    # ~3.5 characters per token + 4 tokens overhead per message
    estimated = int(total_chars / 3.5) + (message_count * 4)
    return estimated


async def compress_context_if_needed(
    messages: list[BaseMessage],
    max_tokens: int = MAX_CONTEXT_TOKENS,
) -> list[BaseMessage]:
    """
    If the context exceeds the threshold (10k–20k tokens), summarize older messages
    using the project's existing summarization strategy while keeping system
    instructions and the latest user request intact.

    Raises:
        ValueError: If even after compression the context exceeds max_tokens.
    """
    tokens = estimate_message_tokens(messages)
    if tokens <= HIGH_CONTEXT_THRESHOLD:
        return messages

    LOGGER.info("cloudflare_compressing_large_context", estimated_tokens=tokens, threshold=HIGH_CONTEXT_THRESHOLD)

    # Separate system instructions, conversation history, and latest user prompt
    system_msgs = [m for m in messages if isinstance(m, SystemMessage) or getattr(m, "type", "") == "system"]
    non_system = [m for m in messages if m not in system_msgs]

    if len(non_system) <= 2:
        if tokens > max_tokens:
            raise ValueError(
                f"Input context size ({tokens} tokens) exceeds maximum limit of {max_tokens} tokens."
            )
        return messages

    # Keep the latest message and active tool turn intact
    latest_msg = non_system[-1]
    older_msgs = non_system[:-1]

    from app.lib.summarizer import summarize
    from app.agent.prompts.summarizer_prompts import SUMMARIZER_HISTORY_INSTRUCTION

    older_text_parts = []
    for m in older_msgs:
        m_type = getattr(m, "type", "message")
        content = getattr(m, "content", "")
        # Preserve tool call names in older summary
        tool_calls = getattr(m, "tool_calls", None)
        tc_hint = f" [Tool calls: {[tc.get('name') for tc in tool_calls]}]" if tool_calls else ""
        older_text_parts.append(f"[{m_type.upper()}]{tc_hint}: {content}")

    older_text = "\n".join(older_text_parts)
    compressed_summary = await summarize(
        older_text,
        instruction=SUMMARIZER_HISTORY_INSTRUCTION,
        max_tokens=400,
    )

    summary_content = (
        "Earlier conversation summary (older turns compressed to fit context window):\n"
        f"{compressed_summary or older_text[:2500]}"
    )
    summary_msg = SystemMessage(content=summary_content)

    new_messages = system_msgs + [summary_msg, latest_msg]
    new_tokens = estimate_message_tokens(new_messages)

    if new_tokens > max_tokens:
        raise ValueError(
            f"Input context size ({new_tokens} tokens after summarization) exceeds maximum limit of {max_tokens} tokens."
        )

    LOGGER.info("cloudflare_context_compressed_success", previous_tokens=tokens, new_tokens=new_tokens)
    return new_messages


# ---------------------------------------------------------------------------
# Fallback Error Classification
# ---------------------------------------------------------------------------

def is_retryable_error(exc: Exception) -> bool:
    """
    Determine whether an error is a temporary capacity/availability/timeout issue
    that qualifies for a single-retry fallback to the alternate Cloudflare model.

    Rules:
      - Retryable: 429 (rate limit), 500, 502, 503, 504, timeouts, connection errors, capacity/overloaded.
      - Non-retryable: 401/403 (auth), 400 (bad request/invalid model), ValueError, validation errors.
    """
    err_str = str(exc).lower()

    # Explicit non-retryable conditions
    non_retryable_markers = (
        "unauthorized",
        "401",
        "403",
        "forbidden",
        "authentication",
        "invalid api token",
        "invalid api key",
        "invalid_api_key",
        "invalid model",
        "model not found",
        "bad request",
        "user message cannot be empty",
    )
    if any(marker in err_str for marker in non_retryable_markers):
        return False

    status_code = getattr(exc, "status_code", None)
    if status_code in (401, 403, 400):
        return False

    # Retryable conditions
    retryable_markers = (
        "429",
        "rate limit",
        "quota",
        "500",
        "502",
        "503",
        "504",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "internal server error",
        "timeout",
        "timed out",
        "connection error",
        "connection reset",
        "capacity",
        "overloaded",
        "resource exhausted",
        "temporarily unavailable",
    )
    if any(marker in err_str for marker in retryable_markers):
        return True

    if status_code in (429, 500, 502, 503, 504):
        return True

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True

    cls_name = exc.__class__.__name__.lower()
    if any(term in cls_name for term in ("timeout", "ratelimit", "internalserver", "connection")):
        return True

    return False


# ---------------------------------------------------------------------------
# Cloudflare Chat Model Wrapper (Runnable)
# ---------------------------------------------------------------------------

class CloudflareChatModelWrapper(Runnable):
    """
    LangChain-compatible ChatModel wrapper for Cloudflare Workers AI models.
    Supports GLM (@cf/zai-org/glm-4.7-flash) and Gemma (@cf/google/gemma-4-26b-a4b-it).
    """

    def __init__(
        self,
        config: dict[str, Any],
        tools: Optional[Sequence[Any]] = None,
        tool_kwargs: Optional[dict[str, Any]] = None,
        runnable_config: Optional[dict[str, Any]] = None,
    ):
        self._config = config
        self._tools = tools
        self._tool_kwargs = tool_kwargs or {}
        self._runnable_config = runnable_config or {}

        self.primary_model_id = config["model"]
        self.fallback_model_id = get_alternate_model(self.primary_model_id)
        self.enable_fallback = config["enable_fallback"]

        self._primary_chat: Optional[Any] = None
        self._fallback_chat: Optional[Any] = None

    @property
    def model_name(self) -> str:
        return self.primary_model_id

    def _create_inner_chat(self, model_id: str) -> Any:
        """Create a ChatOpenAI instance for the given Cloudflare model."""
        chat = ChatOpenAI(
            model=model_id,
            api_key=self._config["api_token"],
            base_url=self._config["base_url"],
            temperature=self._config["temperature"],
            max_tokens=self._config["max_tokens"],
            streaming=True,
            request_timeout=self._config["timeout"],
        )
        if self._tools is not None:
            chat = chat.bind_tools(self._tools, **self._tool_kwargs)
        if self._runnable_config:
            chat = chat.with_config(self._runnable_config)
        return chat

    def _get_primary_chat(self) -> Any:
        if self._primary_chat is None:
            self._primary_chat = self._create_inner_chat(self.primary_model_id)
        return self._primary_chat

    def _get_fallback_chat(self) -> Any:
        if self._fallback_chat is None:
            self._fallback_chat = self._create_inner_chat(self.fallback_model_id)
        return self._fallback_chat

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> CloudflareChatModelWrapper:
        """Return a new wrapper instance with tools bound."""
        return CloudflareChatModelWrapper(
            config=self._config,
            tools=tools,
            tool_kwargs=kwargs,
            runnable_config=self._runnable_config,
        )

    def with_config(self, config: Optional[RunnableConfig] = None, **kwargs: Any) -> CloudflareChatModelWrapper:
        """Return a new wrapper instance with runnable config attached."""
        merged_config = {**self._runnable_config, **(config or {}), **kwargs}
        return CloudflareChatModelWrapper(
            config=self._config,
            tools=self._tools,
            tool_kwargs=self._tool_kwargs,
            runnable_config=merged_config,
        )

    async def ainvoke(
        self,
        input: Union[list[BaseMessage], str],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """
        Asynchronously invoke Cloudflare Workers AI with validation, context
        compression, and automatic single-retry fallback on transient errors.
        """
        validated_messages = validate_messages(input)
        prepared_messages = await compress_context_if_needed(validated_messages, max_tokens=MAX_CONTEXT_TOKENS)

        merged_config = {**self._runnable_config, **(config or {})}
        primary = self._get_primary_chat()

        try:
            response = await primary.ainvoke(prepared_messages, config=merged_config, **kwargs)
            if hasattr(response, "response_metadata"):
                response.response_metadata["model_name"] = self.primary_model_id
                response.response_metadata["fallback_used"] = False
            LOGGER.info(
                "cloudflare_model_executed",
                model=self.primary_model_id,
                fallback_used=False,
            )
            return response
        except Exception as exc:
            if not self.enable_fallback or not is_retryable_error(exc):
                raise exc

            LOGGER.warning(
                "cloudflare_model_fallback_triggered",
                primary_model=self.primary_model_id,
                fallback_model=self.fallback_model_id,
                error=str(exc)[:200],
            )

            fallback = self._get_fallback_chat()
            response = await fallback.ainvoke(prepared_messages, config=merged_config, **kwargs)
            if hasattr(response, "response_metadata"):
                response.response_metadata["model_name"] = self.fallback_model_id
                response.response_metadata["fallback_used"] = True
            LOGGER.info(
                "cloudflare_model_executed",
                model=self.fallback_model_id,
                fallback_used=True,
            )
            return response

    def invoke(
        self,
        input: Union[list[BaseMessage], str],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """
        Synchronous invoke with validation and fallback.
        """
        validated_messages = validate_messages(input)
        tokens = estimate_message_tokens(validated_messages)
        if tokens > MAX_CONTEXT_TOKENS:
            raise ValueError(
                f"Input context size ({tokens} tokens) exceeds maximum limit of {MAX_CONTEXT_TOKENS} tokens."
            )

        merged_config = {**self._runnable_config, **(config or {})}
        primary = self._get_primary_chat()

        try:
            response = primary.invoke(validated_messages, config=merged_config, **kwargs)
            if hasattr(response, "response_metadata"):
                response.response_metadata["model_name"] = self.primary_model_id
                response.response_metadata["fallback_used"] = False
            LOGGER.info("cloudflare_model_executed", model=self.primary_model_id, fallback_used=False)
            return response
        except Exception as exc:
            if not self.enable_fallback or not is_retryable_error(exc):
                raise exc

            LOGGER.warning(
                "cloudflare_model_fallback_triggered",
                primary_model=self.primary_model_id,
                fallback_model=self.fallback_model_id,
                error=str(exc)[:200],
            )
            fallback = self._get_fallback_chat()
            response = fallback.invoke(validated_messages, config=merged_config, **kwargs)
            if hasattr(response, "response_metadata"):
                response.response_metadata["model_name"] = self.fallback_model_id
                response.response_metadata["fallback_used"] = True
            LOGGER.info("cloudflare_model_executed", model=self.fallback_model_id, fallback_used=True)
            return response

    async def astream(
        self,
        input: Union[list[BaseMessage], str],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[BaseMessageChunk]:
        """
        Stream response tokens from Cloudflare Workers AI.
        If an error occurs before streaming starts, falls back to the alternate model.
        """
        validated_messages = validate_messages(input)
        prepared_messages = await compress_context_if_needed(validated_messages, max_tokens=MAX_CONTEXT_TOKENS)

        merged_config = {**self._runnable_config, **(config or {})}
        primary = self._get_primary_chat()

        try:
            # Yield chunks from primary
            async for chunk in primary.astream(prepared_messages, config=merged_config, **kwargs):
                yield chunk
            LOGGER.info("cloudflare_stream_completed", model=self.primary_model_id, fallback_used=False)
        except Exception as exc:
            if not self.enable_fallback or not is_retryable_error(exc):
                raise exc

            LOGGER.warning(
                "cloudflare_stream_fallback_triggered",
                primary_model=self.primary_model_id,
                fallback_model=self.fallback_model_id,
                error=str(exc)[:200],
            )
            fallback = self._get_fallback_chat()
            async for chunk in fallback.astream(prepared_messages, config=merged_config, **kwargs):
                yield chunk
            LOGGER.info("cloudflare_stream_completed", model=self.fallback_model_id, fallback_used=True)

    def stream(
        self,
        input: Union[list[BaseMessage], str],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[BaseMessageChunk]:
        """Synchronous streaming from Cloudflare Workers AI."""
        validated_messages = validate_messages(input)
        merged_config = {**self._runnable_config, **(config or {})}
        primary = self._get_primary_chat()

        try:
            for chunk in primary.stream(validated_messages, config=merged_config, **kwargs):
                yield chunk
        except Exception as exc:
            if not self.enable_fallback or not is_retryable_error(exc):
                raise exc
            fallback = self._get_fallback_chat()
            for chunk in fallback.stream(validated_messages, config=merged_config, **kwargs):
                yield chunk


# ---------------------------------------------------------------------------
# Model Factory Functions
# ---------------------------------------------------------------------------

def getChatModel(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    enable_fallback: Optional[bool] = None,
    account_id: Optional[str] = None,
    api_token: Optional[str] = None,
    **kwargs: Any,
) -> CloudflareChatModelWrapper:
    """
    Model factory for Cloudflare Workers AI models.

    1. Reads CLOUDFLARE_AI_MODEL from env/argument (defaults to 'glm').
    2. Supports 'glm' (@cf/zai-org/glm-4.7-flash) and 'gemma' (@cf/google/gemma-4-26b-a4b-it).
    3. Uses Cloudflare's OpenAI-compatible base_url.
    4. Uses CLOUDFLARE_API_TOKEN.
    5. Supports streaming, temperature (default 0.4), and max_tokens (default 2000).
    6. Automatic single-retry fallback between GLM and Gemma on transient capacity/timeout errors.
    7. Does not expose credentials to the frontend.

    Raises:
        ValueError: If CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN is missing, or model is invalid.
    """
    config = validate_cloudflare_config(
        account_id=account_id,
        api_token=api_token,
        model_name=model,
    )

    if temperature is not None:
        config["temperature"] = temperature
    if max_tokens is not None:
        config["max_tokens"] = max_tokens
    if enable_fallback is not None:
        config["enable_fallback"] = enable_fallback

    return CloudflareChatModelWrapper(config=config, **kwargs)


# Idiomatic Python PEP8 alias
get_chat_model = getChatModel
