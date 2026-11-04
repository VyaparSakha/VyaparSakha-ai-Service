"""
app/lib/gemini_keys.py
======================
Gemini API Key Manager with automatic key rotation and rate-limit handling.

Supports:
1. Comma-separated keys in env var `GEMINI_API_KEYS=key1,key2,key3`.
2. Numbered env vars `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, etc.
3. Standard `GEMINI_API_KEY`.

Provides key rotation helper and embedding wrapper with rate-limit fallback.
"""

from __future__ import annotations

import os
import time
import asyncio
from typing import Any
import structlog

from app.config.settings import get_settings

LOGGER = structlog.get_logger("vyaparsathi.ai.gemini_keys")

_key_index: int = 0
_cooldowns: dict[str, float] = {}  # key -> timestamp until on cooldown
_COOLDOWN_DURATION_SECONDS = 60.0  # 1-minute cooldown for rate-limited key


def get_provider_keys(provider_prefix: str) -> list[str]:
    """
    Collect all available API keys for a given provider (e.g. GEMINI, GROQ, NVIDIA, OPENROUTER).
    Supports:
    1. Comma-separated keys in {PREFIX}_API_KEYS.
    2. Settings attribute (e.g. settings.groq_api_key).
    3. Standard {PREFIX}_API_KEY.
    4. Numbered env vars with and without underscore: {PREFIX}_API_KEY_1, {PREFIX}_API_KEY1, etc.
    De-duplicates and preserves order.
    """
    prefix = provider_prefix.upper().rstrip("_")
    keys: list[str] = []
    settings = get_settings()

    raw_keys_sources = [
        os.environ.get(f"{prefix}_API_KEYS", ""),
        getattr(settings, f"{prefix.lower()}_api_key", "") or "",
        os.environ.get(f"{prefix}_API_KEY", ""),
        os.environ.get(f"{prefix}_KEY", ""),
        os.environ.get(f"{prefix}_API_TOKEN", ""),
        os.environ.get(f"{prefix}_TOKEN", ""),
        os.environ.get(f"{prefix}", ""),
    ]

    for raw in raw_keys_sources:
        if raw:
            for part in str(raw).split(","):
                cleaned = part.strip()
                if cleaned and cleaned not in keys:
                    keys.append(cleaned)

    # Check numbered keys: PREFIX_API_KEY_1, PREFIX_API_KEY1, PREFIX_KEY_1, PREFIX_1...
    for i in range(1, 10):
        for pattern in (
            f"{prefix}_API_KEY_{i}",
            f"{prefix}_API_KEY{i}",
            f"{prefix}_KEY_{i}",
            f"{prefix}_KEY{i}",
            f"{prefix}_{i}",
        ):
            val = os.environ.get(pattern, "").strip()
            if val and val not in keys:
                keys.append(val)

    return keys


def get_gemini_api_keys() -> list[str]:
    """
    Collect all available Gemini API keys from settings and environment.
    De-duplicates and preserves order.
    """
    return get_provider_keys("GEMINI")



def get_next_gemini_key() -> str | None:
    """
    Return the next active Gemini API key using round-robin rotation,
    skipping keys currently on rate-limit cooldown.
    """
    global _key_index
    keys = get_gemini_api_keys()
    if not keys:
        return None

    now = time.time()
    n = len(keys)

    for offset in range(n):
        idx = (_key_index + offset) % n
        candidate_key = keys[idx]
        cooldown_until = _cooldowns.get(candidate_key, 0.0)

        if now >= cooldown_until:
            _key_index = (idx + 1) % n
            return candidate_key

    # All keys are on cooldown — return the first key as fallback
    _key_index = (_key_index + 1) % n
    return keys[0]


def mark_key_rate_limited(api_key: str, cooldown_seconds: float = _COOLDOWN_DURATION_SECONDS) -> None:
    """
    Mark an API key as rate-limited so it gets bypassed for cooldown_seconds.
    """
    if not api_key:
        return
    _cooldowns[api_key] = time.time() + cooldown_seconds
    masked_key = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "..."
    LOGGER.warning("gemini_key_rate_limited", key=masked_key, cooldown_seconds=cooldown_seconds)


class RotatingGoogleGenerativeAIEmbeddings:
    """
    Embeddings wrapper that rotates Gemini API keys on rate limits (429/QuotaExceeded).
    """

    def __init__(self, model: str = "models/gemini-embedding-001"):
        self.model = model

    def _get_embeddings_instance(self, api_key: str):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=self.model, google_api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = get_gemini_api_keys()
        if not keys:
            raise ValueError("No GEMINI_API_KEY configured.")

        last_error = None
        for _ in range(len(keys)):
            key = get_next_gemini_key()
            if not key:
                break
            try:
                emb = self._get_embeddings_instance(key)
                return emb.embed_documents(texts)
            except Exception as exc:
                last_error = exc
                err_str = str(exc).lower()
                if "429" in err_str or "quota" in err_str or "rate" in err_str:
                    mark_key_rate_limited(key)
                else:
                    raise exc
        if last_error:
            raise last_error
        return []

    def embed_query(self, text: str) -> list[float]:
        keys = get_gemini_api_keys()
        if not keys:
            raise ValueError("No GEMINI_API_KEY configured.")

        last_error = None
        for _ in range(len(keys)):
            key = get_next_gemini_key()
            if not key:
                break
            try:
                emb = self._get_embeddings_instance(key)
                return emb.embed_query(text)
            except Exception as exc:
                last_error = exc
                err_str = str(exc).lower()
                if "429" in err_str or "quota" in err_str or "rate" in err_str:
                    mark_key_rate_limited(key)
                else:
                    raise exc
        if last_error:
            raise last_error
        return []
