"""gpt-5-mini LLM client + embeddings for the VascuTrace GenAI layer.

Research prototype. The model is a reasoning model (gpt-5 family): it consumes
reasoning tokens, so calls MUST pass a generous ``max_completion_tokens`` and an
explicit ``reasoning_effort`` or the visible content comes back empty. This module
centralizes that so every caller is correct.

Key resolution: env ``OPENAI_API_KEY`` -> env ``OPEN_AI_KEY`` -> the ``.env`` file
at the repo root (the project's key is stored there as ``OPEN_AI_KEY``). No key is
ever logged. Offline / no-key callers get a clear :class:`LLMUnavailableError` so
tests can mock the client and CI never needs the network.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_PATH = _REPO_ROOT / ".env"

CHAT_MODEL = "gpt-5-mini"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536


class LLMUnavailableError(RuntimeError):
    """Raised when the OpenAI key is missing or the client cannot be built.

    Callers that must stay offline-safe (CI, deterministic report fallback)
    should catch this and degrade to the deterministic path rather than fail.
    """


def load_openai_key() -> str | None:
    """Return the OpenAI API key without ever logging it.

    Order: ``OPENAI_API_KEY`` env -> ``OPEN_AI_KEY`` env -> ``.env`` file
    (``OPEN_AI_KEY=...`` or ``OPENAI_API_KEY=...``). Returns ``None`` if none found.
    """
    for var in ("OPENAI_API_KEY", "OPEN_AI_KEY"):
        val = os.environ.get(var)
        if val and val.strip():
            return val.strip()
    if _ENV_PATH.is_file():
        text = _ENV_PATH.read_text(encoding="utf-8", errors="ignore")
        for var in ("OPENAI_API_KEY", "OPEN_AI_KEY"):
            m = re.search(rf"^{var}=(.+)$", text, re.M)
            if m:
                return m.group(1).strip().strip('"').strip("'")
    return None


@dataclass(frozen=True)
class LLMConfig:
    chat_model: str = CHAT_MODEL
    embed_model: str = EMBED_MODEL
    reasoning_effort: str = "low"  # minimal|low|medium|high
    max_completion_tokens: int = 1200
    timeout_s: float = 60.0
    max_retries: int = 2
    embed_batch: int = 96
    extra: dict = field(default_factory=dict)


class VascuTraceLLM:
    """Thin, correct wrapper over the OpenAI SDK for gpt-5-mini + embeddings.

    Construct once and reuse. Raises :class:`LLMUnavailableError` at construction
    if no key/SDK is available so callers can fall back deterministically.
    """

    def __init__(
        self, config: LLMConfig | None = None, api_key: str | None = None
    ) -> None:
        self.config = config or LLMConfig()
        key = api_key or load_openai_key()
        if not key:
            raise LLMUnavailableError(
                "No OpenAI API key found (OPENAI_API_KEY / OPEN_AI_KEY / .env)."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise LLMUnavailableError(f"openai SDK not importable: {exc}") from exc
        self._client = OpenAI(
            api_key=key,
            timeout=self.config.timeout_s,
            max_retries=self.config.max_retries,
        )

    # -- chat -----------------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        *,
        json_mode: bool = False,
        reasoning_effort: str | None = None,
        max_completion_tokens: int | None = None,
    ) -> str:
        """One gpt-5-mini turn -> the assistant's text content (never None).

        ``json_mode=True`` sets ``response_format={"type":"json_object"}`` (the
        prompt must ask for JSON). Reasoning models need room, so a generous
        ``max_completion_tokens`` default is used.
        """
        kwargs: dict = {
            "model": self.config.chat_model,
            "messages": messages,
            "reasoning_effort": reasoning_effort or self.config.reasoning_effort,
            "max_completion_tokens": max_completion_tokens
            or self.config.max_completion_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    # -- embeddings -----------------------------------------------------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts with text-embedding-3-small (batched)."""
        out: list[list[float]] = []
        batch = max(1, self.config.embed_batch)
        for i in range(0, len(texts), batch):
            chunk = [t if t.strip() else " " for t in texts[i : i + batch]]
            resp = self._client.embeddings.create(
                model=self.config.embed_model, input=chunk
            )
            out.extend(d.embedding for d in resp.data)
        return out


@lru_cache(maxsize=1)
def get_llm() -> VascuTraceLLM:
    """Process-wide cached client (raises LLMUnavailableError if no key)."""
    return VascuTraceLLM()
