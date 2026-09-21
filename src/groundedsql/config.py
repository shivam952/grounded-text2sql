"""Application configuration via pydantic-settings.

All settings can be overridden via environment variables or a .env file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── OpenRouter ─────────────────────────────────────────────────────────────
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # ── Models ─────────────────────────────────────────────────────────────────
    #   Primary model used for the ReAct agent loop.
    #   Claude Sonnet is the default: best text-to-SQL accuracy at mid-tier cost.
    groundedsql_model: str = "anthropic/claude-sonnet-4-5"

    #   Model used for the bulk eval run (50–100 BIRD questions).
    #   Default: Gemini Flash Lite (~10× cheaper than Sonnet, good enough for scoring).
    groundedsql_eval_model: str = "google/gemini-2.5-flash-lite"

    #   Model used for the LLM-as-judge grounding check stage.
    #   Always cheap regardless of what the main agent model is.
    groundedsql_grounding_model: str = "google/gemini-2.5-flash-lite"

    # ── Agent tuning ───────────────────────────────────────────────────────────
    max_iterations: int = 15
    tool_row_limit: int = 50
    query_timeout_ms: int = 10_000

    # ── Langfuse (optional — agent runs fine without it) ───────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = ""
    langfuse_timeout: int = 10

    @property
    def langfuse_configured(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @field_validator("openrouter_api_key")
    @classmethod
    def _warn_missing_key(cls, v: str) -> str:
        # Defer the hard error to call time so tests/dry-runs still import cleanly
        return v

    def openai_client(self, **kwargs: Any):  # type: ignore[return]
        """Return an openai.OpenAI client pointed at OpenRouter."""
        from openai import OpenAI
        return OpenAI(
            base_url=self.openrouter_base_url,
            api_key=self.openrouter_api_key or "no-key",
            **kwargs,
        )


# Module-level singleton — import this everywhere rather than constructing Settings()
# repeatedly; pydantic-settings reads .env once on construction.
settings = Settings()
