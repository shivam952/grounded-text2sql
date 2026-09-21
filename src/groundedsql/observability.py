"""Langfuse observability wrapper — optional, degrades silently if unconfigured.

Copied from an internal project's react_sql/observability.py (generic Langfuse code,
no company-specific content). Changes made:
  - Default trace_name updated to "groundedsql/ask"
  - response_usage() helper removed (not needed — chat.completions usage is
    accessed directly via response.usage.prompt_tokens etc.)
  - OpenRouter model slug normalisation added to from_env() so Langfuse can
    look up pricing (strips provider prefix: "anthropic/claude-..." → "claude-...")

If LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are absent, all Tracer methods
are no-ops.  The agent runs identically with or without tracing enabled.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

LOG = logging.getLogger(__name__)


def _normalise_model_for_langfuse(model: str) -> str:
    """Strip OpenRouter provider prefix for Langfuse's pricing lookup.

    OpenRouter slugs look like "anthropic/claude-sonnet-4-5".
    Langfuse's pricing table uses "claude-sonnet-4-5".
    """
    if "/" in model:
        return model.split("/", 1)[1]
    return model


class Tracer:
    def __init__(
        self,
        client: Any = None,
        trace_name: str = "groundedsql/ask",
        tags: list[str] | None = None,
        session_id: str | None = None,
    ) -> None:
        self.client = client
        self.trace_name = trace_name
        self.tags = tags or []
        self.session_id = session_id
        self.trace_id = ""

    @property
    def enabled(self) -> bool:
        return self.client is not None

    @classmethod
    def from_settings(
        cls,
        settings: Any,
        trace_name: str = "groundedsql/ask",
        tags: list[str] | None = None,
        session_id: str | None = None,
    ) -> "Tracer":
        """Create a Tracer using the already-initialised Langfuse client.

        Prefers get_client() over a fresh Langfuse() constructor because
        configure_tracing() already called get_client() at startup — creating
        a second Langfuse() instance in SDK v3 causes traces to be silently dropped.
        """
        if not getattr(settings, "langfuse_configured", False):
            return cls(trace_name=trace_name, tags=tags, session_id=session_id)
        try:
            from langfuse import get_client
            client = get_client()
            return cls(client=client, trace_name=trace_name, tags=tags, session_id=session_id)
        except Exception as exc:
            LOG.warning("Langfuse init failed (from_settings): %s", exc)
            return cls(trace_name=trace_name, tags=tags, session_id=session_id)

    @classmethod
    def from_env(
        cls,
        trace_name: str = "groundedsql/ask",
        tags: list[str] | None = None,
        session_id: str | None = None,
    ) -> "Tracer":
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
        host = (
            os.getenv("LANGFUSE_BASE_URL", "").strip()
            or os.getenv("LANGFUSE_HOST", "").strip()
            or os.getenv("LANGFUSE_HOST_URL", "").strip()
        )
        if not public_key or not secret_key:
            return cls(trace_name=trace_name, tags=tags, session_id=session_id)
        try:
            from langfuse import Langfuse
            client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host or None,
                timeout=int(os.getenv("LANGFUSE_TIMEOUT", "10") or "10"),
            )
            return cls(client=client, trace_name=trace_name, tags=tags, session_id=session_id)
        except Exception as exc:
            LOG.warning("Langfuse init failed: %s", exc)
            return cls(trace_name=trace_name, tags=tags, session_id=session_id)

    @contextmanager
    def trace(self, *, input_payload: Any = None, metadata: dict[str, Any] | None = None) -> Iterator[Any]:
        if not self.enabled:
            yield None
            return
        try:
            from langfuse import propagate_attributes
            attribute_context = propagate_attributes(
                trace_name=self.trace_name,
                session_id=self.session_id,
                tags=self.tags or None,
                metadata={key: str(value)[:200] for key, value in (metadata or {}).items()},
            )
        except Exception:
            attribute_context = None

        def _start() -> Iterator[Any]:
            with self.client.start_as_current_observation(
                as_type="span",
                name=self.trace_name,
                input=input_payload,
                metadata=metadata or {},
            ) as observation:
                self.trace_id = self._safe_current_trace_id()
                try:
                    yield observation
                finally:
                    self.trace_id = self._safe_current_trace_id()

        if attribute_context is None:
            yield from _start()
            return
        with attribute_context:
            yield from _start()

    @contextmanager
    def span(self, name: str, *, input_payload: Any = None, metadata: dict[str, Any] | None = None) -> Iterator[Any]:
        if not self.enabled:
            yield None
            return
        with self.client.start_as_current_observation(
            as_type="span", name=name, input=input_payload, metadata=metadata or {},
        ) as observation:
            yield observation

    @contextmanager
    def generation(self, name: str, *, model: str, input_payload: Any = None, metadata: dict[str, Any] | None = None) -> Iterator[Any]:
        if not self.enabled:
            yield None
            return
        # Normalise model slug for Langfuse pricing table
        langfuse_model = _normalise_model_for_langfuse(model)
        with self.client.start_as_current_observation(
            as_type="generation", name=name, model=langfuse_model,
            input=input_payload, metadata=metadata or {},
        ) as observation:
            yield observation

    def update(self, observation: Any, *, output: Any = None, metadata: dict[str, Any] | None = None) -> None:
        if observation is None:
            return
        try:
            payload: dict[str, Any] = {}
            if output is not None:
                payload["output"] = output
            if metadata is not None:
                payload["metadata"] = metadata
            if payload:
                observation.update(**payload)
        except Exception as exc:
            LOG.warning("Langfuse observation update failed: %s", exc)

    def update_generation(
        self,
        observation: Any,
        *,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        usage_details: dict[str, int] | None = None,
        cost_details: dict[str, float] | None = None,
    ) -> None:
        if observation is None:
            return
        try:
            payload: dict[str, Any] = {}
            if output is not None:
                payload["output"] = output
            if metadata is not None:
                payload["metadata"] = metadata
            if usage_details:
                payload["usage_details"] = usage_details
            if cost_details:
                payload["cost_details"] = cost_details
            if payload:
                observation.update(**payload)
        except Exception as exc:
            LOG.warning("Langfuse generation update failed: %s", exc)

    def error(self, name: str, exc: Exception) -> None:
        if not self.enabled:
            return
        with self.span(name, metadata={"error": type(exc).__name__}) as observation:
            self.update(observation, output={"error": str(exc)})

    def trace_url(self) -> str:
        if not self.enabled or not self.trace_id:
            return ""
        try:
            return str(self.client.get_trace_url(trace_id=self.trace_id))
        except Exception:
            return ""

    def flush(self) -> None:
        if not self.enabled:
            return
        try:
            self.client.flush()
        except Exception as exc:
            LOG.warning("Langfuse flush failed: %s", exc)

    def _safe_current_trace_id(self) -> str:
        if not self.enabled:
            return ""
        try:
            return str(self.client.get_current_trace_id() or "")
        except Exception:
            return ""
