from __future__ import annotations

import os

import pytest

os.environ["LANGFUSE_ENABLED"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"


@pytest.fixture(autouse=True)
def disable_external_trace_export(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests from exporting Langfuse/OpenTelemetry traces.

    Local developer `.env` files may point Langfuse to a real self-hosted
    server. Tests should not inherit that side effect; individual tests can
    still monkeypatch doubles after this fixture runs.
    """
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")

    from app.core import config

    monkeypatch.setattr(config, "LANGFUSE_ENABLED", False)
    monkeypatch.setattr(config, "LANGFUSE_PUBLIC_KEY", None)
    monkeypatch.setattr(config, "LANGFUSE_SECRET_KEY", None)
    monkeypatch.setattr(config, "LANGFUSE_BASE_URL", None)
    monkeypatch.setattr(config, "LANGFUSE_HOST", None)
