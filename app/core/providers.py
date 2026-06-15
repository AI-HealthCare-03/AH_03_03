from __future__ import annotations

from typing import Any


def _has_value(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def has_openai_config(settings: Any) -> bool:
    return _has_value(getattr(settings, "OPENAI_API_KEY", None))


def has_smtp_config(settings: Any) -> bool:
    return (
        _has_value(getattr(settings, "SMTP_HOST", None))
        and bool(getattr(settings, "SMTP_PORT", None))
        and _has_value(getattr(settings, "SMTP_USERNAME", None))
        and _has_value(getattr(settings, "SMTP_PASSWORD", None))
        and _has_value(getattr(settings, "SMTP_FROM_EMAIL", None))
    )


def has_langfuse_config(settings: Any) -> bool:
    return (
        _has_value(getattr(settings, "LANGFUSE_BASE_URL", None))
        and _has_value(getattr(settings, "LANGFUSE_PUBLIC_KEY", None))
        and _has_value(getattr(settings, "LANGFUSE_SECRET_KEY", None))
    )


def has_paddle_ocr_runtime() -> bool:
    # PaddleOCR is intentionally checked lazily because importing it can be slow
    # and may fail on platforms without the optional OCR dependency group.
    try:
        import paddleocr  # noqa: F401
    except Exception:
        return False
    return True
