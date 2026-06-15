from types import SimpleNamespace

from app.core.providers import has_langfuse_config, has_openai_config, has_smtp_config


def test_openai_config_requires_api_key() -> None:
    assert has_openai_config(SimpleNamespace(OPENAI_API_KEY="sk-test")) is True
    assert has_openai_config(SimpleNamespace(OPENAI_API_KEY="")) is False
    assert has_openai_config(SimpleNamespace()) is False


def test_smtp_config_requires_delivery_fields() -> None:
    configured = SimpleNamespace(
        SMTP_HOST="smtp.example.com",
        SMTP_PORT=587,
        SMTP_USERNAME="user",
        SMTP_PASSWORD="password",
        SMTP_FROM_EMAIL="noreply@example.com",
    )
    missing_password = SimpleNamespace(
        SMTP_HOST="smtp.example.com",
        SMTP_PORT=587,
        SMTP_USERNAME="user",
        SMTP_PASSWORD="",
        SMTP_FROM_EMAIL="noreply@example.com",
    )
    missing_username = SimpleNamespace(
        SMTP_HOST="smtp.example.com",
        SMTP_PORT=587,
        SMTP_USERNAME="",
        SMTP_PASSWORD="password",
        SMTP_FROM_EMAIL="noreply@example.com",
    )

    assert has_smtp_config(configured) is True
    assert has_smtp_config(missing_password) is False
    assert has_smtp_config(missing_username) is False


def test_langfuse_config_requires_host_and_keys() -> None:
    configured = SimpleNamespace(
        LANGFUSE_BASE_URL="http://localhost:3000",
        LANGFUSE_PUBLIC_KEY="pk-test",
        LANGFUSE_SECRET_KEY="sk-test",
    )
    missing_secret = SimpleNamespace(
        LANGFUSE_BASE_URL="http://localhost:3000",
        LANGFUSE_PUBLIC_KEY="pk-test",
        LANGFUSE_SECRET_KEY=None,
    )

    assert has_langfuse_config(configured) is True
    assert has_langfuse_config(missing_secret) is False
