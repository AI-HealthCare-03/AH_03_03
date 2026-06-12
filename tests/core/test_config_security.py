import pytest
from pydantic import ValidationError

from app.core.config import Config, Env


def build_config(**overrides) -> Config:
    defaults = {
        "ENV": Env.PRODUCTION,
        "SECRET_KEY": "prod-secret-key-with-enough-entropy",
        "COOKIE_DOMAIN": "example.com",
    }
    defaults.update(overrides)
    return Config(_env_file=None, **defaults)


@pytest.mark.parametrize(
    "secret_key",
    ["", "PLEASE_CHANGE_ME", "default-secret-keyabc123"],
)
def test_production_rejects_empty_placeholder_or_default_secret_key(secret_key: str) -> None:
    with pytest.raises(ValidationError, match="SECRET_KEY"):
        build_config(SECRET_KEY=secret_key)


def test_local_allows_generated_default_secret_key() -> None:
    config = Config(ENV=Env.LOCAL, SECRET_KEY="default-secret-keyabc123", _env_file=None)

    assert config.SECRET_KEY == "default-secret-keyabc123"


def test_production_rejects_insecure_refresh_cookie() -> None:
    with pytest.raises(ValidationError, match="REFRESH_TOKEN_COOKIE_SECURE"):
        build_config(REFRESH_TOKEN_COOKIE_SECURE=False)


@pytest.mark.parametrize("cookie_domain", ["localhost", "127.0.0.1", "::1"])
def test_production_rejects_localhost_cookie_domain(cookie_domain: str) -> None:
    with pytest.raises(ValidationError, match="COOKIE_DOMAIN"):
        build_config(COOKIE_DOMAIN=cookie_domain)


def test_production_accepts_secure_cookie_defaults_with_real_domain() -> None:
    config = build_config()

    assert config.is_production is True
    assert config.refresh_token_cookie_secure is True
    assert config.refresh_token_cookie_domain == "example.com"


def test_production_rejects_invalid_refresh_cookie_samesite() -> None:
    with pytest.raises(ValidationError, match="REFRESH_TOKEN_COOKIE_SAMESITE"):
        build_config(REFRESH_TOKEN_COOKIE_SAMESITE="invalid")
