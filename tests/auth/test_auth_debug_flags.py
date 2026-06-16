from fastapi.testclient import TestClient

from app.apis.v1 import auth_routers
from app.core.config import Env
from app.main import app
from app.services.auth import AuthService


def test_auth_debug_response_requires_explicit_flag(monkeypatch):
    monkeypatch.setattr(auth_routers.config, "ENV", Env.LOCAL)

    assert auth_routers._allow_auth_debug_response(False) is False
    assert auth_routers._allow_auth_debug_response(True) is True


def test_auth_debug_response_is_blocked_in_production(monkeypatch):
    monkeypatch.setattr(auth_routers.config, "ENV", Env.PRODUCTION)

    assert auth_routers._allow_auth_debug_response(True) is False


async def _fake_send_email_verification_code(_self: AuthService, email: str) -> str:
    assert email == "demo@example.com"
    return "123456"


def test_email_verification_send_includes_debug_code_only_when_enabled(monkeypatch):
    monkeypatch.setattr(auth_routers.config, "ENV", Env.LOCAL)
    monkeypatch.setattr(auth_routers.config, "EMAIL_VERIFICATION_DEBUG", True)
    monkeypatch.setattr(AuthService, "send_email_verification_code", _fake_send_email_verification_code)

    with TestClient(app) as client:
        response = client.post("/api/v1/auth/email-verifications/send", json={"email": "demo@example.com"})

    assert response.status_code == 200
    assert response.json()["debug_code"] == "123456"


def test_email_verification_send_omits_debug_code_when_disabled(monkeypatch):
    monkeypatch.setattr(auth_routers.config, "ENV", Env.LOCAL)
    monkeypatch.setattr(auth_routers.config, "EMAIL_VERIFICATION_DEBUG", False)
    monkeypatch.setattr(AuthService, "send_email_verification_code", _fake_send_email_verification_code)

    with TestClient(app) as client:
        response = client.post("/api/v1/auth/email-verifications/send", json={"email": "demo@example.com"})

    assert response.status_code == 200
    body = response.json()
    assert "debug_code" not in body
    assert body["detail"] == "인증코드가 생성되었습니다."
