from app.services import auth as auth_service
from app.services.auth import AuthService


def test_password_reset_url_uses_frontend_auth_route(monkeypatch) -> None:
    monkeypatch.setattr(auth_service.config, "FRONTEND_BASE_URL", "https://healthladder.example/")

    reset_url = AuthService()._password_reset_url("token with/slash")

    assert reset_url == "https://healthladder.example/auth/password-reset/confirm?token=token%20with%2Fslash"
