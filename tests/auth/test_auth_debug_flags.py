from app.apis.v1 import auth_routers
from app.core.config import Env


def test_auth_debug_response_requires_explicit_flag(monkeypatch):
    monkeypatch.setattr(auth_routers.config, "ENV", Env.LOCAL)

    assert auth_routers._allow_auth_debug_response(False) is False
    assert auth_routers._allow_auth_debug_response(True) is True


def test_auth_debug_response_is_blocked_in_production(monkeypatch):
    monkeypatch.setattr(auth_routers.config, "ENV", Env.PRODUCTION)

    assert auth_routers._allow_auth_debug_response(True) is False
