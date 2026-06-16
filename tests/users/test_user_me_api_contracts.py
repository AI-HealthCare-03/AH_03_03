from datetime import datetime
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.apis.v1.dependencies import get_request_user
from app.main import app
from app.models.users import Gender
from app.services.users import UserManageService


def _fake_user(**overrides):
    values = {
        "id": 1,
        "login_id": "meuser01",
        "name": "내정보테스터",
        "nickname": "내정보닉",
        "email": "me@example.com",
        "phone_number": None,
        "birthday": "1992-02-02",
        "gender": Gender.FEMALE,
        "address": None,
        "profile_image_url": None,
        "role": "USER",
        "is_active": True,
        "email_verified_at": None,
        "deactivated_at": None,
        "created_at": datetime(2026, 5, 31, 0, 0, 0),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeUserManageService:
    def __init__(self, updated_user):
        self.updated_user = updated_user
        self.received_user = None
        self.received_data = None

    async def update_user(self, user, data):
        self.received_user = user
        self.received_data = data
        return self.updated_user


def test_user_me_api_returns_current_user_profile() -> None:
    user = _fake_user()

    async def fake_current_user():
        return user

    app.dependency_overrides[get_request_user] = fake_current_user

    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/users/me")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["email"] == "me@example.com"
    assert response.json()["name"] == "내정보테스터"
    assert response.json()["nickname"] == "내정보닉"


def test_update_user_me_api_returns_updated_profile() -> None:
    current_user = _fake_user()
    updated_user = _fake_user(name="수정후", nickname="수정닉")
    user_service = FakeUserManageService(updated_user)

    async def fake_current_user():
        return current_user

    app.dependency_overrides[get_request_user] = fake_current_user
    app.dependency_overrides[UserManageService] = lambda: user_service

    try:
        with TestClient(app) as client:
            response = client.patch("/api/v1/users/me", json={"name": "수정후", "nickname": "수정닉"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["name"] == "수정후"
    assert response.json()["nickname"] == "수정닉"
    assert user_service.received_user is current_user
    assert user_service.received_data.name == "수정후"
    assert user_service.received_data.nickname == "수정닉"


def test_user_me_api_requires_authentication() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/users/me")

    assert response.status_code == 401
