from fastapi.testclient import TestClient

from app.main import app
from app.repositories.user_repository import UserRepository


def test_check_phone_reports_available_for_unused_normalized_number(monkeypatch):
    seen_numbers: list[str] = []

    async def fake_exists_by_phone_number(_self: UserRepository, phone_number: str) -> bool:
        seen_numbers.append(phone_number)
        return False

    monkeypatch.setattr(UserRepository, "exists_by_phone_number", fake_exists_by_phone_number)

    with TestClient(app) as client:
        response = client.get("/api/v1/auth/check-phone", params={"phone_number": "010-1234-5678"})

    assert response.status_code == 200
    assert response.json() == {
        "available": True,
        "message": "사용 가능한 휴대폰 번호입니다.",
    }
    assert seen_numbers == ["01012345678"]


def test_check_phone_reports_unavailable_for_existing_normalized_number(monkeypatch):
    seen_numbers: list[str] = []

    async def fake_exists_by_phone_number(_self: UserRepository, phone_number: str) -> bool:
        seen_numbers.append(phone_number)
        return phone_number == "01012345678"

    monkeypatch.setattr(UserRepository, "exists_by_phone_number", fake_exists_by_phone_number)

    with TestClient(app) as client:
        response = client.get("/api/v1/auth/check-phone", params={"phone_number": "+82 10-1234-5678"})

    assert response.status_code == 200
    assert response.json() == {
        "available": False,
        "message": "이미 사용중인 휴대폰 번호입니다.",
    }
    assert seen_numbers == ["01012345678"]
