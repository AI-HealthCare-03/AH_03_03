from fastapi.testclient import TestClient

from app.main import app


def test_phone_verification_send_is_deferred_from_mvp():
    with TestClient(app) as client:
        response = client.post("/api/v1/auth/phone-verifications/send", json={"phone_number": "01012345678"})

    assert response.status_code == 410
    assert "이메일 인증" in response.json()["detail"]


def test_phone_verification_verify_is_deferred_from_mvp():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/auth/phone-verifications/verify",
            json={"phone_number": "01012345678", "code": "123456"},
        )

    assert response.status_code == 410
    assert "이메일 인증" in response.json()["detail"]
