from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.apis.v1 import auth_routers
from app.main import app
from app.services.auth import AuthService
from app.services.jwt import JwtService


class FakeToken:
    def __init__(self, value: str) -> None:
        self.value = value
        self.payload = {
            "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
            "jti": f"{value}-jti",
            "user_id": 1,
        }

    def __str__(self) -> str:
        return self.value


class FakeLoginAuthService:
    def __init__(self) -> None:
        self.authenticated_request = None
        self.logged_in_user = None

    async def authenticate(self, request):
        self.authenticated_request = request
        return SimpleNamespace(id=1, email=request.email, is_active=True)

    async def login(self, user):
        self.logged_in_user = user
        return {
            "access_token": FakeToken("access-token"),
            "refresh_token": FakeToken("refresh-token"),
        }


class FakeSignupAuthService:
    def __init__(self) -> None:
        self.signup_request = None

    async def signup(self, request) -> None:
        self.signup_request = request


class FakeRefreshJwtService:
    def __init__(self, refresh_token):
        self.refresh_token = refresh_token

    def verify_jwt(self, token: str, token_type: str):
        assert token == "refresh-token"
        assert token_type == "refresh"
        return self.refresh_token


class FakeRefreshAuthService:
    def __init__(self) -> None:
        self.checked_refresh_token = None

    async def ensure_refresh_token_active(self, refresh_token) -> None:
        self.checked_refresh_token = refresh_token


def test_login_api_returns_access_token_and_sets_refresh_cookie() -> None:
    auth_service = FakeLoginAuthService()
    app.dependency_overrides[AuthService] = lambda: auth_service

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/auth/login",
                json={"email": "login@example.com", "password": "Password123!"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"access_token": "access-token"}
    assert auth_service.authenticated_request.email == "login@example.com"
    assert auth_service.logged_in_user.email == "login@example.com"
    assert any(
        f"{auth_routers.config.REFRESH_TOKEN_COOKIE_NAME}=refresh-token" in header
        for header in response.headers.get_list("set-cookie")
    )


def test_signup_api_success_uses_current_signup_contract() -> None:
    auth_service = FakeSignupAuthService()
    app.dependency_overrides[AuthService] = lambda: auth_service

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/auth/signup",
                json={
                    "login_id": "signupapi",
                    "email": "signup-api@example.com",
                    "password": "Password123!",
                    "name": "테스터",
                    "nickname": "테스트닉",
                    "gender": "MALE",
                    "birth_date": "1990-01-01",
                    "phone_number": "01012345678",
                    "privacy_consent_agreed": True,
                    "sensitive_data_agreed": True,
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 201
    assert response.json() == {"detail": "회원가입이 성공적으로 완료되었습니다."}
    assert auth_service.signup_request.email == "signup-api@example.com"
    assert auth_service.signup_request.nickname == "테스트닉"
    assert auth_service.signup_request.privacy_consent_agreed is True
    assert auth_service.signup_request.sensitive_data_agreed is True


def test_token_refresh_api_returns_new_access_token_from_refresh_cookie() -> None:
    refresh_token = SimpleNamespace(
        access_token=FakeToken("new-access-token"),
        payload={"jti": "refresh-token-jti", "user_id": 1},
    )
    jwt_service = FakeRefreshJwtService(refresh_token)
    auth_service = FakeRefreshAuthService()
    app.dependency_overrides[JwtService] = lambda: jwt_service
    app.dependency_overrides[AuthService] = lambda: auth_service

    try:
        with TestClient(app) as client:
            client.cookies.set(auth_routers.config.REFRESH_TOKEN_COOKIE_NAME, "refresh-token")
            response = client.get("/api/v1/auth/token/refresh")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"access_token": "new-access-token"}
    assert auth_service.checked_refresh_token is refresh_token


def test_token_refresh_api_requires_refresh_cookie() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/auth/token/refresh")

    assert response.status_code == 401
    assert response.json()["detail"] == "Refresh token is missing."
