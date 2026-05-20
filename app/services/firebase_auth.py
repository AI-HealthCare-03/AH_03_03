from datetime import date, datetime
from typing import Any

from app.core import config
from app.models.users import Gender, User
from app.repositories.user_repository import UserRepository

FIREBASE_DEFAULT_DOMAIN = "firebase.local"


async def sync_firebase_user(decoded_token: dict[str, Any]) -> User:
    firebase_uid = str(decoded_token.get("uid") or decoded_token.get("sub") or "")
    if not firebase_uid:
        raise ValueError("Firebase token does not contain uid.")

    email = _safe_email(decoded_token, firebase_uid)
    name = _safe_name(decoded_token, email)
    nickname = _safe_optional(decoded_token.get("name"), 30)
    profile_image_url = _safe_optional(decoded_token.get("picture"), 500)
    email_verified_at = datetime.now(config.TIMEZONE) if decoded_token.get("email_verified") else None

    repo = UserRepository()
    user = await repo.get_user_by_firebase_uid(firebase_uid)
    if user is None:
        user = await repo.get_user_by_email(email)

    if user is not None:
        await repo.update_instance(
            user,
            {
                "firebase_uid": firebase_uid,
                "auth_provider": "firebase",
                "email": email,
                "name": name,
                "nickname": nickname,
                "profile_image_url": profile_image_url,
                "email_verified_at": email_verified_at,
                "role": user.role or "USER",
            },
        )
        await user.refresh_from_db()
        return user

    return await repo.create_user(
        login_id=_build_firebase_login_id(firebase_uid),
        firebase_uid=firebase_uid,
        auth_provider="firebase",
        email=email,
        hashed_password="firebase-auth-user",
        name=name,
        nickname=nickname,
        phone_number=_build_firebase_phone_number(firebase_uid),
        gender=Gender.MALE,
        birthday=date(1970, 1, 1),
        profile_image_url=profile_image_url,
        email_verified_at=email_verified_at,
        role="USER",
    )


def _safe_email(decoded_token: dict[str, Any], firebase_uid: str) -> str:
    email = str(decoded_token.get("email") or "").strip()
    if not email:
        email = f"{firebase_uid[:20]}@{FIREBASE_DEFAULT_DOMAIN}"
    return email[:40]


def _safe_name(decoded_token: dict[str, Any], email: str) -> str:
    name = str(decoded_token.get("name") or decoded_token.get("display_name") or email.split("@", 1)[0]).strip()
    return name[:20] or "Firebase User"


def _safe_optional(value: Any, max_length: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:max_length] if text else None


def _build_firebase_login_id(firebase_uid: str) -> str:
    return f"fb_{firebase_uid[:37]}"


def _build_firebase_phone_number(firebase_uid: str) -> str:
    digits = "".join(str(ord(char) % 10) for char in firebase_uid)
    return f"fb{digits[:9]}".ljust(11, "0")
