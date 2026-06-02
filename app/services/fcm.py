from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class FCMProviderUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class FCMSendResult:
    success_count: int
    failure_count: int
    provider_message_ids: list[str]


class FCMService:
    """Thin Firebase Admin SDK wrapper.

    Token registration is handled by app.services.notifications. This class is
    intentionally limited to provider calls so tests can mock it cleanly.
    """

    def _get_messaging(self) -> Any:
        try:
            import firebase_admin
            from firebase_admin import messaging
        except ImportError as exc:
            raise FCMProviderUnavailableError("Firebase Admin SDK is not installed.") from exc

        if not firebase_admin._apps:
            firebase_admin.initialize_app()
        return messaging

    def send_to_tokens(
        self,
        *,
        tokens: Sequence[str],
        title: str,
        body: str,
        data: Mapping[str, str] | None = None,
    ) -> FCMSendResult:
        if not tokens:
            return FCMSendResult(success_count=0, failure_count=0, provider_message_ids=[])

        messaging = self._get_messaging()
        message = messaging.MulticastMessage(
            tokens=list(tokens),
            notification=messaging.Notification(title=title, body=body),
            data=dict(data or {}),
        )
        response = messaging.send_each_for_multicast(message)
        provider_message_ids = [
            item.message_id for item in response.responses if getattr(item, "success", False) and item.message_id
        ]
        return FCMSendResult(
            success_count=response.success_count,
            failure_count=response.failure_count,
            provider_message_ids=provider_message_ids,
        )
