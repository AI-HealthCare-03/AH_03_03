from types import SimpleNamespace

from app.services.fcm import FCMService


class FakeMessaging:
    class Notification:
        def __init__(self, title: str, body: str) -> None:
            self.title = title
            self.body = body

    class MulticastMessage:
        def __init__(self, *, tokens: list[str], notification, data: dict[str, str]) -> None:
            self.tokens = tokens
            self.notification = notification
            self.data = data

    @staticmethod
    def send_each_for_multicast(message) -> SimpleNamespace:
        assert message.tokens == ["token-1", "token-2"]
        assert message.notification.title == "알림"
        assert message.notification.body == "내용"
        assert message.data == {"kind": "demo"}
        return SimpleNamespace(
            success_count=1,
            failure_count=1,
            responses=[
                SimpleNamespace(success=True, message_id="provider-message-1"),
                SimpleNamespace(success=False, message_id=None),
            ],
        )


def test_fcm_service_skips_provider_when_tokens_empty() -> None:
    class ProviderShouldNotRunService(FCMService):
        def _get_messaging(self):
            raise AssertionError("Provider should not be initialized without tokens")

    result = ProviderShouldNotRunService().send_to_tokens(tokens=[], title="알림", body="내용")

    assert result.success_count == 0
    assert result.failure_count == 0
    assert result.provider_message_ids == []


def test_fcm_service_uses_mockable_firebase_messaging() -> None:
    class FakeFCMService(FCMService):
        def _get_messaging(self):
            return FakeMessaging

    result = FakeFCMService().send_to_tokens(
        tokens=["token-1", "token-2"],
        title="알림",
        body="내용",
        data={"kind": "demo"},
    )

    assert result.success_count == 1
    assert result.failure_count == 1
    assert result.provider_message_ids == ["provider-message-1"]
