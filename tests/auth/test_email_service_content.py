from email.mime.multipart import MIMEMultipart

import pytest

from app.services import auth as auth_service
from app.services.email_service import EmailService


def test_email_code_and_password_reset_token_ttls_are_10_minutes() -> None:
    assert auth_service.VERIFICATION_CODE_TTL_MINUTES == 10
    assert auth_service.PASSWORD_RESET_TOKEN_TTL_MINUTES == 10
    assert auth_service.EMAIL_VERIFICATION_SIGNUP_TTL_MINUTES == 30


@pytest.mark.asyncio
async def test_email_verification_message_includes_plain_text_and_html_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, MIMEMultipart] = {}

    def fake_send_smtp(self: EmailService, message: MIMEMultipart) -> None:
        captured["message"] = message

    monkeypatch.setattr("app.services.email_service.config.EMAIL_ENABLED", True)
    monkeypatch.setattr(EmailService, "_is_configured", lambda self: True)
    monkeypatch.setattr(EmailService, "_send_smtp", fake_send_smtp)

    sent = await EmailService().send_email_verification_code("user@example.com", "123456")

    assert sent is True
    message = captured["message"]
    assert message.is_multipart()
    assert message.get_content_subtype() == "alternative"
    text_body, html_body = [part.get_payload(decode=True).decode("utf-8") for part in message.get_payload()]
    assert "인증 코드: 123456" in text_body
    assert "10분 동안만 유효" in text_body
    assert '<p style="font-size: 20px; font-weight: 700;">인증 코드: 123456</p>' in html_body
    assert "10분 동안만 유효" in html_body
    assert "<strong>Health Ladder</strong>" in html_body


@pytest.mark.asyncio
async def test_password_reset_message_includes_plain_text_and_html_link(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, MIMEMultipart] = {}
    reset_url = "http://localhost:8080/password-reset/confirm?token=reset-token"

    def fake_send_smtp(self: EmailService, message: MIMEMultipart) -> None:
        captured["message"] = message

    monkeypatch.setattr("app.services.email_service.config.EMAIL_ENABLED", True)
    monkeypatch.setattr(EmailService, "_is_configured", lambda self: True)
    monkeypatch.setattr(EmailService, "_send_smtp", fake_send_smtp)

    sent = await EmailService().send_password_reset_email("user@example.com", reset_url)

    assert sent is True
    text_body, html_body = [part.get_payload(decode=True).decode("utf-8") for part in captured["message"].get_payload()]
    assert reset_url in text_body
    assert "10분 동안만 유효" in text_body
    assert f'<a href="{reset_url}" target="_blank" rel="noopener noreferrer">비밀번호 재설정하기</a>' in html_body
    assert "10분 동안만 유효" in html_body
    assert "<strong>Health Ladder</strong>" in html_body


@pytest.mark.asyncio
async def test_family_invite_message_includes_numeric_code_and_family_page_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, MIMEMultipart] = {}
    invite_url = "http://localhost:8080/family?invite_code=12345678"

    def fake_send_smtp(self: EmailService, message: MIMEMultipart) -> None:
        captured["message"] = message

    monkeypatch.setattr("app.services.email_service.config.EMAIL_ENABLED", True)
    monkeypatch.setattr(EmailService, "_is_configured", lambda self: True)
    monkeypatch.setattr(EmailService, "_send_smtp", fake_send_smtp)

    sent = await EmailService().send_family_invite_email(
        "family@example.com",
        inviter_display_name="동욱",
        invite_code="12345678",
        invite_url=invite_url,
        expires_at_text="2026-05-31 10:00",
    )

    assert sent is True
    text_body, html_body = [part.get_payload(decode=True).decode("utf-8") for part in captured["message"].get_payload()]
    assert "초대 코드: 12345678" in text_body
    assert "30분 동안 유효" in text_body
    assert "가장 최근에 발송된 초대코드만 사용할 수 있습니다." in text_body
    assert "회원가입 또는 로그인 후 가족 페이지에서 아래 초대코드를 입력해 주세요." in text_body
    assert invite_url in text_body
    assert "/family/invitations/accept" not in text_body
    assert "승인" not in text_body
    assert '<p style="font-size: 20px; font-weight: 700;">초대 코드: 12345678</p>' in html_body
    assert "30분 동안 유효" in html_body
    assert "가장 최근에 발송된 초대코드만 사용할 수 있습니다." in html_body
    assert "회원가입 또는 로그인 후 가족 페이지에서 초대코드를 입력해 주세요." in html_body
    assert (
        f'<a href="{invite_url}" target="_blank" rel="noopener noreferrer">가족 페이지에서 초대코드 입력하기</a>'
        in html_body
    )
    assert "/family/invitations/accept" not in html_body
    assert "거절" not in html_body
    assert "<strong>Health Ladder</strong>" in html_body
