import asyncio
import smtplib
from email.message import EmailMessage
from email.utils import formataddr

from app.core import config


class EmailDeliveryError(Exception):
    pass


class EmailConfigurationError(EmailDeliveryError):
    pass


class EmailService:
    def status(self) -> str:
        if not config.EMAIL_ENABLED:
            return "disabled"
        if not self._is_configured():
            return "misconfigured"
        return "configured"

    async def send_email_verification_code(self, email: str, code: str) -> bool:
        subject = "AI HealthCare 이메일 인증 코드"
        body = (
            "AI HealthCare 이메일 인증 코드입니다.\n\n"
            f"인증코드: {code}\n"
            "유효시간: 10분\n\n"
            "본인이 요청하지 않았다면 이 이메일을 무시해주세요."
        )
        return await self._send_email(email, subject, body)

    async def send_password_reset_email(self, email: str, reset_url: str) -> bool:
        subject = "AI HealthCare 비밀번호 재설정 안내"
        body = (
            "AI HealthCare 비밀번호 재설정 안내입니다.\n\n"
            f"아래 링크에서 비밀번호를 재설정해주세요.\n{reset_url}\n\n"
            "유효시간: 30분\n"
            "본인이 요청하지 않았다면 이 이메일을 무시해주세요."
        )
        return await self._send_email(email, subject, body)

    async def send_family_invite_email(
        self,
        email: str,
        *,
        inviter_display_name: str,
        invite_url: str,
        expires_at_text: str,
    ) -> bool:
        subject = "AI HealthCare 가족연동 초대가 도착했습니다."
        body = (
            "AI HealthCare 가족연동 초대 안내입니다.\n\n"
            f"{inviter_display_name}님이 가족연동을 요청했습니다.\n"
            "아래 링크를 눌러 초대를 수락해주세요.\n"
            f"{invite_url}\n\n"
            f"이 초대는 {expires_at_text}까지 유효합니다.\n\n"
            "본인이 요청받은 초대가 아니라면 이 이메일을 무시해주세요."
        )
        return await self._send_email(email, subject, body)

    async def _send_email(self, recipient: str, subject: str, body: str) -> bool:
        if not config.EMAIL_ENABLED:
            if config.is_production:
                raise EmailConfigurationError("Email delivery is disabled.")
            return False
        if not self._is_configured():
            raise EmailConfigurationError("SMTP settings are incomplete.")

        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = formataddr((config.SMTP_FROM_NAME, config.SMTP_FROM_EMAIL or ""))
        message["To"] = recipient
        message.set_content(body)

        try:
            await asyncio.to_thread(self._send_smtp, message)
            return True
        except EmailDeliveryError:
            raise
        except Exception as exc:
            raise EmailDeliveryError("Failed to send email.") from exc

    def _send_smtp(self, message: EmailMessage) -> None:
        if config.SMTP_USE_TLS:
            with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=10) as smtp:
                smtp.starttls()
                self._login_if_needed(smtp)
                smtp.send_message(message)
            return

        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=10) as smtp:
            self._login_if_needed(smtp)
            smtp.send_message(message)

    def _login_if_needed(self, smtp: smtplib.SMTP) -> None:
        if config.SMTP_USERNAME and config.SMTP_PASSWORD:
            smtp.login(config.SMTP_USERNAME, config.SMTP_PASSWORD)

    def _is_configured(self) -> bool:
        return bool(config.SMTP_HOST and config.SMTP_PORT and config.SMTP_FROM_EMAIL)
