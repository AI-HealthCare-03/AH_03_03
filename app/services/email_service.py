import asyncio
import smtplib
from email.message import Message
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from html import escape

from app.core import config
from app.core.providers import has_smtp_config


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
        subject = "[Health Ladder] 이메일 인증 코드 안내"
        body = (
            "안녕하세요, Health Ladder입니다.\n\n"
            "아래 인증 코드를 입력하여 이메일 인증을 완료해 주세요.\n\n\n"
            f"인증 코드: {code}\n\n"
            "이 코드는 10분 동안만 유효합니다.\n\n\n"
            "Health Ladder 드림"
        )
        html_body = (
            "<p>안녕하세요, <strong>Health Ladder</strong>입니다.</p>"
            "<p>아래 인증 코드를 입력하여 이메일 인증을 완료해 주세요.</p>"
            f'<p style="font-size: 20px; font-weight: 700;">인증 코드: {escape(code)}</p>'
            "<p>이 코드는 10분 동안만 유효합니다.</p>"
            "<p>감사합니다.<br><strong>Health Ladder</strong> 드림</p>"
        )
        return await self._send_email(email, subject, body, html_body=html_body)

    async def send_password_reset_email(self, email: str, reset_url: str) -> bool:
        subject = "[Health Ladder] 비밀번호 재설정 안내"
        body = (
            "안녕하세요, Health Ladder입니다.\n\n"
            "비밀번호 재설정을 요청하셨습니다.\n\n"
            "아래 링크를 눌러 새 비밀번호를 설정해 주세요.\n\n\n"
            f"{reset_url}\n\n"
            "이 링크는 10분 동안만 유효합니다.\n\n"
            "계정 보호를 위해 비밀번호를 타인에게 공유하지 마세요.\n\n\n"
            "Health Ladder 드림"
        )
        escaped_reset_url = escape(reset_url, quote=True)
        html_body = (
            "<p>안녕하세요, <strong>Health Ladder</strong>입니다.</p>"
            "<p>비밀번호 재설정을 요청하셨습니다.</p>"
            "<p>아래 링크를 눌러 새 비밀번호를 설정해 주세요.</p>"
            f'<p><a href="{escaped_reset_url}" target="_blank" rel="noopener noreferrer">비밀번호 재설정하기</a></p>'
            "<p>이 링크는 10분 동안만 유효합니다.</p>"
            "<p>계정 보호를 위해 비밀번호를 타인에게 공유하지 마세요.</p>"
            "<p>감사합니다.<br><strong>Health Ladder</strong> 드림</p>"
        )
        return await self._send_email(email, subject, body, html_body=html_body)

    async def send_family_invite_email(
        self,
        email: str,
        *,
        inviter_display_name: str,
        invite_code: str,
        invite_url: str,
        expires_at_text: str,
    ) -> bool:
        subject = "[Health Ladder] 가족 건강관리 초대 안내"
        body = (
            "안녕하세요, Health Ladder입니다.\n\n"
            f"{inviter_display_name}님이 가족 건강관리 기능에 초대했습니다.\n\n"
            "가족 페이지에서 아래 초대코드를 입력해 연결을 완료해 주세요.\n\n\n"
            f"초대 코드: {invite_code}\n\n\n"
            f"초대 링크: {invite_url}\n\n"
            f"초대 만료: {expires_at_text}\n\n"
            "초대코드는 발송 시점부터 30분 동안 유효합니다.\n"
            "가장 최근에 발송된 초대코드만 사용할 수 있습니다.\n"
            "회원이 아니신 경우 회원가입 또는 로그인 후 가족 페이지에서 아래 초대코드를 입력해 주세요.\n\n"
            "Health Ladder에서는 가족과 함께 건강 기록과 건강관리 현황을 확인할 수 있습니다.\n\n\n"
            "Health Ladder 드림"
        )
        escaped_invite_code = escape(invite_code)
        escaped_invite_url = escape(invite_url, quote=True)
        escaped_inviter_display_name = escape(inviter_display_name)
        escaped_expires_at_text = escape(expires_at_text)
        html_body = (
            "<p>안녕하세요, <strong>Health Ladder</strong>입니다.</p>"
            f"<p>{escaped_inviter_display_name}님이 가족 건강관리 기능에 초대했습니다.</p>"
            "<p>가족 페이지에서 아래 초대코드를 입력해 연결을 완료해 주세요.</p>"
            f'<p style="font-size: 20px; font-weight: 700;">초대 코드: {escaped_invite_code}</p>'
            "<p><strong>Health Ladder</strong>에서는 가족과 함께 건강 기록과 건강관리 현황을 확인할 수 있습니다.</p>"
            f'<p><a href="{escaped_invite_url}" target="_blank" rel="noopener noreferrer">가족 페이지에서 초대코드 입력하기</a></p>'
            f"<p>초대 만료: {escaped_expires_at_text}</p>"
            "<p>초대코드는 발송 시점부터 30분 동안 유효합니다.</p>"
            "<p>가장 최근에 발송된 초대코드만 사용할 수 있습니다.</p>"
            "<p>회원이 아니신 경우 회원가입 또는 로그인 후 가족 페이지에서 초대코드를 입력해 주세요.</p>"
            "<p>감사합니다.<br><strong>Health Ladder</strong> 드림</p>"
        )
        return await self._send_email(email, subject, body, html_body=html_body)

    async def send_notification_email(
        self,
        to_email: str,
        title: str,
        message: str,
        action_url: str | None = None,
    ) -> bool:
        subject = f"[Health Ladder] {title}"
        body_parts = [
            "안녕하세요, Health Ladder입니다.",
            "",
            title,
            "",
            message,
        ]
        if action_url:
            body_parts.extend(["", f"알림 확인: {action_url}"])
        body_parts.extend(
            [
                "",
                "본 메일은 Health Ladder 서비스 알림 설정에 따라 발송되었습니다.",
                "",
                "Health Ladder 드림",
            ]
        )
        body = "\n".join(body_parts)

        escaped_title = escape(title)
        escaped_message = escape(message).replace("\n", "<br>")
        html_body = (
            "<p>안녕하세요, <strong>Health Ladder</strong>입니다.</p>"
            f"<p><strong>{escaped_title}</strong></p>"
            f"<p>{escaped_message}</p>"
        )
        if action_url:
            escaped_action_url = escape(action_url, quote=True)
            html_body += (
                f'<p><a href="{escaped_action_url}" target="_blank" rel="noopener noreferrer">'
                "알림 확인하기"
                "</a></p>"
            )
        html_body += (
            "<p>본 메일은 <strong>Health Ladder</strong> 서비스 알림 설정에 따라 발송되었습니다.</p>"
            "<p>감사합니다.<br><strong>Health Ladder</strong> 드림</p>"
        )
        return await self._send_email(to_email, subject, body, html_body=html_body)

    async def _send_email(self, recipient: str, subject: str, body: str, html_body: str | None = None) -> bool:
        if not config.EMAIL_ENABLED:
            if config.is_production:
                raise EmailConfigurationError("Email delivery is disabled.")
            return False
        if not self._is_configured():
            raise EmailConfigurationError("SMTP settings are incomplete.")

        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = formataddr((config.SMTP_FROM_NAME, config.SMTP_FROM_EMAIL or ""))
        message["To"] = recipient
        message.attach(MIMEText(body, "plain", "utf-8"))
        if html_body:
            message.attach(MIMEText(html_body, "html", "utf-8"))

        try:
            await asyncio.to_thread(self._send_smtp, message)
            return True
        except EmailDeliveryError:
            raise
        except Exception as exc:
            raise EmailDeliveryError("Failed to send email.") from exc

    def _send_smtp(self, message: Message) -> None:
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
        return has_smtp_config(config)
