"""Send a notification email smoke only when explicitly confirmed.

Default mode is dry-run and never opens an SMTP connection.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test Health Ladder notification email delivery.")
    parser.add_argument("--confirm-send", action="store_true", help="Actually send the email.")
    parser.add_argument("--to-email", default=None, help="Recipient email. Defaults to TEST_NOTIFICATION_EMAIL.")
    parser.add_argument("--user-id", type=int, default=None, help="Also create an in-app notification for this user.")
    parser.add_argument("--title", default="알림 이메일 스모크")
    parser.add_argument("--message", default="Health Ladder 일반 알림 이메일 발송 확인용 메시지입니다.")
    parser.add_argument("--action-url", default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> int:
    recipient = args.to_email or os.getenv("TEST_NOTIFICATION_EMAIL")
    if not args.confirm_send:
        print("DRY_RUN_ONLY=true")
        print("No email was sent. Re-run with --confirm-send to send a live SMTP smoke.")
        return 0

    if args.user_id is None and not recipient:
        print("[FAIL] TEST_NOTIFICATION_EMAIL or --to-email is required when --user-id is not provided.")
        return 1

    if args.user_id is not None:
        return await _run_with_notification_record(args)
    return await _send_direct_email(recipient, args)


async def _send_direct_email(recipient: str, args: argparse.Namespace) -> int:
    from app.services.email_service import EmailService

    try:
        sent = await EmailService().send_notification_email(
            recipient,
            title=args.title,
            message=args.message,
            action_url=args.action_url,
        )
    except Exception as exc:  # noqa: BLE001 - smoke should report provider failures without secrets.
        print(f"[FAIL] notification email send failed: {type(exc).__name__}")
        return 1

    print(f"[OK] notification email sent={sent}")
    return 0 if sent else 1


async def _run_with_notification_record(args: argparse.Namespace) -> int:
    from tortoise import Tortoise

    from app.core.db.databases import TORTOISE_ORM
    from app.dtos.notifications import NotificationCreateRequest
    from app.models.notifications import NotificationChannel
    from app.services import notification_email as notification_email_service
    from app.services import notifications as notification_service

    await Tortoise.init(config=TORTOISE_ORM)
    try:
        notification = await notification_service.create_notification(
            args.user_id,
            NotificationCreateRequest(
                notification_type="SYSTEM",
                title=args.title,
                message=args.message,
                send_email=False,
            ),
        )
        delivery_result = await notification_email_service.deliver_notification_email_to_user(
            user_id=args.user_id,
            title=args.title,
            message=args.message,
            action_url=args.action_url,
        )
        await notification_service.record_notification_log(
            user_id=args.user_id,
            notification_id=int(notification.id),
            notification_type=notification.notification_type,
            channel=NotificationChannel.EMAIL,
            title=notification.title,
            message_summary=notification.message[:255],
            related_type=notification.related_type,
            related_id=notification.related_id,
            status=delivery_result.status,
            provider=delivery_result.provider,
            error_code=delivery_result.error_code,
            error_message=delivery_result.error_message,
            sent_at=delivery_result.sent_at,
            failed_at=delivery_result.failed_at,
        )
    finally:
        await Tortoise.close_connections()

    print(f"[OK] notification_id={int(notification.id)} email_status={delivery_result.status.value}")
    return 0 if delivery_result.sent else 1


if __name__ == "__main__":
    raise SystemExit(main())
