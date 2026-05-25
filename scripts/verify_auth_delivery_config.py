"""Inspect or explicitly test auth delivery providers without printing secrets.

Default mode only checks configuration status. Actual SMTP/Twilio calls require
explicit --send-email or --send-sms arguments.
"""

from __future__ import annotations

import argparse
import asyncio
import secrets
import sys
from collections.abc import Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify auth delivery provider configuration safely.")
    parser.add_argument("--send-email", metavar="EMAIL", help="Send a real SMTP test email to this address.")
    parser.add_argument("--send-sms", metavar="PHONE", help="Send a real Twilio Verify SMS to this phone number.")
    parser.add_argument(
        "--debug-twilio-ids",
        action="store_true",
        help="Print masked Twilio Account SID and Verify Service SID for account matching.",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> int:
    from fastapi import HTTPException

    from app.core import config
    from app.services.auth import AuthService
    from app.services.email_service import EmailService

    email_service = EmailService()
    print("Auth delivery configuration")
    print("===========================")
    print(f"ENV={config.ENV}")
    print(f"is_production={config.is_production}")
    print(f"EMAIL_ENABLED={config.EMAIL_ENABLED}")
    print(f"email_service_status={email_service.status()}")
    print(f"EMAIL_VERIFICATION_DEBUG={config.EMAIL_VERIFICATION_DEBUG}")
    print(f"PASSWORD_RESET_DEBUG={config.PASSWORD_RESET_DEBUG}")
    print(f"TWILIO_ENABLED={config.TWILIO_ENABLED}")
    print(f"twilio_verify_status={config.twilio_verify_status}")
    print(f"PHONE_VERIFICATION_DEBUG={config.PHONE_VERIFICATION_DEBUG}")
    print("secret_values=hidden")
    if args.debug_twilio_ids:
        print(f"TWILIO_ACCOUNT_SID_MASKED={_mask_twilio_sid(config.TWILIO_ACCOUNT_SID)}")
        print(f"TWILIO_VERIFY_SERVICE_SID_MASKED={_mask_twilio_sid(config.TWILIO_VERIFY_SERVICE_SID)}")
        print("TWILIO_AUTH_TOKEN_MASKED=hidden")
    if config.TWILIO_ENABLED and config.twilio_verify_status == "configured":
        print("twilio_trial_note=Korean live SMS may be blocked on Trial unless the recipient is verified.")

    failures = 0
    if args.send_email:
        if email_service.status() != "configured":
            print("[FAIL] SMTP is not configured; no email sent.")
            failures += 1
        else:
            try:
                code = f"{secrets.randbelow(1_000_000):06d}"
                await email_service.send_email_verification_code(args.send_email, code)
            except Exception as exc:  # noqa: BLE001 - smoke script should report provider failures.
                print(f"[FAIL] SMTP send failed: {type(exc).__name__}")
                failures += 1
            else:
                print("[OK] SMTP test email sent.")

    if args.send_sms:
        auth_service = AuthService()
        if config.twilio_verify_status != "configured":
            print("[FAIL] Twilio Verify is not configured; no SMS sent.")
            failures += 1
        else:
            try:
                normalized_phone = auth_service._normalize_phone_for_e164(args.send_sms)
                print(f"Twilio test recipient={_mask_phone(normalized_phone)}")
                await auth_service._send_twilio_verification(normalized_phone)
            except HTTPException as exc:
                print(f"[FAIL] Twilio Verify send failed: HTTPException status_code={exc.status_code}")
                for line in _format_http_exception_detail(exc.detail):
                    print(f"  {line}")
                _print_twilio_trial_hints()
                failures += 1
            except Exception as exc:  # noqa: BLE001 - smoke script should report provider failures.
                print(f"[FAIL] Twilio Verify send failed: {type(exc).__name__}")
                _print_twilio_trial_hints()
                failures += 1
            else:
                print("[OK] Twilio Verify SMS request sent.")

    if not args.send_email and not args.send_sms:
        print("No external delivery call was made. Use --send-email or --send-sms for an explicit live test.")
    return 1 if failures else 0


def _format_http_exception_detail(detail: object) -> list[str]:
    if isinstance(detail, Mapping):
        safe_keys = (
            "message",
            "provider",
            "provider_status",
            "provider_code",
            "provider_message",
            "provider_more_info",
        )
        return [f"{key}={detail[key]}" for key in safe_keys if key in detail and detail[key] is not None]
    return [f"detail={detail}"]


def _print_twilio_trial_hints() -> None:
    print("  Common Twilio Verify causes to check:")
    print("  - Trial account recipient is not a verified recipient")
    print("  - Geo Permissions block sending to the target country")
    print("  - Verify Service SID is invalid or belongs to another project")
    print("  - SMS channel is disabled for the Verify service")
    print("  - Trial/account permission or balance is insufficient")


def _mask_phone(phone_number: str) -> str:
    digits = "".join(char for char in phone_number if char.isdigit())
    if len(digits) <= 4:
        return "***"
    return f"***{digits[-4:]}"


def _mask_twilio_sid(value: str | None) -> str:
    if not value:
        return "MISSING"
    prefix = value[:2]
    suffix = value[-4:] if len(value) >= 4 else "***"
    return f"{prefix}****{suffix}"


if __name__ == "__main__":
    raise SystemExit(main())
