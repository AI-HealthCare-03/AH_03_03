"""Inspect or explicitly test email auth delivery without printing secrets.

Default mode only checks configuration status. Actual SMTP calls require an
explicit --send-email argument.
"""

from __future__ import annotations

import argparse
import asyncio
import secrets
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify auth delivery provider configuration safely.")
    parser.add_argument("--send-email", metavar="EMAIL", help="Send a real SMTP test email to this address.")
    args = parser.parse_args()
    return asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> int:
    from app.core import config
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
    print("phone_verification_status=deferred_from_mvp")
    print("required_auth_delivery=email_verification")
    print("secret_values=hidden")
    if not config.is_production and not config.EMAIL_ENABLED and config.EMAIL_VERIFICATION_DEBUG:
        print("demo_email_verification=debug_code_response_enabled")
    elif not config.is_production and not config.EMAIL_ENABLED:
        print("demo_email_verification=needs_EMAIL_VERIFICATION_DEBUG_true_or_SMTP")

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

    if not args.send_email:
        print("No external delivery call was made. Use --send-email for an explicit live SMTP test.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
