import re


def normalize_phone_number(phone_number: str) -> str:
    """Normalize Korean mobile numbers to the local DB format: 01012345678."""
    digits = re.sub(r"\D", "", phone_number.strip())
    if digits.startswith("82"):
        digits = f"0{digits[2:]}"

    if not re.fullmatch(r"010\d{8}", digits):
        raise ValueError("유효하지 않은 휴대폰 번호 형식입니다.")

    return digits


def normalize_phone_number_e164(phone_number: str) -> str:
    """Normalize Korean mobile numbers to E.164 format for Twilio: +821012345678."""
    local_number = normalize_phone_number(phone_number)
    return f"+82{local_number[1:]}"
