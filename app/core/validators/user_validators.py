import re
from datetime import date, datetime

from dateutil.relativedelta import relativedelta

from app.core import config
from app.core.utils.common import normalize_phone_number


def validate_password(password: str) -> str:
    if len(password) < 8:
        raise ValueError("비밀번호는 8자 이상이어야 합니다.")

    # 영문자를 포함하고 있는지
    if not re.search(r"[A-Za-z]", password):
        raise ValueError("비밀번호에는 영문자, 숫자, 특수문자가 각 하나 이상 포함되어야 합니다.")

    # 숫자를 포함하고 있는지
    if not re.search(r"[0-9]", password):
        raise ValueError("비밀번호에는 영문자, 숫자, 특수문자가 각 하나 이상 포함되어야 합니다.")

    # 특수문자를 포함하고 있는지
    if not re.search(r"[^a-zA-Z0-9]", password):
        raise ValueError("비밀번호에는 영문자, 숫자, 특수문자가 각 하나 이상 포함되어야 합니다.")

    return password


def validate_phone_number(phone_number: str) -> str:
    try:
        normalize_phone_number(phone_number)
    except ValueError as exc:
        raise ValueError("유효하지 않은 휴대폰 번호 형식입니다.") from exc

    return phone_number


def validate_birthday(birthday: date | str) -> date:
    if isinstance(birthday, str):
        try:
            birthday = date.fromisoformat(birthday)
        except ValueError as e:
            raise ValueError("올바르지 않은 날짜 형식입니다. format: YYYY-MM-DD") from e

    is_over_14 = birthday < datetime.now(tz=config.TIMEZONE).date() - relativedelta(years=14)
    if not is_over_14:
        raise ValueError("서비스 약관에 따라 만14세 미만은 회원가입이 불가합니다.")

    return birthday
