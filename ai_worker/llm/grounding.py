import re

HEALTH_FACTOR_KEYWORDS = [
    "혈압",
    "수축기 혈압",
    "이완기 혈압",
    "혈당",
    "공복혈당",
    "BMI",
    "체중",
    "비만",
    "콜레스테롤",
    "LDL",
    "HDL",
    "중성지방",
    "지질",
    "지질 수치",
]

CHALLENGE_KEYWORDS = [
    "단 음료 줄이기",
    "하루 7000보 걷기",
    "짠 음식 줄이기",
    "포화지방 줄이기",
    "현재 생활습관 유지하기",
    "식후 10분 걷기",
    "저염식",
    "운동",
    "스트레스 관리",
    "식습관",
    "생활습관",
]

GENERAL_ALLOWED_CHALLENGE_TERMS = {"생활습관"}


def check_result_chatbot_grounding(
    answer: str,
    allowed_factors: list[str],
    allowed_challenges: list[str],
    allowed_numbers: list[str] | None = None,
    allow_numeric_values: bool = False,
) -> dict:
    allowed_numbers_set = build_allowed_numbers_set(
        allowed_challenges=allowed_challenges,
        allowed_numbers=allowed_numbers,
        allow_numeric_values=allow_numeric_values,
    )

    ungrounded_terms = find_ungrounded_terms(
        answer=answer,
        allowed_factors=allowed_factors,
        allowed_challenges=allowed_challenges,
    )
    ungrounded_numbers = [number for number in extract_numbers(answer) if number not in allowed_numbers_set]
    notes = build_grounding_notes(answer)

    return {
        "is_grounded": not ungrounded_terms and not ungrounded_numbers,
        "ungrounded_terms": ungrounded_terms,
        "ungrounded_numbers": ungrounded_numbers,
        "notes": notes,
    }


def find_ungrounded_terms(
    answer: str,
    allowed_factors: list[str],
    allowed_challenges: list[str],
) -> list[str]:
    ungrounded_terms: list[str] = []

    for keyword in HEALTH_FACTOR_KEYWORDS:
        if keyword in answer and not is_allowed_keyword(keyword, allowed_factors):
            ungrounded_terms.append(keyword)

    for keyword in CHALLENGE_KEYWORDS:
        if keyword not in answer:
            continue
        if keyword in GENERAL_ALLOWED_CHALLENGE_TERMS:
            continue
        if not is_allowed_keyword(keyword, allowed_challenges):
            ungrounded_terms.append(keyword)

    return deduplicate(ungrounded_terms)


def is_allowed_keyword(keyword: str, allowed_terms: list[str]) -> bool:
    return any(keyword in allowed_term or allowed_term in keyword for allowed_term in allowed_terms)


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", text)


def build_allowed_numbers_set(
    allowed_challenges: list[str],
    allowed_numbers: list[str] | None,
    allow_numeric_values: bool,
) -> set[str]:
    allowed_numbers_set = set(extract_numbers(" ".join(allowed_challenges)))
    if allow_numeric_values:
        allowed_numbers_set.update(allowed_numbers or [])
    return allowed_numbers_set


def build_grounding_notes(answer: str) -> list[str]:
    notes: list[str] = []
    if "생활습관" in answer:
        notes.append("생활습관은 일반 건강관리 표현으로 허용했습니다.")
    return notes


def deduplicate(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)

    return result
