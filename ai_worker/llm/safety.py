FORBIDDEN_KEYWORDS = [
    "진단되었습니다",
    "확진",
    "치료하세요",
    "복용하세요",
    "약을 끊으세요",
    "처방",
    "반드시 치료",
]

REQUIRED_KEYWORDS = [
    "진단이 아니",
    "의료진 상담",
]


def check_medical_safety(text: str) -> dict:
    detected_forbidden = [keyword for keyword in FORBIDDEN_KEYWORDS if keyword in text]

    missing_required_keywords = [keyword for keyword in REQUIRED_KEYWORDS if keyword not in text]

    return {
        "is_safe": len(detected_forbidden) == 0 and len(missing_required_keywords) == 0,
        "detected_forbidden": detected_forbidden,
        "missing_required_keywords": missing_required_keywords,
    }
