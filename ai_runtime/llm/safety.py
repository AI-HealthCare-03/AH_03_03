from dataclasses import dataclass

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

MENTAL_HEALTH_CRISIS_KEYWORDS = [
    "자해",
    "극단 선택",
    "극단적 선택",
    "죽고 싶",
    "죽고싶",
    "삶을 끝",
    "목숨을 끊",
    "스스로 해치",
    "suicide",
    "self-harm",
]

MENTAL_HEALTH_SUPPORT_KEYWORDS = [
    "우울",
    "무기력",
    "번아웃",
    "burnout",
    "depress",
]

MENTAL_HEALTH_SELF_CARE_KEYWORDS = [
    "스트레스",
    "불안",
    "수면",
    "잠이 안",
    "불면",
    "긴장",
    "stress",
    "anxiety",
    "sleep",
]


@dataclass(frozen=True)
class MentalHealthSafetyResult:
    level: str
    intent: str
    response: str


def check_medical_safety(text: str) -> dict:
    detected_forbidden = [keyword for keyword in FORBIDDEN_KEYWORDS if keyword in text]

    missing_required_keywords = [keyword for keyword in REQUIRED_KEYWORDS if keyword not in text]

    return {
        "is_safe": len(detected_forbidden) == 0 and len(missing_required_keywords) == 0,
        "detected_forbidden": detected_forbidden,
        "missing_required_keywords": missing_required_keywords,
    }


def detect_mental_health_safety(user_message: str) -> MentalHealthSafetyResult | None:
    normalized_message = user_message.lower()

    if _contains_any(normalized_message, MENTAL_HEALTH_CRISIS_KEYWORDS):
        return MentalHealthSafetyResult(
            level="crisis",
            intent="mental_health_crisis_support",
            response=(
                "정신건강 관련 위기 신호가 포함될 수 있어 지금은 챌린지 추천보다 안전 확보가 우선입니다. "
                "혼자 버티지 말고 가까운 보호자나 신뢰할 수 있는 사람에게 바로 알려 주세요. "
                "당장 위험하다고 느껴지면 119, 112, 가까운 응급실 또는 정신건강 위기 상담 등 전문기관의 도움을 받아 주세요."
            ),
        )

    if _contains_any(normalized_message, MENTAL_HEALTH_SUPPORT_KEYWORDS):
        return MentalHealthSafetyResult(
            level="professional_support",
            intent="mental_health_professional_support",
            response=(
                "우울감, 무기력, 번아웃처럼 느껴지는 상태는 생활습관만으로 판단하기 어렵습니다. "
                "수면, 식사, 가벼운 움직임 같은 자기관리 챌린지를 작게 시작해 볼 수 있지만, "
                "상태가 이어지거나 일상 기능에 영향을 준다면 전문 상담이나 의료진 상담을 권장합니다."
            ),
        )

    if _contains_any(normalized_message, MENTAL_HEALTH_SELF_CARE_KEYWORDS):
        return MentalHealthSafetyResult(
            level="self_care",
            intent="mental_health_self_care_guidance",
            response=(
                "스트레스, 불안, 수면 문제는 일상 리듬과 생활습관의 영향을 받을 수 있습니다. "
                "짧은 호흡, 수면 시간 기록, 가벼운 산책처럼 부담이 낮은 정신건강 관련 자기관리 챌린지부터 시도해 볼 수 있습니다. "
                "증상이 심해지거나 오래 지속되면 전문 상담을 함께 고려해 주세요."
            ),
        )

    return None


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword.lower() in text for keyword in keywords)
