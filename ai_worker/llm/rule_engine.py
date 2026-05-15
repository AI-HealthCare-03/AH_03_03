from dataclasses import dataclass

from ai_worker.llm.schemas import MainHealthChatbotInput, ResultChatbotInput


@dataclass
class RuleEngineResult:
    is_matched: bool
    intent: str
    response: str
    source: str = "rule_engine"


MEDICAL_CONSULT_KEYWORDS = [
    "약",
    "복용",
    "끊어도",
    "중단",
    "치료",
    "처방",
]


def has_medical_consult_keyword(user_message: str) -> bool:
    return any(keyword in user_message for keyword in MEDICAL_CONSULT_KEYWORDS)


def unmatched_rule_result(intent: str = "unknown_complex_question") -> RuleEngineResult:
    return RuleEngineResult(
        is_matched=False,
        intent=intent,
        response="",
        source="llm_fallback_stub",
    )


def try_result_chatbot_rule_engine(
    input_data: ResultChatbotInput,
) -> RuleEngineResult:
    user_message = input_data.user_message

    if has_medical_consult_keyword(user_message):
        return RuleEngineResult(
            is_matched=True,
            intent="medical_consult_required",
            response=(
                "약 복용, 중단, 치료 여부는 개인의 검사 결과와 진료 이력을 함께 확인해야 하므로 "
                "의료진과 상담해 결정하는 것이 좋습니다. 입력된 건강정보는 생활습관 관리 방향을 참고하는 용도로 활용해 주세요."
            ),
        )

    factor_names = [factor.name for factor in input_data.risk_factors]
    challenge_names = [challenge.name for challenge in input_data.recommended_challenges]

    factor_text = ", ".join(factor_names) if factor_names else "입력된 건강정보"
    challenge_text = ", ".join(challenge_names) if challenge_names else "생활습관 관리"

    if any(keyword in user_message for keyword in ["당뇨", "혈당"]):
        return RuleEngineResult(
            is_matched=True,
            intent="diabetes_result_guidance",
            response=(
                f"입력된 건강정보 기준으로 {factor_text} 항목이 혈당 관리와 관련될 수 있습니다. "
                f"{challenge_text} 챌린지는 혈당 변동을 줄이는 생활습관 관리에 도움이 될 수 있습니다."
            ),
        )

    if any(keyword in user_message for keyword in ["고혈압", "혈압"]):
        return RuleEngineResult(
            is_matched=True,
            intent="hypertension_result_guidance",
            response=(
                f"입력된 건강정보 기준으로 {factor_text} 항목이 혈압 관리와 관련될 수 있습니다. "
                f"{challenge_text} 챌린지는 혈압 관리에 도움이 될 수 있는 생활습관입니다."
            ),
        )

    if any(keyword in user_message for keyword in ["추천", "챌린지", "뭐 해야"]):
        return RuleEngineResult(
            is_matched=True,
            intent="challenge_recommendation",
            response=(
                f"입력된 건강정보 기준으로 {factor_text} 항목이 건강 관리에 영향을 줄 수 있습니다. "
                f"현재는 {challenge_text}부터 실천해 보는 것이 도움이 될 수 있습니다."
            ),
        )

    return unmatched_rule_result("result_chatbot_fallback")


def try_main_health_chatbot_rule_engine(
    input_data: MainHealthChatbotInput,
) -> RuleEngineResult:
    user_message = input_data.user_message

    if has_medical_consult_keyword(user_message):
        return RuleEngineResult(
            is_matched=True,
            intent="medical_consult_required",
            response=(
                "약 복용, 중단, 치료 여부는 일반 안내만으로 결정하기 어렵습니다. "
                "현재 상태와 진료 이력을 알고 있는 의료진과 상담해 결정하는 것이 좋습니다."
            ),
        )

    if any(keyword in user_message for keyword in ["고혈압", "혈압"]):
        return RuleEngineResult(
            is_matched=True,
            intent="hypertension_guidance",
            response=(
                "고혈압 관리는 혈관과 심장 부담을 줄이는 생활습관을 꾸준히 만드는 것이 중요합니다. "
                "짠 음식 줄이기, 규칙적인 유산소 운동, 체중 관리, 절주가 도움이 될 수 있습니다."
            ),
        )

    if any(keyword in user_message for keyword in ["당뇨", "혈당"]):
        return RuleEngineResult(
            is_matched=True,
            intent="diabetes_guidance",
            response=(
                "당뇨 관리는 혈당 변동을 줄이는 생활습관을 꾸준히 유지하는 것이 중요합니다. "
                "단 음료와 정제 탄수화물 섭취를 줄이고, 식후 가벼운 걷기를 실천하는 것이 도움이 될 수 있습니다."
            ),
        )

    if any(keyword in user_message for keyword in ["이상지질", "콜레스테롤", "중성지방"]):
        return RuleEngineResult(
            is_matched=True,
            intent="dyslipidemia_guidance",
            response=(
                "이상지질혈증 관리는 식사와 활동 습관을 함께 조절하는 방향이 좋습니다. "
                "포화지방과 튀긴 음식 섭취를 줄이고, 규칙적인 운동을 병행하는 것이 도움이 될 수 있습니다."
            ),
        )

    if any(keyword in user_message for keyword in ["비만", "BMI", "체중"]):
        return RuleEngineResult(
            is_matched=True,
            intent="obesity_guidance",
            response=(
                "비만 관리는 단기간 감량보다 지속 가능한 식사, 활동량, 수면 습관을 만드는 것이 중요합니다. "
                "단 음료와 야식을 줄이고, 걷기 같은 활동을 늘리는 것이 도움이 될 수 있습니다."
            ),
        )

    return unmatched_rule_result("main_health_chatbot_fallback")
