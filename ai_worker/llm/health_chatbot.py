from ai_worker.llm.rule_engine import (
    try_main_health_chatbot_rule_engine,
    try_result_chatbot_rule_engine,
)
from ai_worker.llm.safety import check_medical_safety
from ai_worker.llm.schemas import (
    HealthChatbotInput,
    HealthChatbotOutput,
    MainHealthChatbotInput,
    MainHealthChatbotOutput,
    ResultChatbotInput,
    ResultChatbotOutput,
)

CAUTION_MESSAGE = "이 정보는 진단이 아니며, 정확한 진단과 치료는 의료진 상담이 필요합니다."


def infer_result_chatbot_intent(user_message: str) -> str:
    if any(keyword in user_message for keyword in ["뭐 해야", "추천", "챌린지"]):
        return "challenge_recommendation"

    if any(keyword in user_message for keyword in ["수치", "혈당", "혈압", "BMI", "콜레스테롤"]):
        return "health_metric_explanation"

    if any(keyword in user_message for keyword in ["위험", "상태", "괜찮"]):
        return "health_status_summary"

    return "general_health_guidance"


def infer_main_health_chatbot_intent(user_message: str) -> str:
    if any(keyword in user_message for keyword in ["고혈압", "혈압"]):
        return "hypertension_guidance"

    if any(keyword in user_message for keyword in ["당뇨", "혈당"]):
        return "diabetes_guidance"

    if any(keyword in user_message for keyword in ["이상지질혈증", "콜레스테롤", "중성지방"]):
        return "dyslipidemia_guidance"

    if any(keyword in user_message for keyword in ["비만", "체중", "BMI"]):
        return "obesity_guidance"

    return "chronic_disease_prevention"


def generate_result_chatbot_response(
    input_data: ResultChatbotInput,
) -> ResultChatbotOutput:
    factor_names = [factor.name for factor in input_data.risk_factors]
    rule_result = try_result_chatbot_rule_engine(input_data)

    if rule_result.is_matched:
        intent = rule_result.intent
        answer = rule_result.response
        source = rule_result.source
    else:
        intent = infer_result_chatbot_intent(input_data.user_message)
        challenge_names = [challenge.name for challenge in input_data.recommended_challenges]

        factor_text = ", ".join(factor_names) if factor_names else "입력된 건강정보"
        challenge_text = ", ".join(challenge_names) if challenge_names else "생활습관 관리"

        if intent == "challenge_recommendation":
            answer = (
                f"입력된 건강정보 기준으로 {factor_text}이/가 건강 관리에 영향을 줄 수 있습니다. "
                f"현재는 {challenge_text} 같은 챌린지가 도움이 될 수 있습니다. "
                "무리하게 한 번에 바꾸기보다는 실천 가능한 목표부터 시작하는 것이 좋습니다."
            )

        elif intent == "health_metric_explanation":
            answer = (
                f"질문하신 내용은 건강검진 수치 해석과 관련된 것으로 보입니다. "
                f"입력된 건강정보 기준으로 {factor_text}이/가 생활습관 관리와 관련될 수 있습니다. "
                "각 수치는 단독으로 확정 판단하기보다 다른 검사결과와 함께 보는 것이 좋습니다."
            )

        elif intent == "health_status_summary":
            answer = (
                f"입력된 건강정보 기준으로 {factor_text}이/가 현재 건강 관리에 영향을 준 것으로 보입니다. "
                "다만 이 결과만으로 질환 여부를 확정할 수는 없으며, 생활습관 관리와 추가 확인이 필요할 수 있습니다."
            )

        else:
            answer = (
                "건강 관리는 식사, 운동, 수면, 음주, 흡연 여부 같은 생활습관이 함께 영향을 줍니다. "
                f"입력된 건강정보 기준으로는 {factor_text}을/를 참고하여 관리 방향을 잡을 수 있습니다."
            )
        source = rule_result.source

    final_answer = f"{answer} {CAUTION_MESSAGE}"
    safety_result = check_medical_safety(final_answer)

    return ResultChatbotOutput(
        answer=final_answer,
        intent=intent,
        source=source,
        referenced_health_factors=factor_names,
        recommended_challenges=input_data.recommended_challenges,
        caution_message=CAUTION_MESSAGE,
        tone=input_data.tone,
        is_safe=safety_result["is_safe"],
        safety_result=safety_result,
    )


def generate_main_health_chatbot_response(
    input_data: MainHealthChatbotInput,
) -> MainHealthChatbotOutput:
    rule_result = try_main_health_chatbot_rule_engine(input_data)

    if rule_result.is_matched:
        intent = rule_result.intent
        answer = rule_result.response
        source = rule_result.source
    else:
        intent = infer_main_health_chatbot_intent(input_data.user_message)

        if intent == "hypertension_guidance":
            answer = (
                "고혈압은 혈관과 심장에 부담을 줄 수 있어 꾸준한 생활습관 관리가 중요합니다. "
                "짠 음식 줄이기, 규칙적인 유산소 운동, 체중 관리, 절주, 충분한 수면이 도움이 될 수 있습니다."
            )

        elif intent == "diabetes_guidance":
            answer = (
                "당뇨 관리는 혈당 변동을 줄이는 생활습관을 꾸준히 만드는 것이 중요합니다. "
                "단 음료와 정제 탄수화물 섭취를 줄이고, 식사 후 가벼운 걷기와 규칙적인 운동을 실천하는 것이 도움이 될 수 있습니다."
            )

        elif intent == "dyslipidemia_guidance":
            answer = (
                "이상지질혈증 관리는 혈중 지질 수치에 영향을 줄 수 있는 식사와 활동 습관을 함께 보는 것이 좋습니다. "
                "튀긴 음식과 포화지방 섭취를 줄이고, 채소와 불포화지방을 적절히 섭취하며, 규칙적인 운동을 병행하는 것이 도움이 될 수 있습니다."
            )

        elif intent == "obesity_guidance":
            answer = (
                "비만 관리는 단기간 감량보다 지속 가능한 식사, 활동량, 수면 습관을 만드는 방향이 좋습니다. "
                "무리한 제한보다는 식사량을 기록하고, 걷기 같은 활동을 늘리며, 야식과 단 음료를 줄이는 것이 도움이 될 수 있습니다."
            )

        else:
            answer = (
                "만성질환 예방에는 식사, 운동, 수면, 음주, 흡연, 스트레스 관리가 함께 영향을 줍니다. "
                "채소와 단백질을 포함한 균형 잡힌 식사, 규칙적인 신체활동, 정기적인 건강검진이 도움이 될 수 있습니다."
            )
        source = rule_result.source

    final_answer = f"{answer} {CAUTION_MESSAGE}"
    safety_result = check_medical_safety(final_answer)

    return MainHealthChatbotOutput(
        answer=final_answer,
        intent=intent,
        source=source,
        caution_message=CAUTION_MESSAGE,
        tone=input_data.tone,
        is_safe=safety_result["is_safe"],
        safety_result=safety_result,
    )


def infer_intent(user_message: str) -> str:
    return infer_result_chatbot_intent(user_message)


def generate_health_chatbot_response(
    input_data: HealthChatbotInput,
) -> HealthChatbotOutput:
    return generate_result_chatbot_response(input_data)
