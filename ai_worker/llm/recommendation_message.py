from ai_worker.llm.safety import check_medical_safety
from ai_worker.llm.schemas import (
    RecommendationMessageInput,
    RecommendationMessageOutput,
)


def generate_recommendation_message(
    input_data: RecommendationMessageInput,
) -> RecommendationMessageOutput:
    risk_factor_names = [factor.name for factor in input_data.risk_factors]
    challenge_names = [challenge.name for challenge in input_data.recommended_challenges]

    risk_text = ", ".join(risk_factor_names) if risk_factor_names else "입력된 건강정보"
    challenge_text = ", ".join(challenge_names) if challenge_names else "생활습관 관리"

    summary_message = (
        f"입력된 건강정보 기준으로 {risk_text}이/가 건강 관리에 영향을 준 것으로 보입니다. "
        "이를 고려하여 생활습관 관리가 도움이 될 수 있습니다."
    )

    challenge_message = f"{challenge_text}은/는 건강 관리에 도움이 될 수 있는 방법입니다."

    caution_message = (
        "이 정보는 진단이 아니며, 건강관리 참고용으로 활용하시기 바랍니다. "
        "정확한 진단과 치료는 의료진 상담이 필요합니다."
    )

    full_text = f"{summary_message} {challenge_message} {caution_message}"
    safety_result = check_medical_safety(full_text)

    return RecommendationMessageOutput(
        summary_message=summary_message,
        challenge_message=challenge_message,
        caution_message=caution_message,
        tone=input_data.tone,
        is_safe=safety_result["is_safe"],
    )
