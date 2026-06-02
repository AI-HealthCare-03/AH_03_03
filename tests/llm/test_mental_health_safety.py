from ai_runtime.llm.response_router import route_result_chatbot_response
from ai_runtime.llm.safety import detect_mental_health_safety
from ai_runtime.llm.schemas import ChallengeRecommendation, ResultChatbotInput


def test_detect_mental_health_crisis_keywords() -> None:
    result = detect_mental_health_safety("자해 생각이 들어요")

    assert result is not None
    assert result.level == "crisis"
    assert result.intent == "mental_health_crisis_support"
    assert "전문기관" in result.response


def test_detect_mental_health_support_keywords() -> None:
    result = detect_mental_health_safety("요즘 번아웃 때문에 무기력합니다")

    assert result is not None
    assert result.level == "professional_support"
    assert "전문 상담" in result.response


def test_result_chatbot_crisis_response_does_not_return_challenge_recommendations() -> None:
    response = route_result_chatbot_response(
        ResultChatbotInput(
            user_message="죽고 싶다는 생각이 듭니다",
            recommended_challenges=[
                ChallengeRecommendation(name="걷기 챌린지", reason="활동량 관리"),
            ],
        ),
        use_llm_rewrite=True,
        use_real_llm=True,
    )

    assert response.source == "safety_policy"
    assert response.intent == "mental_health_crisis_support"
    assert response.recommended_challenges == []
    assert "챌린지 추천보다 안전 확보가 우선" in response.answer
