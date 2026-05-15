from ai_worker.llm.schemas import (
    ChallengeRecommendation,
    DiseasePrediction,
    DiseasePredictionSet,
    HealthRiskFactor,
    MainHealthChatbotInput,
    ResultChatbotInput,
)


def get_result_chatbot_dummy_cases() -> list[ResultChatbotInput]:
    return [
        ResultChatbotInput(
            user_message="당뇨 위험이 있다는데 뭘 해야 하나요?",
            risk_factors=[
                HealthRiskFactor(
                    name="공복혈당",
                    value=126,
                    reason="혈당 관리와 관련될 수 있습니다.",
                )
            ],
            recommended_challenges=[
                ChallengeRecommendation(
                    name="단 음료 줄이기",
                    reason="혈당 관리에 도움이 될 수 있습니다.",
                )
            ],
            tone="friendly",
        ),
        ResultChatbotInput(
            user_message="혈압이 높다는데 왜 걷기를 추천했나요?",
            risk_factors=[
                HealthRiskFactor(
                    name="수축기 혈압",
                    value=145,
                    reason="혈압 관리와 관련될 수 있습니다.",
                )
            ],
            recommended_challenges=[
                ChallengeRecommendation(
                    name="하루 7000보 걷기",
                    reason="혈압 관리에 도움이 될 수 있습니다.",
                )
            ],
            tone="friendly",
        ),
    ]


def get_main_health_chatbot_dummy_cases() -> list[MainHealthChatbotInput]:
    return [
        MainHealthChatbotInput(
            user_message="당뇨가 있으면 뭘 조심해야 하나요?",
            tone="friendly",
        ),
        MainHealthChatbotInput(
            user_message="혈압약 끊어도 되나요?",
            tone="friendly",
        ),
    ]


def get_disease_prediction_dummy_cases() -> list[DiseasePredictionSet]:
    return [
        DiseasePredictionSet(
            hypertension=DiseasePrediction(
                disease_name="hypertension",
                pred=0,
                probability=0.21,
            ),
            diabetes=DiseasePrediction(
                disease_name="diabetes",
                pred=0,
                probability=0.18,
            ),
            dyslipidemia=DiseasePrediction(
                disease_name="dyslipidemia",
                pred=0,
                probability=0.25,
            ),
            obesity=DiseasePrediction(
                disease_name="obesity",
                pred=0,
                probability=0.22,
            ),
        ),
        DiseasePredictionSet(
            hypertension=DiseasePrediction(
                disease_name="hypertension",
                pred=1,
                probability=0.81,
            ),
            diabetes=DiseasePrediction(
                disease_name="diabetes",
                pred=0,
                probability=0.34,
            ),
            dyslipidemia=DiseasePrediction(
                disease_name="dyslipidemia",
                pred=0,
                probability=0.28,
            ),
            obesity=DiseasePrediction(
                disease_name="obesity",
                pred=0,
                probability=0.39,
            ),
        ),
        DiseasePredictionSet(
            hypertension=DiseasePrediction(
                disease_name="hypertension",
                pred=0,
                probability=0.37,
            ),
            diabetes=DiseasePrediction(
                disease_name="diabetes",
                pred=1,
                probability=0.78,
            ),
            dyslipidemia=DiseasePrediction(
                disease_name="dyslipidemia",
                pred=0,
                probability=0.42,
            ),
            obesity=DiseasePrediction(
                disease_name="obesity",
                pred=1,
                probability=0.72,
            ),
        ),
        DiseasePredictionSet(
            hypertension=DiseasePrediction(
                disease_name="hypertension",
                pred=1,
                probability=0.86,
            ),
            diabetes=DiseasePrediction(
                disease_name="diabetes",
                pred=1,
                probability=0.79,
            ),
            dyslipidemia=DiseasePrediction(
                disease_name="dyslipidemia",
                pred=1,
                probability=0.74,
            ),
            obesity=DiseasePrediction(
                disease_name="obesity",
                pred=1,
                probability=0.83,
            ),
        ),
    ]
