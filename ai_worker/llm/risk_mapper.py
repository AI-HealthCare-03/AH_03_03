from ai_worker.llm.schemas import (
    ChallengeRecommendation,
    DiseasePredictionSet,
    HealthRiskFactor,
    RiskMappingResult,
)


def map_predictions_to_risk_context(
    predictions: DiseasePredictionSet,
) -> RiskMappingResult:
    risk_factors: list[HealthRiskFactor] = []
    challenges: list[ChallengeRecommendation] = []
    active_diseases: list[str] = []

    if predictions.hypertension.pred == 1:
        active_diseases.append("hypertension")
        risk_factors.append(
            HealthRiskFactor(
                name="혈압",
                value=predictions.hypertension.probability,
                reason="고혈압 위험 예측 확률을 기준으로 혈압 관리가 필요할 수 있습니다.",
            )
        )
        challenges.append(
            ChallengeRecommendation(
                name="짠 음식 줄이기",
                reason="나트륨 섭취 조절은 혈압 관리에 도움이 될 수 있습니다.",
            )
        )

    if predictions.diabetes.pred == 1:
        active_diseases.append("diabetes")
        risk_factors.append(
            HealthRiskFactor(
                name="혈당",
                value=predictions.diabetes.probability,
                reason="당뇨 위험 예측 확률을 기준으로 혈당 관리가 필요할 수 있습니다.",
            )
        )
        challenges.append(
            ChallengeRecommendation(
                name="단 음료 줄이기",
                reason="당류 섭취 조절은 혈당 관리에 도움이 될 수 있습니다.",
            )
        )

    if predictions.dyslipidemia.pred == 1:
        active_diseases.append("dyslipidemia")
        risk_factors.append(
            HealthRiskFactor(
                name="지질 수치",
                value=predictions.dyslipidemia.probability,
                reason="이상지질혈증 위험 예측 확률을 기준으로 지질 관리가 필요할 수 있습니다.",
            )
        )
        challenges.append(
            ChallengeRecommendation(
                name="포화지방 줄이기",
                reason="포화지방 섭취 조절은 지질 관리에 도움이 될 수 있습니다.",
            )
        )

    if predictions.obesity.pred == 1:
        active_diseases.append("obesity")
        risk_factors.append(
            HealthRiskFactor(
                name="BMI",
                value=predictions.obesity.probability,
                reason="비만 위험 예측 확률을 기준으로 체중 관리가 필요할 수 있습니다.",
            )
        )
        challenges.append(
            ChallengeRecommendation(
                name="하루 7000보 걷기",
                reason="체중 관리와 대사 건강 관리에 도움이 될 수 있습니다.",
            )
        )

    if not active_diseases:
        return RiskMappingResult(
            risk_group="low_risk",
            risk_factors=[],
            recommended_challenges=[
                ChallengeRecommendation(
                    name="현재 생활습관 유지하기",
                    reason="현재 예측 결과 기준으로 큰 위험 신호가 두드러지지 않습니다.",
                )
            ],
        )

    return RiskMappingResult(
        risk_group=f"{'_'.join(active_diseases)}_risk",
        risk_factors=risk_factors,
        recommended_challenges=deduplicate_challenges(challenges),
    )


def deduplicate_challenges(
    challenges: list[ChallengeRecommendation],
) -> list[ChallengeRecommendation]:
    seen: set[str] = set()
    result: list[ChallengeRecommendation] = []

    for challenge in challenges:
        if challenge.name in seen:
            continue
        seen.add(challenge.name)
        result.append(challenge)

    return result
