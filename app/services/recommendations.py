from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from app.core import config
from app.models.analysis import AnalysisType, RiskLevel
from app.models.challenges import UserChallengeStatus
from app.services import analysis as analysis_service
from app.services import challenges as challenge_service
from app.services import diets as diet_service
from app.services import health as health_service

MAX_TODAY_RECOMMENDATIONS = 3


@dataclass(frozen=True)
class RuleRecommendation:
    title: str
    description: str
    reason: str
    action_type: str
    related_disease: str
    priority: int

    def to_response(self) -> dict[str, object]:
        return {
            "title": self.title,
            "description": self.description,
            "reason": self.reason,
            "action_type": self.action_type,
            "related_disease": self.related_disease,
            "priority": self.priority,
        }


async def get_today_recommendations(user_id: int) -> dict[str, Any]:
    today = datetime.now(config.TIMEZONE).date()
    latest_analysis_results = await analysis_service.list_latest_analysis_results(user_id)
    latest_health_record = await health_service.get_latest_health_record(user_id)
    user_challenges = await challenge_service.list_user_challenges(user_id, limit=20)
    diet_records = await diet_service.list_diet_records(user_id, limit=5)

    candidates = _build_rule_based_recommendations(
        today=today,
        latest_analysis_results=latest_analysis_results,
        latest_health_record=latest_health_record,
        user_challenges=user_challenges,
        diet_records=diet_records,
    )
    items = _dedupe_and_rank(candidates)[:MAX_TODAY_RECOMMENDATIONS]
    return {
        "date": today,
        "items": [item.to_response() for item in items],
    }


def _build_rule_based_recommendations(
    *,
    today: date,
    latest_analysis_results: list[Any],
    latest_health_record: Any | None,
    user_challenges: list[Any],
    diet_records: list[Any],
) -> list[RuleRecommendation]:
    candidates: list[RuleRecommendation] = []
    candidates.extend(_analysis_based_recommendations(latest_analysis_results))
    candidates.extend(_health_record_recommendations(latest_health_record))
    candidates.extend(_diet_recommendations(today, diet_records))
    candidates.extend(_challenge_recommendations(user_challenges))

    if candidates:
        return candidates

    return [
        RuleRecommendation(
            title="오늘 건강 기록 1개 남기기",
            description="혈압, 식사, 걷기 중 하나를 선택해 오늘 상태를 기록해보세요.",
            reason="기록이 쌓이면 위험 관리와 생활습관 변화 추이를 더 쉽게 확인할 수 있습니다.",
            action_type="tracking",
            related_disease="GENERAL",
            priority=90,
        )
    ]


def _analysis_based_recommendations(latest_analysis_results: list[Any]) -> list[RuleRecommendation]:
    recommendations: list[RuleRecommendation] = []
    for result in latest_analysis_results:
        risk_level = getattr(result, "risk_level", None)
        if _risk_level_priority(risk_level) < _risk_level_priority(RiskLevel.ATTENTION):
            continue
        disease_type = getattr(result, "analysis_type", None)
        recommendation = _disease_recommendation(disease_type, risk_level)
        if recommendation is not None:
            recommendations.append(recommendation)
    return recommendations


def _disease_recommendation(
    disease_type: AnalysisType | str | None, risk_level: RiskLevel | str
) -> RuleRecommendation | None:
    risk_label = _risk_level_recommendation_label(risk_level)
    disease = disease_type.value if hasattr(disease_type, "value") else str(disease_type or "GENERAL")
    priority = _risk_level_recommendation_priority(risk_level)

    if disease == AnalysisType.DIABETES.value:
        return RuleRecommendation(
            title="식후 10분 가볍게 움직이기",
            description="오늘 한 끼 식사 후 바로 앉기보다 10분 정도 천천히 걸어보세요.",
            reason=f"최근 분석에서 혈당 관련 위험 관리 필요도가 {risk_label} 나타났습니다.",
            action_type="movement",
            related_disease=AnalysisType.DIABETES.value,
            priority=priority,
        )
    if disease == AnalysisType.HYPERTENSION.value:
        return RuleRecommendation(
            title="오늘 한 끼는 싱겁게 선택하기",
            description="국물은 적게 먹고, 짠 반찬은 양을 줄여 혈압 관리 습관을 만들어보세요.",
            reason=f"최근 분석에서 혈압 관련 위험 관리 필요도가 {risk_label} 나타났습니다.",
            action_type="diet",
            related_disease=AnalysisType.HYPERTENSION.value,
            priority=priority,
        )
    if disease == AnalysisType.DYSLIPIDEMIA.value:
        return RuleRecommendation(
            title="채소 한 접시 먼저 먹기",
            description="오늘 식사에서 채소를 먼저 먹고 튀김이나 기름진 음식은 조금 줄여보세요.",
            reason=f"최근 분석에서 콜레스테롤·중성지방 관리 필요도가 {risk_label} 나타났습니다.",
            action_type="diet",
            related_disease=AnalysisType.DYSLIPIDEMIA.value,
            priority=priority,
        )
    if disease == AnalysisType.OBESITY.value:
        return RuleRecommendation(
            title="30분 걷기 시간 확보하기",
            description="한 번에 어렵다면 10분씩 나누어 걸어도 좋습니다.",
            reason=f"최근 분석에서 체중 관리 필요도가 {risk_label} 나타났습니다.",
            action_type="movement",
            related_disease=AnalysisType.OBESITY.value,
            priority=priority,
        )
    return None


def _risk_level_priority(risk_level: RiskLevel | str | None) -> int:
    value = getattr(risk_level, "value", risk_level)
    return {
        RiskLevel.LOW.value: 0,
        RiskLevel.ATTENTION.value: 1,
        RiskLevel.CAUTION.value: 2,
        RiskLevel.HIGH_CAUTION.value: 3,
        "MEDIUM": 2,
        "HIGH": 3,
    }.get(str(value), 0)


def _risk_level_recommendation_label(risk_level: RiskLevel | str) -> str:
    value = getattr(risk_level, "value", risk_level)
    return {
        RiskLevel.ATTENTION.value: "관심 필요로",
        RiskLevel.CAUTION.value: "주의 단계로",
        RiskLevel.HIGH_CAUTION.value: "높은 주의 단계로",
        "MEDIUM": "주의 단계로",
        "HIGH": "높은 주의 단계로",
    }.get(str(value), "관리 필요로")


def _risk_level_recommendation_priority(risk_level: RiskLevel | str) -> int:
    priority = _risk_level_priority(risk_level)
    if priority >= 3:
        return 10
    if priority == 2:
        return 15
    return 20


def _health_record_recommendations(latest_health_record: Any | None) -> list[RuleRecommendation]:
    if latest_health_record is None:
        return [
            RuleRecommendation(
                title="건강정보 업데이트하기",
                description="키, 몸무게, 혈압 같은 기본 정보를 입력해 오늘 상태를 점검해보세요.",
                reason="최신 건강정보가 있으면 맞춤형 위험 관리 안내가 더 정확해집니다.",
                action_type="tracking",
                related_disease="GENERAL",
                priority=30,
            )
        ]

    recommendations: list[RuleRecommendation] = []
    systolic = _to_float(getattr(latest_health_record, "systolic_bp", None))
    diastolic = _to_float(getattr(latest_health_record, "diastolic_bp", None))
    fasting_glucose = _to_float(getattr(latest_health_record, "fasting_glucose", None))
    walking_days = _to_float(getattr(latest_health_record, "walking_days_per_week", None))

    if (systolic is not None and systolic >= 130) or (diastolic is not None and diastolic >= 80):
        recommendations.append(
            RuleRecommendation(
                title="혈압 기록 시간 정하기",
                description="오늘은 같은 시간대에 혈압을 한 번 더 기록하고, 평소와 다른지 확인해보세요.",
                reason="최근 혈압 수치가 생활습관 관리가 필요한 범위에 가까울 수 있습니다.",
                action_type="tracking",
                related_disease=AnalysisType.HYPERTENSION.value,
                priority=12,
            )
        )

    if fasting_glucose is not None and fasting_glucose >= 100:
        recommendations.append(
            RuleRecommendation(
                title="단 음료 대신 물 선택하기",
                description="오늘은 단 음료를 줄이고 물이나 무가당 음료를 선택해보세요.",
                reason="최근 공복혈당 기록을 기준으로 혈당 위험 관리 습관이 도움이 될 수 있습니다.",
                action_type="diet",
                related_disease=AnalysisType.DIABETES.value,
                priority=14,
            )
        )

    if walking_days is not None and walking_days < 3:
        recommendations.append(
            RuleRecommendation(
                title="가벼운 걷기 20분 채우기",
                description="출퇴근길이나 식후 시간을 활용해 부담 없는 걷기부터 시작해보세요.",
                reason="최근 걷기 일수가 적어 활동량을 조금 늘리는 생활습관 개선이 필요할 수 있습니다.",
                action_type="movement",
                related_disease="GENERAL",
                priority=28,
            )
        )
    return recommendations


def _diet_recommendations(today: date, diet_records: list[Any]) -> list[RuleRecommendation]:
    latest_diet = diet_records[0] if diet_records else None
    if latest_diet is None:
        return [
            RuleRecommendation(
                title="오늘 식단 사진 남기기",
                description="한 끼만 기록해도 식사 패턴을 돌아보는 데 도움이 됩니다.",
                reason="최근 식단 기록이 없어 오늘 식사 구성을 확인하기 어렵습니다.",
                action_type="diet_record",
                related_disease="GENERAL",
                priority=40,
            )
        ]

    created_at = getattr(latest_diet, "created_at", None)
    if isinstance(created_at, datetime) and created_at.astimezone(config.TIMEZONE).date() < today - timedelta(days=2):
        return [
            RuleRecommendation(
                title="최근 식단 다시 기록하기",
                description="오늘 먹은 음식을 간단히 기록하고 식사 균형을 확인해보세요.",
                reason="최근 며칠간 식단 기록이 부족해 식습관 변화를 확인하기 어렵습니다.",
                action_type="diet_record",
                related_disease="GENERAL",
                priority=42,
            )
        ]

    diet_score = _to_float(getattr(latest_diet, "diet_score", None))
    if diet_score is not None and diet_score < 70:
        return [
            RuleRecommendation(
                title="다음 식사에 단백질과 채소 보강하기",
                description="밥이나 면만 먹기보다 단백질 반찬과 채소를 함께 챙겨보세요.",
                reason="최근 식단 점수를 기준으로 식사 구성 개선 여지가 있습니다.",
                action_type="diet",
                related_disease="GENERAL",
                priority=32,
            )
        ]
    return []


def _challenge_recommendations(user_challenges: list[Any]) -> list[RuleRecommendation]:
    joined = [
        challenge
        for challenge in user_challenges
        if getattr(challenge, "status", None) == UserChallengeStatus.JOINED
        or str(getattr(challenge, "status", "")).upper() == UserChallengeStatus.JOINED.value
    ]
    if not joined:
        return [
            RuleRecommendation(
                title="생활습관 챌린지 1개 시작하기",
                description="걷기, 식사, 수분 섭취처럼 부담이 적은 챌린지부터 선택해보세요.",
                reason="진행 중인 챌린지가 없으면 예방 행동을 꾸준히 이어가기 어렵습니다.",
                action_type="challenge",
                related_disease="GENERAL",
                priority=36,
            )
        ]

    if any(not bool(getattr(challenge, "today_completed", False)) for challenge in joined):
        return [
            RuleRecommendation(
                title="오늘 챌린지 기록하기",
                description="진행 중인 챌린지의 오늘 수행 여부를 확인하고 기록해보세요.",
                reason="작은 실천을 매일 기록하면 생활습관 개선 흐름을 유지하기 좋습니다.",
                action_type="challenge",
                related_disease="GENERAL",
                priority=34,
            )
        ]
    return []


def _dedupe_and_rank(items: list[RuleRecommendation]) -> list[RuleRecommendation]:
    deduped: dict[tuple[str, str], RuleRecommendation] = {}
    for item in items:
        key = (item.action_type, item.related_disease)
        current = deduped.get(key)
        if current is None or item.priority < current.priority:
            deduped[key] = item
    return sorted(deduped.values(), key=lambda item: (item.priority, item.title, item.action_type))


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
