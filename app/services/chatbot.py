import re
from typing import Any

from ai_runtime.llm.graph import run_chatbot_graph_async
from app.core import config
from app.core.providers import has_openai_config
from app.dtos.chatbot import ChatbotAskRequest, ChatbotAskResponse
from app.models.challenges import ChallengeStatus, UserChallengeStatus
from app.repositories import analysis_repository, health_repository
from app.services import analysis as analysis_service
from app.services import challenges as challenge_service
from app.services import diet_recommendations
from app.services import diets as diet_service

SAFETY_NOTICE = "본 서비스는 진단/처방이 아니며, 치료 변경은 반드시 의료진과 상담해야 합니다."
CONTEXT_FALLBACK_STATUS = "fallback"
CONTEXT_LOADED_STATUS = "loaded"


async def ask_chatbot(request: ChatbotAskRequest, user_id: int | None = None) -> ChatbotAskResponse:
    response, _ = await ask_chatbot_with_runtime_trace(request, user_id=user_id)
    return response


async def ask_chatbot_with_runtime_trace(
    request: ChatbotAskRequest, user_id: int | None = None
) -> tuple[ChatbotAskResponse, dict]:
    use_real_llm = config.CHATBOT_USE_REAL_LLM and has_openai_config(config)
    domain_context = await load_chatbot_domain_context(
        user_id=user_id,
        context_type=request.context_type.value,
        target_id=request.target_id,
    )
    routed = await run_chatbot_graph_async(
        user_message=request.message,
        user_context={
            "target_id": request.target_id,
            "domain_context": domain_context,
            **_graph_context_hints(domain_context),
        },
        context_type=request.context_type.value,
        use_real_llm=use_real_llm,
        use_rag=True,
    )
    return (
        ChatbotAskResponse(
            answer=routed.answer,
            source=routed.source,
            context_type=request.context_type,
            recommended_actions=routed.recommended_actions,
            safety_notice=routed.caution_message,
        ),
        build_chatbot_rag_runtime_trace(routed.trace_metadata),
    )


async def load_chatbot_domain_context(
    user_id: int | None,
    context_type: str,
    target_id: int | None,
) -> dict[str, Any] | None:
    normalized_context = str(context_type or "").upper()
    if normalized_context == "ANALYSIS":
        return await load_analysis_chatbot_context(user_id=user_id, target_id=target_id)
    if normalized_context == "DIET":
        return await load_diet_chatbot_context(user_id=user_id, target_id=target_id)
    if normalized_context == "CHALLENGE":
        return await load_challenge_chatbot_context(user_id=user_id, target_id=target_id)
    return None


async def load_analysis_chatbot_context(user_id: int | None, target_id: int | None) -> dict[str, Any]:
    if user_id is None:
        return _fallback_domain_context(
            "ANALYSIS", "로그인 사용자 정보를 확인할 수 없어 일반 건강관리 정보만 안내합니다."
        )

    result = await analysis_service.get_analysis_result(target_id) if target_id is not None else None
    if result is None and target_id is None:
        latest_results = await analysis_service.list_analysis_results(user_id=user_id, limit=1)
        result = latest_results[0] if latest_results else None
    if result is None:
        return _fallback_domain_context(
            "ANALYSIS", "아직 확인 가능한 분석 결과가 없어 일반 건강관리 정보만 안내합니다."
        )
    if int(result.user_id) != int(user_id):
        return _fallback_domain_context(
            "ANALYSIS", "요청한 분석 결과를 확인할 수 없어 일반 건강관리 정보만 안내합니다."
        )

    factors = await analysis_service.list_analysis_factors(int(result.id))
    recommendations = await challenge_service.list_challenge_recommendations(user_id=user_id, limit=3)
    risk_factors = [
        {
            "name": _safe_text(getattr(factor, "factor_name", None), limit=40),
            "value": _safe_text(getattr(factor, "factor_value", None), limit=40),
            "direction": _enum_value(getattr(factor, "direction", None)),
        }
        for factor in factors[:5]
    ]
    challenge_summaries = _challenge_recommendation_summaries(recommendations)
    lines = [
        "분석 결과 요약 context입니다.",
        f"- 관리 항목: {_analysis_type_label(getattr(result, 'analysis_type', None))}",
        f"- 위험 단계: {_risk_level_label(getattr(result, 'risk_level', None))}",
    ]
    summary = _safe_text(getattr(result, "summary", None), limit=160)
    if summary:
        lines.append(f"- 결과 요약: {summary}")
    if risk_factors:
        factor_text = ", ".join(item["name"] for item in risk_factors if item.get("name"))
        if factor_text:
            lines.append(f"- 주요 요인: {_safe_text(factor_text, limit=180)}")
    if challenge_summaries:
        lines.append(
            "- 연결 가능한 ACTIVE 추천 챌린지: "
            + _safe_text(", ".join(item["title"] for item in challenge_summaries), limit=180)
        )
    lines.append("- 이 정보는 진단이 아니라 생활관리 참고 정보이며 의료진 상담이 필요할 수 있습니다.")
    return {
        "context_type": "ANALYSIS",
        "status": CONTEXT_LOADED_STATUS,
        "safe_context_text": "\n".join(lines),
        "risk_factors": risk_factors,
        "analysis_result": {
            "disease_type": _analysis_type_label(getattr(result, "analysis_type", None)),
            "risk_level": _risk_level_label(getattr(result, "risk_level", None)),
            "risk_factors": risk_factors,
        },
        "recommended_challenges": challenge_summaries,
    }


async def load_diet_chatbot_context(user_id: int | None, target_id: int | None) -> dict[str, Any]:
    if user_id is None:
        return _fallback_domain_context("DIET", "로그인 사용자 정보를 확인할 수 없어 일반 식단 정보만 안내합니다.")

    record = await diet_service.get_diet_record(target_id) if target_id is not None else None
    if record is None and target_id is None:
        latest_records = await diet_service.list_diet_records(user_id=user_id, limit=1)
        record = latest_records[0] if latest_records else None
    if record is None:
        return _fallback_domain_context("DIET", "아직 확인 가능한 식단 기록이 없어 일반 식단 정보만 안내합니다.")
    if int(record.user_id) != int(user_id):
        return _fallback_domain_context("DIET", "요청한 식단 기록을 확인할 수 없어 일반 식단 정보만 안내합니다.")

    health_record = await health_repository.get_latest_health_record_by_user(user_id)
    analysis_results = await analysis_repository.list_analysis_results_by_user(user_id, limit=20)
    active_challenges = await challenge_service.list_active_challenges(limit=500)
    recommendation = diet_recommendations.build_diet_health_recommendation_context_response(
        diet_record=record,
        analysis_results=analysis_results,
        health_record=health_record,
        active_challenges=active_challenges,
    )
    food_names = _diet_food_names(record)
    findings = [
        _safe_text(item.get("message"), limit=120)
        for item in recommendation.get("nutrition_findings", [])
        if isinstance(item, dict) and item.get("message")
    ][:3]
    disease_groups = [
        _safe_text(item.get("label") or item.get("disease_group"), limit=40)
        for item in recommendation.get("disease_context", [])
        if isinstance(item, dict)
    ][:5]
    diet_issues = [
        _safe_text(item.get("issue_key") or item.get("label"), limit=40)
        for item in recommendation.get("nutrition_findings", [])
        if isinstance(item, dict)
    ][:5]
    challenge_summaries = [
        {
            "title": _safe_text(item.get("title"), limit=60),
            "reason": _safe_text(item.get("reason"), limit=120),
        }
        for item in recommendation.get("recommended_challenges", [])
        if isinstance(item, dict)
    ][:3]
    rag_comment = recommendation.get("rag_comment") if isinstance(recommendation.get("rag_comment"), dict) else {}
    rag_summary = _safe_text(rag_comment.get("summary") or rag_comment.get("message"), limit=180)
    lines = ["식단 기록 요약 context입니다."]
    if food_names:
        lines.append(f"- 음식 후보: {_safe_text(', '.join(food_names), limit=180)}")
    if findings:
        lines.append(f"- 영양/식습관 포인트: {_safe_text(' '.join(findings), limit=260)}")
    if disease_groups:
        lines.append(f"- 연결된 관리 그룹: {_safe_text(', '.join(disease_groups), limit=160)}")
    if diet_issues:
        lines.append(f"- 식단 이슈: {_safe_text(', '.join(diet_issues), limit=160)}")
    if rag_summary:
        lines.append(f"- 참고 코멘트: {rag_summary}")
    if challenge_summaries:
        lines.append(
            "- 연결 가능한 ACTIVE 추천 챌린지: "
            + _safe_text(", ".join(item["title"] for item in challenge_summaries), limit=180)
        )
    lines.append("- 실제 섭취량은 확정하지 말고, CKD/신장 관련 식사는 개인 수치에 따라 의료진과 상담해야 합니다.")
    return {
        "context_type": "DIET",
        "status": CONTEXT_LOADED_STATUS,
        "safe_context_text": "\n".join(lines),
        "diet_issue": diet_issues,
        "disease_groups": disease_groups,
        "recommended_challenges": challenge_summaries,
    }


async def load_challenge_chatbot_context(user_id: int | None, target_id: int | None) -> dict[str, Any]:
    if user_id is None:
        return _fallback_domain_context(
            "CHALLENGE", "로그인 사용자 정보를 확인할 수 없어 일반 챌린지 정보만 안내합니다."
        )

    challenge = await challenge_service.get_challenge(target_id) if target_id is not None else None
    if challenge is None and target_id is None:
        recommendations = await challenge_service.list_challenge_recommendations(user_id=user_id, limit=1)
        challenge = getattr(recommendations[0], "challenge", None) if recommendations else None
        if challenge is None:
            user_challenges = await challenge_service.list_user_challenges(user_id=user_id, limit=1)
            challenge = (
                await challenge_service.get_challenge(user_challenges[0].challenge_id) if user_challenges else None
            )
    if challenge is None:
        return _fallback_domain_context(
            "CHALLENGE", "확인 가능한 ACTIVE 챌린지가 없어 일반 건강관리 정보만 안내합니다."
        )
    if _enum_value(getattr(challenge, "status", None)) != ChallengeStatus.ACTIVE.value:
        return _fallback_domain_context(
            "CHALLENGE", "비활성 챌린지는 안내 context에 포함하지 않고 일반 정보만 안내합니다."
        )

    user_challenges = await challenge_service.list_user_challenges(user_id=user_id, limit=100)
    progress = next(
        (
            item
            for item in user_challenges
            if int(getattr(item, "challenge_id", 0) or 0) == int(challenge.id)
            and _enum_value(getattr(item, "status", None)) != UserChallengeStatus.CANCELED.value
        ),
        None,
    )
    progress_text = ""
    if progress is not None:
        progress_text = (
            f"- 진행 상태: {getattr(progress, 'progress', 0)}%, "
            f"오늘 완료 여부: {'완료' if getattr(progress, 'today_completed', False) else '미완료'}"
        )
    lines = [
        "챌린지 요약 context입니다.",
        f"- 제목: {_safe_text(getattr(challenge, 'title', None), limit=80)}",
        f"- 카테고리: {_enum_value(getattr(challenge, 'category', None))}",
        f"- 대상 질환군: {_enum_value(getattr(challenge, 'target_disease', None))}",
        f"- 기간: {getattr(challenge, 'duration_days', None) or 0}일",
    ]
    caution = _safe_text(getattr(challenge, "caution_message", None), limit=160)
    if caution:
        lines.append(f"- 주의사항: {caution}")
    if progress_text:
        lines.append(progress_text)
    lines.append("- 챌린지는 생활습관 실천 보조이며 치료 효과를 보장하지 않습니다.")
    return {
        "context_type": "CHALLENGE",
        "status": CONTEXT_LOADED_STATUS,
        "safe_context_text": "\n".join(lines),
        "challenge": {
            "title": _safe_text(getattr(challenge, "title", None), limit=80),
            "category": _enum_value(getattr(challenge, "category", None)),
            "target_disease": _enum_value(getattr(challenge, "target_disease", None)),
        },
    }


def build_chatbot_rag_runtime_trace(trace_metadata: dict | None) -> dict:
    retrieval = (trace_metadata or {}).get("retrieval") or {}
    if not isinstance(retrieval, dict):
        return {}

    return {
        "rag_strategy": retrieval.get("retriever_strategy") or retrieval.get("strategy") or retrieval.get("source"),
        "keyword_returned_count": retrieval.get("keyword_returned_count"),
        "vector_returned_count": retrieval.get("vector_returned_count"),
        "merged_count": retrieval.get("merged_count"),
        "final_count": retrieval.get("final_count") or retrieval.get("document_count"),
        "fallback_used": retrieval.get("fallback_used", retrieval.get("fallback")),
        "fallback_reason": retrieval.get("fallback_reason"),
    }


def _graph_context_hints(domain_context: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(domain_context, dict) or domain_context.get("status") != CONTEXT_LOADED_STATUS:
        return {}
    hints: dict[str, Any] = {}
    for key in ("risk_factors", "analysis_result"):
        if domain_context.get(key):
            hints[key] = domain_context[key]
    return hints


def _fallback_domain_context(context_type: str, message: str) -> dict[str, Any]:
    return {
        "context_type": context_type,
        "status": CONTEXT_FALLBACK_STATUS,
        "safe_context_text": message,
    }


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value) or "")


def _safe_text(value: object, *, limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "[secret]", text)
    text = re.sub(r"eyJ[A-Za-z0-9_.-]{20,}", "[token]", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _analysis_type_label(value: object) -> str:
    labels = {
        "HYPERTENSION": "혈압 관리",
        "DIABETES": "혈당 관리",
        "DYSLIPIDEMIA": "지질 관리",
        "OBESITY": "체중 관리",
        "ABDOMINAL_OBESITY": "복부비만 관리",
        "FATTY_LIVER": "지방간 관리",
        "ANEMIA": "빈혈 관리",
        "LIVER_FUNCTION": "간 기능 관리",
        "KIDNEY_FUNCTION": "신장 기능 관리",
        "CHRONIC_KIDNEY_DISEASE": "신장 관리",
    }
    return labels.get(_enum_value(value), _enum_value(value) or "건강관리")


def _risk_level_label(value: object) -> str:
    labels = {
        "LOW": "낮음",
        "ATTENTION": "관심 필요",
        "CAUTION": "주의",
        "HIGH_CAUTION": "높은 주의",
    }
    return labels.get(_enum_value(value), _enum_value(value) or "확인 필요")


def _challenge_recommendation_summaries(recommendations: list[Any]) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    for recommendation in recommendations:
        challenge = getattr(recommendation, "challenge", None)
        title = _safe_text(getattr(challenge, "title", None), limit=60)
        if not title:
            continue
        summaries.append(
            {
                "title": title,
                "reason": _safe_text(getattr(recommendation, "reason", None), limit=120),
            }
        )
    return summaries[:3]


def _diet_food_names(record: Any) -> list[str]:
    foods = getattr(record, "detected_foods", None)
    if not isinstance(foods, list):
        return []
    names: list[str] = []
    for food in foods[:5]:
        if not isinstance(food, dict):
            continue
        name = _safe_text(food.get("name") or food.get("original_name") or food.get("query_name"), limit=40)
        if name:
            names.append(name)
    return names
