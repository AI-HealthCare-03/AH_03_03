import logging
from typing import Any

from ai_runtime.llm.rag import retrieve_keyword_rag_contexts
from ai_runtime.llm.rag.tracing import trace_keyword_rag_retrieval
from ai_runtime.llm.schemas import (
    AnalysisExplanationInput,
    DietScoreExplanationInput,
    ExplanationOutput,
    RetrievedContext,
)
from app.core import config

SAFETY_NOTICE = "이 설명은 진단이 아니며, 건강관리 참고용입니다. 정확한 진단과 치료는 의료진 상담이 필요합니다."
logger = logging.getLogger(__name__)

DISEASE_LABELS = {
    "DIABETES": "당뇨",
    "DM": "당뇨",
    "HYPERTENSION": "고혈압",
    "HTN": "고혈압",
    "DYSLIPIDEMIA": "이상지질혈증",
    "DL": "이상지질혈증",
    "OBESITY": "비만",
    "OBE": "비만",
    "ANEMIA": "빈혈",
    "ANEM": "빈혈",
}

DIET_LOW_SCORE_MESSAGES = {
    "DM": "혈당 관리 관점에서 주의가 필요한 식단입니다.",
    "HTN": "나트륨과 혈압 관리 관점에서 조절이 필요한 식단입니다.",
    "DL": "포화지방과 지질 관리 관점에서 구성을 점검해 볼 필요가 있습니다.",
    "OBE": "총 섭취량과 체중 관리 관점에서 조절이 필요한 식단입니다.",
    "ANEM": "철분과 단백질 등 균형 잡힌 영양 구성을 확인해 보세요.",
}

DIET_ACTION_MESSAGES = {
    "DM": "정제 탄수화물과 당류를 줄이고 단백질, 채소를 함께 구성해 보세요.",
    "HTN": "국물, 소스, 가공식품의 나트륨을 줄이는 방향으로 조절해 보세요.",
    "DL": "튀김류와 포화지방이 많은 음식을 줄이고 생선, 견과류, 채소를 보완해 보세요.",
    "OBE": "식사량과 간식 빈도를 점검하고 식후 가벼운 활동을 함께 기록해 보세요.",
    "ANEM": "철분이 포함된 식품과 단백질 식품을 균형 있게 챙겨 보세요.",
}


def generate_analysis_explanation(input_data: AnalysisExplanationInput) -> ExplanationOutput:
    """분석 결과를 진단이 아닌 건강관리 참고 설명으로 변환한다."""
    disease_label = _disease_label(input_data.disease_type)
    risk_label = _risk_level_label(input_data.risk_level)
    factor_text = _factor_summary(input_data.factors)
    model_text = _model_summary(input_data.model_name, input_data.model_version)

    summary = f"{disease_label} 관련 위험도는 {risk_label}으로 분류되었습니다. {model_text}"
    caution = f"{factor_text} 결과는 입력된 건강정보와 검진값을 바탕으로 한 참고 신호입니다."
    recommended_action = _analysis_action(input_data.disease_type, input_data.risk_level)

    return ExplanationOutput(
        summary=summary.strip(),
        caution=caution.strip(),
        recommended_action=recommended_action,
        safety_notice=SAFETY_NOTICE,
        source="rule_based_explanation",
    )


def generate_diet_score_explanation(input_data: DietScoreExplanationInput) -> ExplanationOutput:
    """식단 점수 중 가장 주의가 필요한 질병군을 짧게 설명한다."""
    lowest_code, lowest_score = _lowest_diet_score(input_data.disease_scores)
    if lowest_code is None:
        return ExplanationOutput(
            summary="식단 점수 산정에 필요한 음식 매칭 정보가 부족합니다.",
            caution="음식명을 조금 더 구체적으로 입력하면 질병군별 식단 점수를 확인할 수 있습니다.",
            recommended_action="식단 사진 또는 음식명을 보완해 다시 분석해 보세요.",
            safety_notice=SAFETY_NOTICE,
            source="rule_based_explanation",
        )

    score_text = f"{lowest_score:.0f}점" if lowest_score is not None else "점수 미산정"
    summary = DIET_LOW_SCORE_MESSAGES.get(
        lowest_code,
        f"{_disease_label(lowest_code)} 관리 관점에서 식단 구성을 확인해 볼 필요가 있습니다.",
    )
    return ExplanationOutput(
        summary=f"{summary} 가장 낮은 질병군별 식단 점수는 {_disease_label(lowest_code)} {score_text}입니다.",
        caution="식단 점수는 음식 구성과 영양 기준에 따른 참고용 평가입니다.",
        recommended_action=DIET_ACTION_MESSAGES.get(lowest_code, "음식 구성과 섭취량을 함께 기록해 보세요."),
        safety_notice=SAFETY_NOTICE,
        source="rule_based_explanation",
    )


def retrieve_health_context(query: str, disease_type: str | None = None) -> list[RetrievedContext]:
    """로컬 후보 문서 기반 keyword RAG PoC이며 외부 검색/embedding은 수행하지 않는다."""
    if not config.RAG_ENABLED:
        return []

    top_k = 2
    include_safety_disclaimer = True
    try:
        contexts = retrieve_keyword_rag_contexts(
            user_message=query,
            disease_type=disease_type,
            top_k=top_k,
            include_safety_disclaimer=include_safety_disclaimer,
        )
    except Exception:
        logger.exception(
            "Keyword RAG context retrieval failed; continuing with rule-based explanation",
            extra={"disease_type": disease_type},
        )
        return []

    trace_keyword_rag_retrieval(
        query=query,
        disease_type=disease_type,
        contexts=contexts,
        top_k=top_k,
        include_safety_disclaimer=include_safety_disclaimer,
    )
    return contexts


def generate_explanation_with_context(
    input_data: AnalysisExplanationInput,
    contexts: list[RetrievedContext] | None = None,
    use_real_llm: bool = False,
) -> ExplanationOutput:
    """LangGraph analysis explanation node entrypoint.

    The node currently uses safe rule/fallback wording and optional RAG
    references. Real LLM calls are intentionally not required for this path.
    """
    from ai_runtime.llm.graph import run_analysis_explanation_graph

    return run_analysis_explanation_graph(
        input_data=input_data,
        contexts=contexts or [],
        use_real_llm=use_real_llm,
    ).explanation


def _disease_label(disease_type: str) -> str:
    return DISEASE_LABELS.get(str(disease_type).upper(), str(disease_type))


def _risk_level_label(risk_level: str) -> str:
    value = str(risk_level).upper()
    if value == "LOW":
        return "낮음"
    if value == "ATTENTION":
        return "관심 필요"
    if value in {"CAUTION", "MEDIUM"}:
        return "주의"
    if value in {"HIGH_CAUTION", "HIGH"}:
        return "높은 주의"
    return str(risk_level)


def _factor_summary(factors: list[Any]) -> str:
    names = [str(getattr(factor, "name", "")).strip() for factor in factors]
    names = [name for name in names if name]
    if not names:
        return "주요 위험요인은 추가 확인이 필요합니다."
    return f"주요 확인 항목은 {', '.join(names[:3])}입니다."


def _model_summary(model_name: str | None, model_version: str | None) -> str:
    if model_name == "catboost":
        version_text = f"({model_version})" if model_version else ""
        return f"CatBoost 모델 {version_text} 기반 참고 신호가 반영되었습니다."
    return "현재 입력값 기준의 규칙 기반 참고 신호가 반영되었습니다."


def _analysis_action(disease_type: str, risk_level: str) -> str:
    disease_key = str(disease_type).upper()
    level = str(risk_level).upper()
    if level == "LOW":
        return "현재 기록 습관을 유지하면서 정기적으로 건강정보를 업데이트해 보세요."
    if level == "ATTENTION":
        return "건강정보를 꾸준히 기록하며 식사, 활동량, 수면 변화를 함께 확인해 보세요."
    if disease_key in {"DIABETES", "DM"}:
        return "식후 활동, 당류 섭취, 공복혈당 기록을 함께 관리해 보세요."
    if disease_key in {"HYPERTENSION", "HTN"}:
        return "나트륨 섭취와 혈압 기록을 확인하고, 무리 없는 활동을 꾸준히 이어가 보세요."
    if disease_key in {"DYSLIPIDEMIA", "DL"}:
        return "포화지방 섭취와 활동량을 점검하고 검진 수치를 주기적으로 확인해 보세요."
    if disease_key in {"OBESITY", "OBE"}:
        return "체중, 허리둘레, 식사량과 활동량을 함께 기록해 보세요."
    return "생활습관 기록을 이어가고 필요한 경우 의료진 상담을 받아 보세요."


def _lowest_diet_score(disease_scores: dict[str, float | int | None]) -> tuple[str | None, float | None]:
    valid_scores = {str(code).upper(): float(score) for code, score in disease_scores.items() if score is not None}
    if not valid_scores:
        return None, None
    code = min(valid_scores, key=valid_scores.get)
    return code, valid_scores[code]
