import json
from typing import Any

from ai_runtime.llm.llm_client import call_llm
from ai_runtime.llm.prompt_templates import MAIN_HEALTH_RAG_PROMPT, MAIN_HEALTH_RAG_PROMPT_VERSION
from ai_runtime.llm.rag_sources import is_allowed_rag_source
from ai_runtime.llm.safety import check_medical_safety
from ai_runtime.llm.schemas import MainHealthChatbotOutput, RagContextSource

CAUTION_MESSAGE = "이 정보는 진단이 아니며, 정확한 진단과 치료는 의료진 상담이 필요합니다."
RAG_SOURCE = "rag_llm"
RAG_INTENT = "main_health_rag_guidance"


def generate_main_health_rag_response(
    user_message: str,
    retrieved_context: str,
    context_sources: list[dict] | None = None,
    use_real_llm: bool = False,
) -> MainHealthChatbotOutput:
    context_source_count = len(context_sources or [])
    if not retrieved_context.strip():
        return build_fallback_response(
            reason="empty_retrieved_context",
            context_source_count=context_source_count,
        )

    parsed_sources = parse_context_sources(context_sources)
    if context_sources and not are_allowed_context_sources(parsed_sources):
        return build_fallback_response(
            reason="disallowed_context_source",
            context_source_count=context_source_count,
        )

    if not use_real_llm:
        answer = build_context_based_fallback_answer(retrieved_context)
    else:
        raw_answer = call_llm(
            build_main_health_rag_prompt(
                user_message=user_message,
                retrieved_context=retrieved_context,
                context_sources=parsed_sources,
            ),
            metadata={
                "prompt_version": MAIN_HEALTH_RAG_PROMPT_VERSION,
                "source": RAG_SOURCE,
                "chatbot_type": "main_health_chatbot",
                "use_real_llm": True,
            },
        )
        answer = extract_answer_from_rag_response(raw_answer)

    final_answer = ensure_caution_message(answer)
    safety_result = add_rag_metadata(
        check_medical_safety(final_answer),
        context_source_count=context_source_count,
    )

    return MainHealthChatbotOutput(
        answer=final_answer,
        intent=RAG_INTENT,
        source=RAG_SOURCE,
        caution_message=CAUTION_MESSAGE,
        tone="friendly",
        is_safe=safety_result["is_safe"],
        safety_result=safety_result,
    )


def build_main_health_rag_prompt(
    user_message: str,
    retrieved_context: str,
    context_sources: list[RagContextSource],
) -> str:
    sources_text = "\n".join(format_context_source(source) for source in context_sources)

    return f"""
{MAIN_HEALTH_RAG_PROMPT}

사용자 질문:
{user_message}

RAG context:
{retrieved_context}

context sources:
{sources_text or "- 제공된 출처 메타데이터 없음"}
""".strip()


def parse_context_sources(context_sources: list[dict] | None) -> list[RagContextSource]:
    parsed_sources: list[RagContextSource] = []
    for source in context_sources or []:
        if isinstance(source, RagContextSource):
            parsed_sources.append(source)
        elif isinstance(source, dict):
            parsed_sources.append(RagContextSource(**source))
    return parsed_sources


def are_allowed_context_sources(context_sources: list[RagContextSource]) -> bool:
    if not context_sources:
        return False

    return all(
        is_allowed_rag_source(
            source_name=source.source_name,
            url=source.url,
        )
        for source in context_sources
    )


def format_context_source(source: RagContextSource) -> str:
    parts = [
        f"title={source.title}" if source.title else None,
        f"source_name={source.source_name}" if source.source_name else None,
        f"url={source.url}" if source.url else None,
    ]
    return f"- {', '.join(part for part in parts if part)}"


def build_context_based_fallback_answer(retrieved_context: str) -> str:
    context_summary = retrieved_context.strip().splitlines()[0]
    return (
        f"제공된 공신력 있는 건강정보 context 기준으로 보면, {context_summary} "
        "관련 내용은 생활습관 관리와 예방 관점에서 참고할 수 있습니다. "
        "개인의 상태에 따라 해석이 달라질 수 있으므로 검진 결과나 증상이 있다면 의료진과 상담해 주세요."
    )


def extract_answer_from_rag_response(raw_answer: str) -> str:
    try:
        parsed: dict[str, Any] = json.loads(raw_answer)
    except json.JSONDecodeError:
        return raw_answer

    answer = parsed.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer
    return raw_answer


def ensure_caution_message(answer: str) -> str:
    final_answer = answer.strip()
    if "진단이 아니" not in final_answer or "의료진 상담" not in final_answer:
        final_answer = f"{final_answer} {CAUTION_MESSAGE}"
    return final_answer


def build_fallback_response(reason: str, context_source_count: int) -> MainHealthChatbotOutput:
    answer = (
        "현재 질문에 답변할 수 있는 신뢰 가능한 근거 자료가 충분하지 않습니다. "
        "일반적인 건강정보는 참고용으로만 확인하고, 증상이나 검사 결과 해석이 필요하면 의료진과 상담해 주세요. "
        f"{CAUTION_MESSAGE}"
    )
    safety_result = add_rag_metadata(
        check_medical_safety(answer),
        context_source_count=context_source_count,
        fallback_reason=reason,
    )

    return MainHealthChatbotOutput(
        answer=answer,
        intent=RAG_INTENT,
        source=RAG_SOURCE,
        caution_message=CAUTION_MESSAGE,
        tone="friendly",
        is_safe=safety_result["is_safe"],
        safety_result=safety_result,
    )


def add_rag_metadata(
    safety_result: dict,
    context_source_count: int,
    fallback_reason: str | None = None,
) -> dict:
    metadata = {
        **safety_result.get("metadata", {}),
        "prompt_version": MAIN_HEALTH_RAG_PROMPT_VERSION,
        "source": RAG_SOURCE,
        "context_source_count": context_source_count,
    }
    if fallback_reason:
        metadata["fallback_reason"] = fallback_reason

    return {
        **safety_result,
        "metadata": metadata,
    }
