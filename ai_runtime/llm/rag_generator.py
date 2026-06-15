import json
import re
from typing import Any

from ai_runtime.llm.llm_client import call_llm_json
from ai_runtime.llm.prompt_templates import (
    FALLBACK_SAFE_RESPONSE_PROMPT_VERSION,
    MAIN_HEALTH_RAG_PROMPT_VERSION,
    render_prompt,
)
from ai_runtime.llm.rag_sources import is_allowed_rag_source
from ai_runtime.llm.safety import check_medical_safety
from ai_runtime.llm.schemas import MainHealthChatbotOutput, RagContextSource

CAUTION_MESSAGE = "이 정보는 진단이 아니며, 정확한 진단과 치료는 의료진 상담이 필요합니다."
RAG_SOURCE = "rag_llm"
RAG_INTENT = "main_health_rag_guidance"
RAG_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {
            "type": "string",
            "description": "사용자에게 보여줄 자연어 답변. JSON, markdown code fence, 내부 필드명을 포함하지 않는다.",
        }
    },
    "required": ["answer"],
}
_JSON_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(?P<body>\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
_JSON_CODE_FENCE_START_RE = re.compile(r"```(?:json)?\s*(?P<body>\{.*)", re.IGNORECASE | re.DOTALL)
_INTERNAL_RESPONSE_TERMS = (
    "```json",
    "```",
    '"answer":',
    '"intent":',
    '"source":',
    '"is_safe":',
    '"caution_message":',
)


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
        try:
            raw_answer = call_llm_json(
                build_main_health_rag_prompt(
                    user_message=user_message,
                    retrieved_context=retrieved_context,
                    context_sources=parsed_sources,
                ),
                schema=RAG_RESPONSE_SCHEMA,
                schema_name="main_health_rag_answer",
                metadata={
                    "prompt_version": MAIN_HEALTH_RAG_PROMPT_VERSION,
                    "source": RAG_SOURCE,
                    "chatbot_type": "main_health_chatbot",
                    "use_real_llm": True,
                },
            )
            answer = extract_answer_from_rag_response(raw_answer)
            if not is_public_rag_answer(answer):
                raise ValueError("LLM RAG answer still includes internal response fields.")
        except Exception:
            answer = build_context_based_fallback_answer(retrieved_context)

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
    source_titles = [source.title for source in context_sources if source.title]
    reference_summary = f"참고 정보: {', '.join(source_titles)} 후보 문서를 함께 확인했습니다." if source_titles else ""
    return render_prompt(
        "rag_grounded_answer_prompt",
        user_message=user_message,
        retrieved_context=retrieved_context,
        context_sources=sources_text or "- 제공된 출처 메타데이터 없음",
        reference_summary=reference_summary or "- 별도 요약 없음",
    )


def parse_context_sources(context_sources: list[dict] | None) -> list[RagContextSource]:
    parsed_sources: list[RagContextSource] = []
    for source in context_sources or []:
        if isinstance(source, RagContextSource):
            parsed_sources.append(source)
        elif isinstance(source, dict):
            parsed_sources.append(
                RagContextSource(
                    source_name=source.get("source_name") or source.get("source_org"),
                    url=source.get("url") or source.get("source_url"),
                    title=source.get("title"),
                )
            )
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


def extract_answer_from_rag_response(raw_answer: str, *, max_depth: int = 2) -> str:
    raw_answer = _strip_markdown_code_fences(raw_answer.strip())
    if max_depth < 0:
        return raw_answer

    fenced_json = _extract_fenced_json(raw_answer)
    if fenced_json is not None:
        return extract_answer_from_rag_response(fenced_json, max_depth=max_depth - 1)

    try:
        parsed: dict[str, Any] = json.loads(raw_answer)
    except json.JSONDecodeError:
        parsed = _extract_embedded_json_with_answer(raw_answer)
        if parsed is None:
            return raw_answer

    answer = parsed.get("answer")
    if isinstance(answer, str) and answer.strip():
        return extract_answer_from_rag_response(answer, max_depth=max_depth - 1)
    return raw_answer


def _extract_fenced_json(raw_answer: str) -> str | None:
    matched = _JSON_CODE_FENCE_RE.search(raw_answer)
    if matched is not None:
        return matched.group("body").strip()

    matched = _JSON_CODE_FENCE_START_RE.search(raw_answer)
    if matched is None:
        return None
    return _extract_first_json_object_text(matched.group("body"))


def _extract_embedded_json_with_answer(raw_answer: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for start_index, character in enumerate(raw_answer):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw_answer[start_index:])
        except json.JSONDecodeError:
            json_text = _extract_first_json_object_text(raw_answer[start_index:])
            if json_text is None:
                continue
            try:
                parsed = json.loads(json_text)
            except json.JSONDecodeError:
                continue
        if isinstance(parsed, dict) and isinstance(parsed.get("answer"), str):
            return parsed
    return None


def _extract_first_json_object_text(value: str) -> str | None:
    depth = 0
    in_string = False
    escaped = False
    start_index: int | None = None
    for index, character in enumerate(value):
        if start_index is None:
            if character != "{":
                continue
            start_index = index
            depth = 1
            continue

        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return value[start_index : index + 1].strip()
    return None


def _strip_markdown_code_fences(answer: str) -> str:
    stripped = answer.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def is_public_rag_answer(answer: str) -> bool:
    return bool(answer.strip()) and not any(term in answer for term in _INTERNAL_RESPONSE_TERMS)


def ensure_caution_message(answer: str) -> str:
    final_answer = answer.strip()
    if "진단이 아니" not in final_answer or "의료진 상담" not in final_answer:
        final_answer = f"{final_answer} {CAUTION_MESSAGE}"
    return final_answer


def build_fallback_response(reason: str, context_source_count: int) -> MainHealthChatbotOutput:
    answer = render_prompt("fallback_safe_response_prompt")
    safety_result = add_rag_metadata(
        check_medical_safety(answer),
        context_source_count=context_source_count,
        fallback_reason=reason,
        prompt_version=FALLBACK_SAFE_RESPONSE_PROMPT_VERSION,
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
    prompt_version: str = MAIN_HEALTH_RAG_PROMPT_VERSION,
) -> dict:
    metadata = {
        **safety_result.get("metadata", {}),
        "prompt_version": prompt_version,
        "source": RAG_SOURCE,
        "context_source_count": context_source_count,
    }
    if fallback_reason:
        metadata["fallback_reason"] = fallback_reason

    return {
        **safety_result,
        "metadata": metadata,
    }
