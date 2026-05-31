from __future__ import annotations

from ai_runtime.llm.prompt_templates import (
    ANALYSIS_EXPLANATION_PROMPT_VERSION,
    CHALLENGE_RECOMMENDATION_PROMPT_VERSION,
    FALLBACK_SAFE_RESPONSE_PROMPT_VERSION,
    HEALTH_CHAT_PROMPT_VERSION,
    RAG_GROUNDED_ANSWER_PROMPT_VERSION,
    get_prompt_spec,
    get_prompt_version,
    render_prompt,
)
from ai_runtime.llm.rag_generator import build_main_health_rag_prompt
from ai_runtime.llm.schemas import RagContextSource


def test_prompt_registry_exposes_expected_versions() -> None:
    assert get_prompt_version("health_chat_prompt") == HEALTH_CHAT_PROMPT_VERSION
    assert get_prompt_version("analysis_explanation_prompt") == ANALYSIS_EXPLANATION_PROMPT_VERSION
    assert get_prompt_version("challenge_recommendation_prompt") == CHALLENGE_RECOMMENDATION_PROMPT_VERSION
    assert get_prompt_version("rag_grounded_answer_prompt") == RAG_GROUNDED_ANSWER_PROMPT_VERSION
    assert get_prompt_version("fallback_safe_response_prompt") == FALLBACK_SAFE_RESPONSE_PROMPT_VERSION


def test_health_chat_prompt_renders_safety_policy() -> None:
    rendered = render_prompt("health_chat_prompt", user_message="혈압 관리는 어떻게 하나요?")

    assert "혈압 관리는 어떻게 하나요?" in rendered
    assert "진단" in rendered
    assert "처방" in rendered
    assert "의료진 상담" in rendered
    assert "위기 키워드" in rendered


def test_rag_grounded_prompt_includes_reference_structure() -> None:
    rendered = build_main_health_rag_prompt(
        user_message="공복혈당 관리 방법",
        retrieved_context="[diabetes] 당뇨 관리\n생활습관 관리 참고 문서",
        context_sources=[
            RagContextSource(
                title="당뇨 관리 기준",
                source_name="대한당뇨병학회",
                url="https://example.test/diabetes",
            )
        ],
    )

    assert "공복혈당 관리 방법" in rendered
    assert "RAG context" in rendered
    assert "context sources" in rendered
    assert "reference summary" in rendered
    assert "당뇨 관리 기준" in rendered
    assert "대한당뇨병학회" in rendered
    assert "rag_llm" in rendered


def test_fallback_safe_response_prompt_avoids_diagnosis_or_prescription_instruction() -> None:
    spec = get_prompt_spec("fallback_safe_response_prompt")
    rendered = spec.render()

    assert spec.version == FALLBACK_SAFE_RESPONSE_PROMPT_VERSION
    assert "현재 질문에 답변할 수 있는 신뢰 가능한 근거 자료가 충분하지 않습니다" in rendered
    assert "참고용" in rendered
    assert "의료진과 상담" in rendered
    assert "진단이 아니" in rendered
    assert "진단되었습니다" not in rendered
    assert "복용하세요" not in rendered
    assert "처방합니다" not in rendered
