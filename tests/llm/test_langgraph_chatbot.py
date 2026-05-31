from __future__ import annotations

import pytest

from ai_runtime.llm.graph import run_chatbot_graph
from ai_runtime.llm.graph.builder import build_health_chatbot_graph
from ai_runtime.llm.graph.nodes import sanitize_for_trace, trace_graph_node


@pytest.fixture(autouse=True)
def clear_graph_cache():
    build_health_chatbot_graph.cache_clear()
    yield
    build_health_chatbot_graph.cache_clear()


def test_general_health_question_passes_through_langgraph_without_real_llm(monkeypatch) -> None:
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.RAG_ENABLED", False)

    result = run_chatbot_graph(
        user_message="혈당 관리는 어떻게 하나요?",
        context_type="MAIN",
        use_real_llm=False,
    )

    assert result.source == "rule_engine"
    assert result.intent == "diabetes_guidance"
    assert result.safety_level is None
    assert "진단이 아니" in result.answer
    assert "의료진 상담" in result.caution_message
    assert any("건강 분석" in action for action in result.recommended_actions)


def test_graph_result_contract_preserves_expected_fields_and_rule_source(monkeypatch) -> None:
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.RAG_ENABLED", False)

    result = run_chatbot_graph(
        user_message="혈압 관리는 어떻게 하나요?",
        context_type="MAIN",
        use_real_llm=False,
    )

    assert result.answer
    assert result.source == "rule_engine"
    assert result.intent == "hypertension_guidance"
    assert result.safety_level is None
    assert isinstance(result.recommended_actions, list)
    assert result.caution_message
    assert result.reference_sources == []
    assert result.reference_summary is None
    assert "node_path" in result.metadata
    assert "sanitized_message_preview" in result.trace_metadata


def test_empty_and_none_inputs_complete_without_key_error(monkeypatch) -> None:
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.RAG_ENABLED", False)

    for message in ("", None):
        result = run_chatbot_graph(
            user_message=message,
            user_context=None,
            context_type=None,
            use_real_llm=False,
            use_rag=True,
        )

        assert result.answer
        assert result.source == "rule_engine_unmatched"
        assert result.intent == "chronic_disease_prevention"
        assert result.reference_sources == []
        assert result.reference_summary is None
        assert result.trace_metadata["retrieval"]["reason"] == "rag_disabled"


def test_metadata_and_trace_metadata_accumulate_graph_path(monkeypatch) -> None:
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.RAG_ENABLED", False)

    result = run_chatbot_graph(
        user_message="만성질환 예방을 어떻게 시작하면 좋나요?",
        context_type="GENERAL",
        use_real_llm=False,
    )

    assert result.metadata["node_path"] == [
        "normalize_input",
        "check_mental_health_safety",
        "classify_intent",
        "retrieve_rag_context",
        "generate_llm_answer",
        "check_grounding_or_fallback",
        "build_recommended_actions",
        "format_final_response",
    ]
    assert result.trace_metadata["retrieval"] == {"enabled": False, "reason": "rag_disabled"}
    assert result.safety_result["metadata"]["graph_node"] == "format_final_response"


@pytest.mark.parametrize(
    "message",
    [
        "자해하고 싶은 생각이 들어요",
        "극단 선택을 떠올리고 있어요",
        "요즘 죽고 싶다는 생각이 들어요",
    ],
)
def test_crisis_keywords_return_immediate_support_without_challenge_first(message: str, monkeypatch) -> None:
    def fail_if_called(state):
        raise AssertionError("Crisis safety path must bypass generate_llm_answer")

    monkeypatch.setattr("ai_runtime.llm.graph.builder.generate_llm_answer", fail_if_called)

    result = run_chatbot_graph(
        user_message=message,
        context_type="MAIN",
        use_real_llm=True,
    )

    assert result.source == "safety_policy"
    assert result.safety_level == "crisis"
    assert result.intent == "mental_health_crisis_support"
    assert result.fallback_reason == "mental_health_crisis_bypass"
    assert "챌린지 추천보다 안전 확보가 우선" in result.answer
    assert "보호자" in result.answer
    assert "전문기관" in result.answer
    assert any("119" in action or "112" in action for action in result.recommended_actions)


def test_crisis_keyword_bypasses_generation_node(monkeypatch) -> None:
    def fail_if_called(state):
        raise AssertionError("Crisis safety path must bypass generate_llm_answer")

    monkeypatch.setattr("ai_runtime.llm.graph.builder.generate_llm_answer", fail_if_called)

    result = run_chatbot_graph(
        user_message="요즘 죽고 싶다는 생각이 들어요",
        context_type="MAIN",
        use_real_llm=True,
    )

    assert result.source == "safety_policy"
    assert result.intent == "mental_health_crisis_support"
    assert result.safety_level == "crisis"
    assert "챌린지 추천보다 안전 확보가 우선" in result.answer
    assert all("챌린지" not in action for action in result.recommended_actions)


def test_safety_response_is_not_rewritten_by_llm(monkeypatch) -> None:
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.RAG_ENABLED", False)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Safety policy responses must not call LLM rewrite")

    monkeypatch.setattr("ai_runtime.llm.llm_generator.call_llm_json", fail_if_called)

    result = run_chatbot_graph(
        user_message="번아웃이 심하고 너무 무기력해요",
        context_type="MAIN",
        use_real_llm=True,
    )

    assert result.source == "safety_policy"
    assert result.intent == "mental_health_professional_support"
    assert result.safety_level == "professional_support"
    assert "전문 상담" in result.answer


def test_rag_disabled_uses_existing_rule_based_fallback(monkeypatch) -> None:
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.RAG_ENABLED", False)

    result = run_chatbot_graph(
        user_message="만성질환 예방은 어떻게 시작하면 좋나요?",
        context_type="GENERAL",
        use_real_llm=False,
        use_rag=True,
    )

    assert result.source == "rule_engine_unmatched"
    assert result.intent == "chronic_disease_prevention"
    assert result.reference_sources == []
    assert result.reference_summary is None
    assert result.trace_metadata["retrieval"]["reason"] == "rag_disabled"
    assert "진단이 아니" in result.answer


def test_langgraph_rag_node_wraps_keyword_context_as_reference_sources(monkeypatch) -> None:
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.RAG_ENABLED", True)

    result = run_chatbot_graph(
        user_message="공복혈당 관리 방법을 알려줘",
        context_type="MAIN",
        use_real_llm=False,
        use_rag=True,
    )

    assert result.source == "rag_llm"
    assert result.reference_sources
    assert any(source["id"] == "diabetes" for source in result.reference_sources)
    assert result.reference_summary
    assert "진단이 아니" in result.answer


@pytest.mark.parametrize(
    ("message", "expected_level"),
    [
        ("스트레스랑 불안 때문에 잠이 잘 안 와요", "self_care"),
        ("우울하고 번아웃이 심해요", "professional_support"),
    ],
)
def test_non_crisis_mental_health_paths_keep_self_care_actions(message: str, expected_level: str) -> None:
    result = run_chatbot_graph(
        user_message=message,
        context_type="MAIN",
        use_real_llm=False,
    )

    assert result.source == "safety_policy"
    assert result.safety_level == expected_level
    assert result.recommended_actions
    assert "119" not in result.answer
    assert "112" not in result.answer


def test_stress_sleep_anxiety_is_not_over_classified_as_crisis() -> None:
    result = run_chatbot_graph(
        user_message="스트레스와 불안 때문에 잠이 잘 안 와요",
        context_type="MAIN",
        use_real_llm=False,
    )

    assert result.source == "safety_policy"
    assert result.intent == "mental_health_self_care_guidance"
    assert result.safety_level == "self_care"
    assert "정신건강 관련 자기관리 챌린지" in result.answer
    assert "챌린지 추천보다 안전 확보가 우선" not in result.answer


def test_depression_burnout_support_is_distinct_from_crisis_response() -> None:
    result = run_chatbot_graph(
        user_message="우울하고 번아웃이 심해서 무기력해요",
        context_type="MAIN",
        use_real_llm=False,
    )

    assert result.source == "safety_policy"
    assert result.intent == "mental_health_professional_support"
    assert result.safety_level == "professional_support"
    assert "전문 상담" in result.answer
    assert "자기관리 챌린지" in result.answer
    assert "챌린지 추천보다 안전 확보가 우선" not in result.answer
    assert "119" not in result.answer


def test_trace_sanitizing_masks_personal_and_sensitive_values() -> None:
    raw_message = "test@example.com 010-1234-5678 공복혈당 132 혈압 145/90 콜레스테롤: 210"

    sanitized = sanitize_for_trace(raw_message, limit=200)

    assert "test@example.com" not in sanitized
    assert "010-1234-5678" not in sanitized
    assert "132" not in sanitized
    assert "145/90" not in sanitized
    assert "210" not in sanitized
    assert "[email]" in sanitized
    assert "[phone]" in sanitized
    assert "[health_value]" in sanitized


def test_trace_metadata_uses_sanitized_preview_without_raw_sensitive_values(monkeypatch) -> None:
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.RAG_ENABLED", False)

    result = run_chatbot_graph(
        user_message="test@example.com 010-1234-5678 공복혈당 132 관리 방법",
        context_type="MAIN",
        use_real_llm=False,
    )

    preview = result.trace_metadata["sanitized_message_preview"]
    assert "test@example.com" not in preview
    assert "010-1234-5678" not in preview
    assert "132" not in preview
    assert "[email]" in preview
    assert "[phone]" in preview
    assert "[health_value]" in preview


def test_langfuse_disabled_trace_helper_does_not_require_server(monkeypatch) -> None:
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.LANGFUSE_ENABLED", False)

    traced = trace_graph_node(
        "test_node",
        {
            "user_message": "혈당 관리",
            "user_context": {},
            "context_type": "MAIN",
            "intent": None,
            "safety_level": None,
            "safety_response": None,
            "should_bypass_llm": False,
            "retrieved_docs": [],
            "reference_sources": [],
            "reference_summary": None,
            "llm_answer": None,
            "final_answer": None,
            "recommended_actions": [],
            "fallback_reason": None,
            "metadata": {"graph_version": "test"},
            "trace_metadata": {"sanitized_message_preview": "혈당 관리"},
            "source": "langgraph_chatbot",
            "caution_message": "",
            "is_safe": True,
            "safety_result": {},
            "use_real_llm": False,
            "use_rag": False,
        },
    )

    assert traced is False
