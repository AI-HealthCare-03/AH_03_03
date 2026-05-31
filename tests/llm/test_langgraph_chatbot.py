from __future__ import annotations

import pytest

from ai_runtime.llm.graph import run_chatbot_graph
from ai_runtime.llm.graph.builder import build_health_chatbot_graph


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
