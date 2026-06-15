from __future__ import annotations

import pytest

from ai_runtime.llm.graph import run_chatbot_graph
from ai_runtime.llm.graph.builder import build_health_chatbot_graph
from ai_runtime.llm.graph.nodes import sanitize_for_trace, trace_graph_node
from ai_runtime.llm.prompt_templates import FALLBACK_SAFE_RESPONSE_PROMPT_VERSION, RAG_GROUNDED_ANSWER_PROMPT_VERSION
from ai_runtime.llm.rag.retriever import RagRetrievalResult, RetrievedDocument


@pytest.fixture(autouse=True)
def clear_graph_cache(monkeypatch):
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.RAG_RETRIEVAL_STRATEGY", "keyword_only")
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.MAIN_CHATBOT_RAG_STRATEGY", "keyword_only")
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
    assert isinstance(result.recommended_actions, list)
    assert all(isinstance(action, str) for action in result.recommended_actions)
    recommendation_trace = result.trace_metadata["recommended_actions"]
    assert recommendation_trace["recommended_action_count"] == len(result.recommended_actions)
    assert recommendation_trace["action_types"]
    assert recommendation_trace["top_recommendation_reason"]


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
    assert result.trace_metadata["retrieval"]["enabled"] is False
    assert result.trace_metadata["retrieval"]["reason"] == "rag_disabled"
    assert result.trace_metadata["retrieval"]["document_count"] == 0
    assert isinstance(result.metadata["graph_run_id"], str)
    assert result.trace_metadata["graph_run_id"] == result.metadata["graph_run_id"]
    node_durations = result.metadata["node_durations_ms"]
    assert set(result.metadata["node_path"]) <= set(node_durations)
    assert all(isinstance(value, int | float) for value in node_durations.values())
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
    assert "generation" not in result.trace_metadata
    assert result.metadata["node_path"] == [
        "normalize_input",
        "check_mental_health_safety",
        "build_recommended_actions",
        "format_final_response",
    ]
    assert "generate_llm_answer" not in result.metadata["node_durations_ms"]
    assert "grounding" not in result.trace_metadata
    recommendation_trace = result.trace_metadata["recommended_actions"]
    assert recommendation_trace["action_types"][0] == "emergency_support"
    assert recommendation_trace["actions"][0]["requires_professional_help"] is True
    assert "즉시 안전 확보" in recommendation_trace["top_recommendation_reason"]


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
    assert result.trace_metadata["generation"]["prompt_version"] == FALLBACK_SAFE_RESPONSE_PROMPT_VERSION
    assert result.trace_metadata["generation"]["llm_requested"] is False
    assert result.trace_metadata["generation"]["bypassed_rewrite"] is True


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
    retrieval_trace = result.trace_metadata["retrieval"]
    assert retrieval_trace["strategy"] == "local_markdown_keyword_match"
    assert "diabetes" in retrieval_trace["document_ids"]
    assert "official_guideline" in retrieval_trace["source_trust_levels"]
    assert all("content" not in document for document in retrieval_trace["documents"])
    assert all("source_trust_level" in document for document in retrieval_trace["documents"])
    grounding_trace = result.trace_metadata["grounding"]
    assert grounding_trace["retrieved_doc_count"] >= 1
    assert "official_guideline" in grounding_trace["source_trust_levels"]
    assert grounding_trace["grounding_status"] == "grounded"
    assert "diabetes" in grounding_trace["reference_source_ids"]
    assert result.trace_metadata["generation"]["prompt_version"] == RAG_GROUNDED_ANSWER_PROMPT_VERSION
    assert result.safety_result["metadata"]["trace_metadata"]["generation"]["prompt_version"] == (
        RAG_GROUNDED_ANSWER_PROMPT_VERSION
    )
    assert "진단이 아니" in result.answer


def test_langgraph_rag_node_uses_retriever_interface(monkeypatch) -> None:
    class FakeRetriever:
        def retrieve(self, *, query, disease_type=None, top_k=3, include_safety_disclaimer=False):
            assert query == "테스트 검색"
            assert disease_type is None
            assert top_k == 2
            assert include_safety_disclaimer is True
            return RagRetrievalResult(
                documents=[
                    RetrievedDocument(
                        title="테스트 문서",
                        content="테스트 문서 본문",
                        source_name="테스트 기관",
                        url="https://example.test/rag",
                        metadata={"id": "test_source", "status": "test", "score": 0.87},
                        score=0.87,
                    )
                ],
                reference_sources=[
                    {
                        "id": "test_source",
                        "title": "테스트 문서",
                        "source_org": "테스트 기관",
                        "source_url": "https://example.test/rag",
                        "year": None,
                        "status": "test",
                    }
                ],
                reference_summary="참고 정보: 테스트 문서 후보 문서를 함께 확인했습니다.",
                strategy="fake_retriever",
                trace_metadata={
                    "strategy": "fake_retriever",
                    "document_count": 1,
                    "document_ids": ["test_source"],
                    "source_types": ["테스트 기관"],
                    "documents": [
                        {
                            "id": "test_source",
                            "title": "테스트 문서",
                            "source_type": "테스트 기관",
                            "score": 0.87,
                        }
                    ],
                    "fallback": False,
                    "fallback_reason": None,
                    "vector_rag": False,
                    "embedding_search": False,
                },
            )

    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.RAG_ENABLED", True)
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.get_default_rag_retriever", lambda: FakeRetriever())

    result = run_chatbot_graph(
        user_message="테스트 검색",
        context_type="MAIN",
        use_real_llm=False,
        use_rag=True,
    )

    assert result.source == "rag_llm"
    assert result.reference_sources[0]["id"] == "test_source"
    assert result.trace_metadata["retrieval"]["strategy"] == "fake_retriever"
    assert result.trace_metadata["retrieval"]["documents"][0]["score"] == 0.87
    assert result.trace_metadata["grounding"]["grounding_status"] == "weak_reference"
    assert result.trace_metadata["grounding"]["source_trust_levels"] == ["unknown"]
    assert "근거 수준이 제한적" in result.answer


def test_main_chatbot_rag_strategy_default_is_keyword_only() -> None:
    from app.core import config

    assert config.RAG_RETRIEVAL_STRATEGY == "keyword_only"
    assert config.MAIN_CHATBOT_RAG_STRATEGY == "keyword_only"


def test_main_chatbot_keyword_first_strategy_skips_vector_when_keyword_is_sufficient(monkeypatch) -> None:
    class KeywordRetriever:
        def retrieve(self, *, query, disease_type=None, top_k=3, include_safety_disclaimer=False):
            return RagRetrievalResult(
                documents=[
                    RetrievedDocument(
                        title="혈당 관리 문서",
                        content="혈당 관리는 생활습관과 함께 살펴볼 수 있습니다.",
                        source_name="공식 기관",
                        url="https://example.test/diabetes",
                        metadata={"id": "diabetes", "status": "reviewed", "source_type": "official_guideline"},
                    )
                ],
                reference_sources=[
                    {
                        "id": "diabetes",
                        "title": "혈당 관리 문서",
                        "source_org": "공식 기관",
                        "source_url": "https://example.test/diabetes",
                        "source_type": "official_guideline",
                        "source_trust_level": "official_guideline",
                        "status": "reviewed",
                    }
                ],
                reference_summary="참고 정보: 혈당 관리 문서 후보 문서를 함께 확인했습니다.",
                strategy="fake_keyword",
                trace_metadata={"strategy": "fake_keyword", "document_count": 1, "documents": []},
            )

    class VectorRetriever:
        def retrieve(self, **kwargs):
            raise AssertionError("Vector fallback should not be called when keyword result is sufficient")

    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.RAG_ENABLED", True)
    monkeypatch.setattr(
        "ai_runtime.llm.graph.nodes.config.MAIN_CHATBOT_RAG_STRATEGY",
        "keyword_first_vector_fallback",
    )
    monkeypatch.setattr("ai_runtime.llm.graph.nodes._main_chatbot_vector_gate_enabled", lambda: True)
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.get_default_rag_retriever", lambda: KeywordRetriever())
    monkeypatch.setattr("ai_runtime.llm.graph.nodes._build_main_chatbot_vector_retriever", lambda: VectorRetriever())

    result = run_chatbot_graph(user_message="혈당 관리", context_type="MAIN", use_real_llm=False)

    retrieval_trace = result.trace_metadata["retrieval"]
    assert retrieval_trace["retriever_strategy"] == "keyword_first_vector_fallback"
    assert retrieval_trace["keyword_returned_count"] == 1
    assert retrieval_trace["vector_returned_count"] == 0
    assert retrieval_trace["fallback_used"] is False


def test_main_chatbot_keyword_first_strategy_calls_vector_when_keyword_is_empty(monkeypatch) -> None:
    class EmptyKeywordRetriever:
        def retrieve(self, *, query, disease_type=None, top_k=3, include_safety_disclaimer=False):
            return RagRetrievalResult(
                documents=[],
                reference_sources=[],
                reference_summary=None,
                strategy="empty_keyword",
                fallback_reason="no_result",
                trace_metadata={"strategy": "empty_keyword", "document_count": 0, "documents": []},
            )

    class VectorRetriever:
        def __init__(self) -> None:
            self.called = False

        async def retrieve(self, **kwargs):
            self.called = True
            return RagRetrievalResult(
                documents=[
                    RetrievedDocument(
                        title="고혈압 관리 문서",
                        content="나트륨을 줄이는 생활습관은 혈압 관리에 참고할 수 있습니다.",
                        source_name="공식 기관",
                        url="https://example.test/hypertension",
                        metadata={
                            "id": "rag:hypertension:section:000:chunk:0000",
                            "chunk_key": "rag:hypertension:section:000:chunk:0000",
                            "status": "reviewed",
                            "source_type": "official_guideline",
                            "source_trust_level": "official_guideline",
                            "retriever_strategy": "vector",
                            "embedding_provider": "openai",
                            "embedding_model": "text-embedding-3-small",
                        },
                        score=0.91,
                    )
                ],
                reference_sources=[],
                reference_summary=None,
                strategy="vector",
                trace_metadata={
                    "embedding_provider": "openai",
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dimension": 1536,
                },
            )

    vector = VectorRetriever()
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.RAG_ENABLED", True)
    monkeypatch.setattr(
        "ai_runtime.llm.graph.nodes.config.MAIN_CHATBOT_RAG_STRATEGY",
        "keyword_first_vector_fallback",
    )
    monkeypatch.setattr("ai_runtime.llm.graph.nodes._main_chatbot_vector_gate_enabled", lambda: True)
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.get_default_rag_retriever", lambda: EmptyKeywordRetriever())
    monkeypatch.setattr("ai_runtime.llm.graph.nodes._build_main_chatbot_vector_retriever", lambda: vector)

    result = run_chatbot_graph(user_message="혈압 나트륨", context_type="MAIN", use_real_llm=False)

    assert vector.called is True
    retrieval_trace = result.trace_metadata["retrieval"]
    assert retrieval_trace["retriever_strategy"] == "keyword_first_vector_fallback"
    assert retrieval_trace["keyword_returned_count"] == 0
    assert retrieval_trace["vector_returned_count"] == 1
    assert retrieval_trace["fallback_used"] is True
    assert retrieval_trace["fallback_reason"] == "no_keyword_result"
    assert "embedding" not in result.answer
    assert "chunk_key" not in result.answer


def test_main_chatbot_hybrid_parallel_calls_keyword_and_vector(monkeypatch) -> None:
    class KeywordRetriever:
        def retrieve(self, *, query, disease_type=None, top_k=3, include_safety_disclaimer=False):
            return RagRetrievalResult(
                documents=[
                    RetrievedDocument(
                        title="키워드 문서",
                        content="키워드 기반 건강관리 참고 문서입니다.",
                        source_name="공식 기관",
                        url="https://example.test/keyword",
                        metadata={"id": "keyword", "status": "reviewed", "source_type": "official_guideline"},
                        score=80,
                    )
                ],
                strategy="fake_keyword",
                trace_metadata={"strategy": "fake_keyword", "document_count": 1, "documents": []},
            )

    class VectorRetriever:
        def __init__(self) -> None:
            self.called = False

        async def retrieve(self, **kwargs):
            self.called = True
            return RagRetrievalResult(
                documents=[
                    RetrievedDocument(
                        title="벡터 문서",
                        content="벡터 기반 건강관리 참고 문서입니다.",
                        source_name="공식 기관",
                        url="https://example.test/vector",
                        metadata={
                            "id": "vector",
                            "chunk_key": "chunk-vector",
                            "status": "reviewed",
                            "source_type": "official_guideline",
                            "source_trust_level": "official_guideline",
                            "embedding_provider": "openai",
                            "embedding_model": "text-embedding-3-small",
                        },
                        score=0.95,
                    )
                ],
                strategy="vector",
                trace_metadata={
                    "embedding_provider": "openai",
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dimension": 1536,
                },
            )

    vector = VectorRetriever()
    captured_logs = []
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.RAG_ENABLED", True)
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.RAG_RETRIEVAL_STRATEGY", "hybrid_parallel")
    monkeypatch.setattr("ai_runtime.llm.graph.nodes._main_chatbot_vector_gate_enabled", lambda: True)
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.get_default_rag_retriever", lambda: KeywordRetriever())
    monkeypatch.setattr("ai_runtime.llm.graph.nodes._build_main_chatbot_vector_retriever", lambda: vector)
    monkeypatch.setattr(
        "ai_runtime.llm.graph.nodes.logger.info",
        lambda message, *args, **kwargs: captured_logs.append({"message": message, "args": args, **kwargs}),
    )

    result = run_chatbot_graph(user_message="건강관리 참고", context_type="MAIN", use_real_llm=False)

    assert vector.called is True
    retrieval_trace = result.trace_metadata["retrieval"]
    assert retrieval_trace["retriever_strategy"] == "hybrid_parallel"
    assert retrieval_trace["keyword_returned_count"] == 1
    assert retrieval_trace["vector_returned_count"] == 1
    assert retrieval_trace["merged_count"] == 2
    assert retrieval_trace["final_count"] == 2
    assert retrieval_trace["keyword_weight"] == 0.5
    assert retrieval_trace["vector_weight"] == 0.5
    assert "chunk-vector" not in result.answer
    assert "score" not in result.answer
    runtime_logs = [
        entry["extra"]["rag_retrieval"]
        for entry in captured_logs
        if str(entry["message"]).startswith("main_chatbot_rag_retrieval")
    ]
    assert runtime_logs
    assert runtime_logs[-1]["rag_strategy"] == "hybrid_parallel"
    assert runtime_logs[-1]["keyword_returned_count"] == 1
    assert runtime_logs[-1]["vector_returned_count"] == 1
    assert runtime_logs[-1]["merged_count"] == 2
    assert runtime_logs[-1]["final_count"] == 2


def test_main_chatbot_rag_llm_json_code_fence_is_not_exposed(monkeypatch) -> None:
    class KeywordRetriever:
        def retrieve(self, *, query, disease_type=None, top_k=3, include_safety_disclaimer=False):
            return RagRetrievalResult(
                documents=[
                    RetrievedDocument(
                        title="혈당 관리 문서",
                        content="혈당 관리는 단 음료와 정제 탄수화물을 줄이는 생활습관을 참고할 수 있습니다.",
                        source_name="질병관리청 국가건강정보포털",
                        url="https://health.kdca.go.kr/diabetes",
                        metadata={
                            "id": "diabetes",
                            "status": "reviewed",
                            "source_type": "official_guideline",
                            "source_trust_level": "official_guideline",
                        },
                    )
                ],
                reference_sources=[
                    {
                        "id": "diabetes",
                        "title": "혈당 관리 문서",
                        "source_org": "질병관리청 국가건강정보포털",
                        "source_url": "https://health.kdca.go.kr/diabetes",
                        "source_type": "official_guideline",
                        "source_trust_level": "official_guideline",
                        "status": "reviewed",
                    }
                ],
                reference_summary="참고 정보: 혈당 관리 문서를 함께 확인했습니다.",
                strategy="fake_keyword",
                trace_metadata={
                    "strategy": "fake_keyword",
                    "document_count": 1,
                    "document_ids": ["diabetes"],
                    "source_types": ["질병관리청 국가건강정보포털"],
                    "source_trust_levels": ["official_guideline"],
                    "documents": [],
                    "fallback": False,
                    "fallback_reason": None,
                },
            )

    def fake_call_llm_json(*args, **kwargs):
        return """
```json
{
  "answer": "혈당 관리가 필요한 경우 단 음료를 줄이고 식후 가벼운 활동을 참고해 보세요.",
  "intent": "internal",
  "source": "rag_llm",
  "is_safe": true
}
```
""".strip()

    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.RAG_ENABLED", True)
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.get_default_rag_retriever", lambda: KeywordRetriever())
    monkeypatch.setattr("ai_runtime.llm.rag_generator.call_llm_json", fake_call_llm_json)

    result = run_chatbot_graph(user_message="혈당 관리는 어떻게 하나요?", context_type="MAIN", use_real_llm=True)

    assert result.source == "rag_llm"
    assert "혈당 관리가 필요한 경우" in result.answer
    forbidden_terms = [
        '"answer":',
        '"intent":',
        '"source":',
        '"is_safe":',
        "chunk_key",
        "score",
        "embedding",
        "pgvector",
        "text-embedding",
        "vector retriever",
        "similarity",
        "```",
    ]
    for term in forbidden_terms:
        assert term not in result.answer


def test_langgraph_rag_no_result_falls_back_safely(monkeypatch) -> None:
    class EmptyRetriever:
        def retrieve(self, *, query, disease_type=None, top_k=3, include_safety_disclaimer=False):
            return RagRetrievalResult(
                documents=[],
                reference_sources=[],
                reference_summary=None,
                strategy="empty_retriever",
                fallback_reason="no_result",
                trace_metadata={
                    "strategy": "empty_retriever",
                    "document_count": 0,
                    "document_ids": [],
                    "source_types": [],
                    "documents": [],
                    "fallback": True,
                    "fallback_reason": "no_result",
                    "vector_rag": False,
                    "embedding_search": False,
                },
            )

    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.RAG_ENABLED", True)
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.get_default_rag_retriever", lambda: EmptyRetriever())

    result = run_chatbot_graph(
        user_message="관련 문서가 없는 질문",
        context_type="MAIN",
        use_real_llm=False,
        use_rag=True,
    )

    assert result.source == "rule_engine_unmatched"
    assert result.reference_sources == []
    assert result.reference_summary is None
    assert result.fallback_reason == "no_result"
    assert result.trace_metadata["retrieval"]["fallback"] is True
    assert result.trace_metadata["grounding"]["grounding_status"] == "no_reference"
    assert result.trace_metadata["grounding"]["reference_source_ids"] == []
    assert "근거 문서를 찾지 못했습니다" in result.answer


def test_langgraph_node_exception_records_fallback_state(monkeypatch) -> None:
    def fail_retrieval(state):
        raise RuntimeError("retriever secret=my-test-token failed")

    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.RAG_ENABLED", True)
    monkeypatch.setattr("ai_runtime.llm.graph.builder.retrieve_rag_context", fail_retrieval)

    result = run_chatbot_graph(
        user_message="혈당 관리 방법",
        context_type="MAIN",
        use_real_llm=False,
        use_rag=True,
    )

    graph_error = result.metadata["graph_error"]
    assert result.fallback_reason == "retrieve_rag_context_exception_fallback"
    assert result.metadata["graph_errors"] == [graph_error]
    assert result.trace_metadata["graph_error"] == graph_error
    assert graph_error["node"] == "retrieve_rag_context"
    assert graph_error["exception_type"] == "RuntimeError"
    assert graph_error["fallback_required"] is True
    assert "my-test-token" not in graph_error["exception_message"]
    assert result.answer


def test_langgraph_rag_low_trust_source_uses_conservative_grounding_copy(monkeypatch) -> None:
    class LowTrustRetriever:
        def retrieve(self, *, query, disease_type=None, top_k=3, include_safety_disclaimer=False):
            return RagRetrievalResult(
                documents=[
                    RetrievedDocument(
                        title="블로그 후보",
                        content="블로그 후보 본문",
                        source_name="알 수 없는 출처",
                        url="https://example.test/blog",
                        metadata={"id": "blog_source", "status": "test", "source_type": "unknown"},
                    )
                ],
                reference_sources=[
                    {
                        "id": "blog_source",
                        "title": "블로그 후보",
                        "source_org": "알 수 없는 출처",
                        "source_url": "https://example.test/blog",
                        "source_type": "unknown",
                        "source_trust_level": "unknown",
                        "status": "test",
                    }
                ],
                reference_summary="참고 정보: 블로그 후보 문서를 함께 확인했습니다.",
                strategy="low_trust_retriever",
                trace_metadata={
                    "strategy": "low_trust_retriever",
                    "document_count": 1,
                    "document_ids": ["blog_source"],
                    "source_types": ["알 수 없는 출처"],
                    "source_trust_levels": ["unknown"],
                    "documents": [
                        {
                            "id": "blog_source",
                            "title": "블로그 후보",
                            "source_type": "unknown",
                            "source_trust_level": "unknown",
                        }
                    ],
                    "fallback": False,
                    "fallback_reason": None,
                    "vector_rag": False,
                    "embedding_search": False,
                },
            )

    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.RAG_ENABLED", True)
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.get_default_rag_retriever", lambda: LowTrustRetriever())

    result = run_chatbot_graph(
        user_message="건강관리 질문",
        context_type="MAIN",
        use_real_llm=False,
        use_rag=True,
    )

    assert result.source == "rag_llm"
    assert "근거 수준이 제한적" in result.answer
    assert result.trace_metadata["grounding"]["grounding_status"] == "weak_reference"
    assert result.trace_metadata["grounding"]["fallback_reason"] == "rag_weak_reference"


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
    assert result.trace_metadata["recommended_actions"]["safety_level"] == expected_level


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
    recommendation_trace = result.trace_metadata["recommended_actions"]
    assert "self_care_challenge" in recommendation_trace["action_types"]
    assert all(action["requires_professional_help"] is False for action in recommendation_trace["actions"])


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
    recommendation_trace = result.trace_metadata["recommended_actions"]
    assert "professional_support" in recommendation_trace["action_types"]
    assert any(action["requires_professional_help"] for action in recommendation_trace["actions"])


def test_recommendation_actions_can_include_analysis_risk_metadata(monkeypatch) -> None:
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.RAG_ENABLED", False)

    result = run_chatbot_graph(
        user_message="운동 챌린지를 추천해줘",
        user_context={"risk_factors": [{"name": "공복혈당", "reason": "관리 참고"}]},
        context_type="CHALLENGE",
        use_real_llm=False,
    )

    assert isinstance(result.recommended_actions, list)
    assert any("챌린지" in action or "걷기" in action for action in result.recommended_actions)
    recommendation_trace = result.trace_metadata["recommended_actions"]
    assert recommendation_trace["recommended_action_count"] == len(result.recommended_actions)
    assert "challenge_browse" in recommendation_trace["action_types"]
    assert any(action["related_risk_factor"] == "공복혈당" for action in recommendation_trace["actions"])


def test_trace_sanitizing_masks_personal_and_sensitive_values() -> None:
    raw_message = (
        "test@example.com 010-1234-5678 900101-1234567 "
        "공복혈당 132 혈압 145/90 콜레스테롤: 210 "
        "OPENAI_API_KEY=sk-testsecret123456 password=super-secret-token"
    )

    sanitized = sanitize_for_trace(raw_message, limit=200)

    assert "test@example.com" not in sanitized
    assert "010-1234-5678" not in sanitized
    assert "900101-1234567" not in sanitized
    assert "132" not in sanitized
    assert "145/90" not in sanitized
    assert "210" not in sanitized
    assert "sk-testsecret123456" not in sanitized
    assert "super-secret-token" not in sanitized
    assert "[email]" in sanitized
    assert "[phone]" in sanitized
    assert "[rrn]" in sanitized
    assert "[secret]" in sanitized
    assert "[health_value]" in sanitized


def test_trace_sanitizing_limits_long_user_input_preview() -> None:
    raw_message = "혈당 관리 " + ("매우 긴 문장 " * 100)

    sanitized = sanitize_for_trace(raw_message)

    assert len(sanitized) <= 80
    assert "매우 긴 문장" in sanitized


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


def test_trace_graph_node_records_sanitized_metadata_without_raw_message(monkeypatch) -> None:
    captured = {}

    def fake_record_langfuse_event(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr("ai_runtime.llm.graph.nodes.record_langfuse_event", fake_record_langfuse_event)

    traced = trace_graph_node(
        "generate_llm_answer",
        {
            "user_message": "test@example.com 공복혈당 132",
            "user_context": {},
            "context_type": "MAIN",
            "intent": "diabetes_guidance",
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
            "metadata": {
                "graph_run_id": "graph-test-id",
                "graph_version": "test",
                "node_path": ["normalize_input", "generate_llm_answer"],
                "node_durations_ms": {"generate_llm_answer": 1.23},
            },
            "trace_metadata": {"sanitized_message_preview": "[email] 공복혈당 [health_value]"},
            "source": "rule_engine",
            "caution_message": "",
            "is_safe": True,
            "safety_result": {},
            "use_real_llm": False,
            "use_rag": False,
        },
        {"prompt_version": "health_chat_v1"},
    )

    assert traced is True
    assert captured["input_payload"] == {
        "message_length": len("test@example.com 공복혈당 132"),
        "context_type": "MAIN",
        "graph_run_id": "graph-test-id",
    }
    assert captured["metadata"]["graph_run_id"] == "graph-test-id"
    assert captured["metadata"]["node_path"] == ["normalize_input", "generate_llm_answer"]
    assert captured["metadata"]["node_duration_ms"] == 1.23
    assert captured["metadata"]["prompt_version"] == "health_chat_v1"
    assert captured["metadata"]["sanitized_message_preview"] == "[email] 공복혈당 [health_value]"
    assert "test@example.com" not in str(captured)
    assert "132" not in str(captured["metadata"])


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
