from ai_runtime.llm.llm_client import record_langfuse_event
from ai_runtime.llm.rag.keyword_retriever import retrieve_keyword_rag_contexts, retrieve_keyword_rag_matches
from ai_runtime.llm.rag.retriever import KeywordRagRetriever, disabled_rag_retrieval_result
from ai_runtime.llm.rag.source_trust import source_trust_level_for_type
from ai_runtime.llm.rag.tracing import build_keyword_rag_trace_metadata, trace_keyword_rag_retrieval


def test_retrieve_by_disease_type() -> None:
    matches = retrieve_keyword_rag_matches(disease_type="DIABETES")

    assert matches
    assert matches[0].source_id == "diabetes"
    assert matches[0].document.metadata.status == "candidate_unreviewed"


def test_retrieve_by_hypertension_keywords() -> None:
    contexts = retrieve_keyword_rag_contexts(user_message="수축기 혈압과 이완기 혈압이 높아요")

    assert contexts
    assert contexts[0].metadata["id"] == "hypertension"
    assert contexts[0].metadata["source_org"] == "대한고혈압학회"
    assert contexts[0].metadata["status"] == "candidate_unreviewed"
    assert contexts[0].metadata["source_type"] == "official_society"
    assert contexts[0].metadata["source_trust_level"] == "public_health_agency"


def test_retrieve_food_nutrition_api_keywords() -> None:
    matches = retrieve_keyword_rag_matches(user_message="식품영양성분 API로 음식 영양성분을 확인하고 싶어요")
    source_ids = [match.source_id for match in matches]

    assert "food_nutrition_api" not in source_ids


def test_include_safety_disclaimer() -> None:
    contexts = retrieve_keyword_rag_contexts(
        user_message="공복혈당이 높게 나왔어요",
        include_safety_disclaimer=True,
    )
    source_ids = [context.metadata["id"] for context in contexts]

    assert "diabetes" in source_ids
    assert "diet_caution" in source_ids


def test_keyword_retriever_adapter_returns_interface_result() -> None:
    result = KeywordRagRetriever().retrieve(
        query="공복혈당이 높게 나왔어요",
        top_k=2,
        include_safety_disclaimer=True,
    )

    assert result.strategy == "local_markdown_keyword_match"
    assert result.documents
    assert result.contexts
    assert result.reference_sources
    assert result.reference_summary
    assert "diabetes" in result.trace_metadata["document_ids"]
    assert result.trace_metadata["vector_rag"] is False
    assert result.trace_metadata["embedding_search"] is False
    assert "official_guideline" in result.trace_metadata["source_trust_levels"]
    assert all("content" not in document for document in result.trace_metadata["documents"])
    assert all("source_trust_level" in document for document in result.trace_metadata["documents"])


def test_keyword_retriever_adapter_no_result_contract_without_safety() -> None:
    result = KeywordRagRetriever().retrieve(
        query="전혀 관련 없는 문장입니다",
        top_k=2,
        include_safety_disclaimer=False,
    )

    assert result.documents == []
    assert result.contexts == []
    assert result.reference_sources == []
    assert result.reference_summary is None
    assert result.fallback_reason == "no_result"
    assert result.trace_metadata["document_count"] == 0
    assert result.trace_metadata["source_trust_levels"] == []
    assert result.trace_metadata["fallback"] is True
    assert result.trace_metadata["fallback_reason"] == "no_result"


def test_source_trust_level_mapping_for_rag_sources() -> None:
    assert source_trust_level_for_type("clinical_guideline") == "official_guideline"
    assert source_trust_level_for_type("nutrition_guideline") == "official_guideline"
    assert source_trust_level_for_type("official_society") == "public_health_agency"
    assert source_trust_level_for_type("public_data_api") == "public_health_agency"
    assert source_trust_level_for_type("safety_policy") == "internal_policy"
    assert source_trust_level_for_type("unexpected_source") == "unknown"


def test_disabled_rag_retrieval_result_is_empty_and_safe() -> None:
    result = disabled_rag_retrieval_result()

    assert result.documents == []
    assert result.contexts == []
    assert result.reference_sources == []
    assert result.reference_summary is None
    assert result.strategy == "disabled"
    assert result.fallback_reason == "rag_disabled"
    assert result.trace_metadata["enabled"] is False
    assert result.trace_metadata["document_count"] == 0


def test_unknown_keyword_returns_empty_without_safety() -> None:
    matches = retrieve_keyword_rag_matches(user_message="전혀 관련 없는 문장입니다")

    assert matches == []


def test_unknown_keyword_can_return_safety_only() -> None:
    contexts = retrieve_keyword_rag_contexts(
        user_message="전혀 관련 없는 문장입니다",
        include_safety_disclaimer=True,
    )

    assert len(contexts) == 1
    assert contexts[0].metadata["id"] == "diet_caution"


def test_keyword_rag_trace_metadata_contains_source_metadata() -> None:
    contexts = retrieve_keyword_rag_contexts(
        user_message="공복혈당 관리",
        disease_type="DIABETES",
        include_safety_disclaimer=True,
    )
    metadata = build_keyword_rag_trace_metadata(
        query="공복혈당 관리",
        disease_type="DIABETES",
        contexts=contexts,
        top_k=3,
        include_safety_disclaimer=True,
    )

    assert metadata["trace_version"] == "keyword_rag_poc_v1"
    assert metadata["prompt_version"] == "keyword_rag_context_v1"
    assert metadata["retrieval_strategy"] == "local_markdown_keyword_match"
    assert metadata["vector_rag"] is False
    assert metadata["embedding_search"] is False
    assert metadata["llm_call"] is False
    assert metadata["retrieved_source_count"] == len(contexts)
    assert "diabetes" in metadata["retrieved_source_ids"]
    assert metadata["source_status"]["diabetes"] == "candidate_unreviewed"
    assert metadata["top_k"] == 3
    assert metadata["fallback"] is False
    assert any(source["id"] == "diabetes" for source in metadata["retrieved_sources"])


def test_keyword_rag_trace_is_noop_when_rag_disabled(monkeypatch) -> None:
    import ai_runtime.llm.rag.tracing as tracing

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Langfuse event should be skipped when RAG is disabled")

    monkeypatch.setattr(tracing.config, "RAG_ENABLED", False)
    monkeypatch.setattr(tracing, "record_langfuse_event", fail_if_called)

    recorded = trace_keyword_rag_retrieval(
        query="공복혈당 관리",
        disease_type="DIABETES",
        contexts=[],
        top_k=2,
        include_safety_disclaimer=True,
    )

    assert recorded is False


def test_langfuse_event_is_noop_without_env(monkeypatch) -> None:
    import ai_runtime.llm.llm_client as llm_client

    monkeypatch.setattr(llm_client.config, "LANGFUSE_ENABLED", False)
    monkeypatch.setattr(llm_client.config, "LANGFUSE_PUBLIC_KEY", None)
    monkeypatch.setattr(llm_client.config, "LANGFUSE_SECRET_KEY", None)
    monkeypatch.setattr(llm_client.config, "LANGFUSE_BASE_URL", None)

    recorded = record_langfuse_event(
        name="rag.keyword_retrieval",
        input_payload={"query_length": 4},
        output_payload={"retrieved_source_count": 0},
        metadata={"source": "keyword_rag_poc"},
    )

    assert recorded is False


def test_langfuse_event_is_noop_when_disabled_even_with_env(monkeypatch) -> None:
    import ai_runtime.llm.llm_client as llm_client

    monkeypatch.setattr(llm_client.config, "LANGFUSE_ENABLED", False)
    monkeypatch.setattr(llm_client.config, "LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setattr(llm_client.config, "LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setattr(llm_client.config, "LANGFUSE_BASE_URL", "http://localhost:3000")

    recorded = record_langfuse_event(
        name="rag.keyword_retrieval",
        input_payload={"query_preview": "혈당", "query_length": 2},
        output_payload={"retrieved_source_count": 1},
        metadata={"source": "keyword_rag_poc"},
    )

    assert recorded is False


def test_langfuse_event_is_noop_when_client_fails(monkeypatch) -> None:
    import ai_runtime.llm.llm_client as llm_client

    class FailingLangfuse:
        def start_as_current_observation(self, *args, **kwargs):
            raise RuntimeError("langfuse unavailable")

    monkeypatch.setattr(llm_client, "build_langfuse_client", lambda: FailingLangfuse())

    recorded = llm_client.record_langfuse_event(
        name="rag.keyword_retrieval",
        input_payload={"query_preview": "혈당"},
        output_payload={"retrieved_source_count": 1},
        metadata={"source": "keyword_rag_poc"},
    )

    assert recorded is False


def test_keyword_rag_trace_metadata_marks_safety_only_fallback() -> None:
    contexts = retrieve_keyword_rag_contexts(
        user_message="전혀 관련 없는 문장입니다",
        include_safety_disclaimer=True,
    )
    metadata = build_keyword_rag_trace_metadata(
        query="전혀 관련 없는 문장입니다",
        disease_type=None,
        contexts=contexts,
        top_k=3,
        include_safety_disclaimer=True,
    )

    assert metadata["retrieved_source_ids"] == ["diet_caution"]
    assert metadata["source_status"]["diet_caution"] == "candidate_unreviewed"
    assert metadata["fallback"] is True
    assert metadata["fallback_reason"] == "safety_disclaimer_only"
