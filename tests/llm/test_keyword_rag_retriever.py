from ai_worker.llm.llm_client import record_langfuse_event
from ai_worker.llm.rag.keyword_retriever import retrieve_keyword_rag_contexts, retrieve_keyword_rag_matches
from ai_worker.llm.rag.tracing import build_keyword_rag_trace_metadata


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


def test_retrieve_food_nutrition_api_keywords() -> None:
    matches = retrieve_keyword_rag_matches(user_message="식품영양성분 API로 음식 영양성분을 확인하고 싶어요")
    source_ids = [match.source_id for match in matches]

    assert "food_nutrition_api" in source_ids


def test_include_safety_disclaimer() -> None:
    contexts = retrieve_keyword_rag_contexts(
        user_message="공복혈당이 높게 나왔어요",
        include_safety_disclaimer=True,
    )
    source_ids = [context.metadata["id"] for context in contexts]

    assert "diabetes" in source_ids
    assert "safety_disclaimer" in source_ids


def test_unknown_keyword_returns_empty_without_safety() -> None:
    matches = retrieve_keyword_rag_matches(user_message="전혀 관련 없는 문장입니다")

    assert matches == []


def test_unknown_keyword_can_return_safety_only() -> None:
    contexts = retrieve_keyword_rag_contexts(
        user_message="전혀 관련 없는 문장입니다",
        include_safety_disclaimer=True,
    )

    assert len(contexts) == 1
    assert contexts[0].metadata["id"] == "safety_disclaimer"


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


def test_langfuse_event_is_noop_without_env(monkeypatch) -> None:
    for key in ("LANGFUSE_ENABLED", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(key, raising=False)

    recorded = record_langfuse_event(
        name="rag.keyword_retrieval",
        input_payload={"query_length": 4},
        output_payload={"retrieved_source_count": 0},
        metadata={"source": "keyword_rag_poc"},
    )

    assert recorded is False


def test_langfuse_event_is_noop_when_disabled_even_with_env(monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://localhost:3000")

    recorded = record_langfuse_event(
        name="rag.keyword_retrieval",
        input_payload={"query_preview": "혈당", "query_length": 2},
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

    assert metadata["retrieved_source_ids"] == ["safety_disclaimer"]
    assert metadata["source_status"]["safety_disclaimer"] == "candidate_unreviewed"
    assert metadata["fallback"] is True
    assert metadata["fallback_reason"] == "safety_disclaimer_only"
