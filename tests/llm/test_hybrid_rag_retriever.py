from __future__ import annotations

import pytest

from ai_runtime.llm.rag.hybrid_retriever import (
    HYBRID_STRATEGY_KEYWORD_FIRST_VECTOR_FALLBACK,
    HYBRID_STRATEGY_KEYWORD_ONLY,
    HYBRID_STRATEGY_VECTOR_DISABLED,
    HybridRagRetriever,
    clamp_vector_score,
    deduplicate_retrieved_documents,
)
from ai_runtime.llm.rag.retriever import RagRetrievalResult, RetrievedDocument
from ai_runtime.llm.rag.tracing import build_hybrid_rag_trace_metadata


@pytest.mark.asyncio
async def test_keyword_only_does_not_call_vector_retriever() -> None:
    keyword = FakeRetriever([_document("kw-1", content="keyword document")])
    vector = FailingRetriever()
    retriever = HybridRagRetriever(keyword_retriever=keyword, vector_retriever=vector)

    result = await retriever.retrieve(
        query_text="혈압 관리",
        strategy=HYBRID_STRATEGY_KEYWORD_ONLY,
    )

    assert result.strategy == HYBRID_STRATEGY_KEYWORD_ONLY
    assert [document.metadata["id"] for document in result.documents] == ["kw-1"]
    assert keyword.called == 1
    assert vector.called == 0
    assert result.trace_metadata["retriever_strategy"] == HYBRID_STRATEGY_KEYWORD_ONLY


@pytest.mark.asyncio
async def test_vector_disabled_does_not_call_vector_retriever() -> None:
    keyword = FakeRetriever([_document("kw-1")])
    vector = FailingRetriever()
    retriever = HybridRagRetriever(keyword_retriever=keyword, vector_retriever=vector)

    result = await retriever.retrieve(
        query_text="혈당 관리",
        strategy=HYBRID_STRATEGY_VECTOR_DISABLED,
    )

    assert result.strategy == HYBRID_STRATEGY_VECTOR_DISABLED
    assert len(result.documents) == 1
    assert vector.called == 0
    assert result.trace_metadata["fallback_reason"] == "vector_disabled"


@pytest.mark.asyncio
async def test_keyword_result_above_threshold_skips_vector_fallback() -> None:
    keyword = FakeRetriever([_document("kw-1"), _document("kw-2")])
    vector = FailingRetriever()
    retriever = HybridRagRetriever(keyword_retriever=keyword, vector_retriever=vector)

    result = await retriever.retrieve(
        query_text="식이섬유",
        strategy=HYBRID_STRATEGY_KEYWORD_FIRST_VECTOR_FALLBACK,
        min_keyword_results=2,
    )

    assert [document.metadata["id"] for document in result.documents] == ["kw-1", "kw-2"]
    assert vector.called == 0
    assert result.trace_metadata["fallback_used"] is False
    assert result.trace_metadata["vector_returned_count"] == 0


@pytest.mark.asyncio
async def test_zero_keyword_result_calls_vector_fallback() -> None:
    keyword = FakeRetriever([])
    vector = FakeRetriever([_document("vec-1", chunk_key="chunk-1", content="vector document")])
    retriever = HybridRagRetriever(keyword_retriever=keyword, vector_retriever=vector)

    result = await retriever.retrieve(
        query_text="나트륨",
        strategy=HYBRID_STRATEGY_KEYWORD_FIRST_VECTOR_FALLBACK,
        min_keyword_results=1,
        disease_code="HTN",
        source_key="hypertension",
        issue_keys=["sodium_high"],
    )

    assert vector.called == 1
    assert [document.metadata["chunk_key"] for document in result.documents] == ["chunk-1"]
    assert result.trace_metadata["fallback_used"] is True
    assert result.trace_metadata["fallback_reason"] == "no_keyword_result"
    assert result.trace_metadata["keyword_returned_count"] == 0
    assert result.trace_metadata["vector_returned_count"] == 1


@pytest.mark.asyncio
async def test_keyword_below_threshold_calls_vector_fallback_and_deduplicates() -> None:
    keyword = FakeRetriever([_document("shared", chunk_key="chunk-shared"), _document("kw-2")])
    vector = FakeRetriever(
        [_document("shared-vector", chunk_key="chunk-shared"), _document("vec-2", chunk_key="chunk-2")]
    )
    retriever = HybridRagRetriever(keyword_retriever=keyword, vector_retriever=vector)

    result = await retriever.retrieve(
        query_text="비만 야식",
        strategy=HYBRID_STRATEGY_KEYWORD_FIRST_VECTOR_FALLBACK,
        min_keyword_results=3,
        top_k=5,
    )

    assert vector.called == 1
    assert [document.metadata.get("chunk_key") or document.metadata.get("id") for document in result.documents] == [
        "chunk-shared",
        "kw-2",
        "chunk-2",
    ]
    assert result.trace_metadata["merged_count"] == 3
    assert result.trace_metadata["selected_chunk_keys"] == ["chunk-shared", "kw-2", "chunk-2"]


@pytest.mark.asyncio
async def test_vector_fallback_failure_keeps_keyword_result() -> None:
    keyword = FakeRetriever([_document("kw-1")])
    vector = FailingRetriever(raise_on_call=True)
    retriever = HybridRagRetriever(keyword_retriever=keyword, vector_retriever=vector)

    result = await retriever.retrieve(
        query_text="콜레스테롤",
        strategy=HYBRID_STRATEGY_KEYWORD_FIRST_VECTOR_FALLBACK,
        min_keyword_results=2,
    )

    assert vector.called == 1
    assert [document.metadata["id"] for document in result.documents] == ["kw-1"]
    assert result.trace_metadata["fallback_used"] is True
    assert result.trace_metadata["fallback_reason"] == "vector_retrieval_failed:RuntimeError"


def test_deduplicate_documents_prefers_chunk_key_then_id_then_fallback() -> None:
    documents = [
        _document("first", chunk_key="chunk-1"),
        _document("second", chunk_key="chunk-1"),
        _document("id-1"),
        _document("id-1"),
        RetrievedDocument(content="same content body", title="Same", url="https://example.test"),
        RetrievedDocument(content="same content body", title="Same", url="https://example.test"),
    ]

    deduplicated = deduplicate_retrieved_documents(documents)

    assert [document.title for document in deduplicated] == ["first", "id-1", "Same"]


def test_vector_score_clamp_documents_future_normalization_policy() -> None:
    assert clamp_vector_score(None) is None
    assert clamp_vector_score(-0.2) == 0.0
    assert clamp_vector_score(0.42) == 0.42
    assert clamp_vector_score(1.8) == 1.0


def test_hybrid_trace_metadata_does_not_include_chunk_content() -> None:
    document = _document("chunk-doc", chunk_key="chunk-1", content="trace에 들어가면 안 되는 본문")

    metadata = build_hybrid_rag_trace_metadata(
        query="고혈압 저염 식단",
        strategy=HYBRID_STRATEGY_KEYWORD_FIRST_VECTOR_FALLBACK,
        top_k=3,
        documents=[document],
        keyword_returned_count=0,
        vector_returned_count=1,
        fallback_used=True,
        fallback_reason="no_keyword_result",
        disease_code="HTN",
        source_key="hypertension",
        issue_keys=["sodium_high"],
        latency_ms=1.2,
        embedding_provider="mock",
        embedding_model="mock-model",
    )

    serialized = str(metadata)
    assert "trace에 들어가면 안 되는 본문" not in serialized
    assert metadata["query_preview"] == "고혈압 저염 식단"
    assert metadata["query_length"] == len("고혈압 저염 식단")
    assert metadata["selected_chunk_keys"] == ["chunk-1"]
    assert "content" not in metadata["retrieved_sources"][0]


class FakeRetriever:
    def __init__(self, documents: list[RetrievedDocument], *, fallback_reason: str | None = None) -> None:
        self.documents = documents
        self.fallback_reason = fallback_reason
        self.called = 0
        self.calls: list[dict] = []

    async def retrieve(self, **kwargs) -> RagRetrievalResult:
        self.called += 1
        self.calls.append(kwargs)
        return RagRetrievalResult(
            documents=self.documents,
            strategy="fake",
            fallback_reason=self.fallback_reason,
            trace_metadata={
                "embedding_provider": "mock",
                "embedding_model": "mock-model",
            },
        )


class FailingRetriever:
    def __init__(self, *, raise_on_call: bool = False) -> None:
        self.called = 0
        self.raise_on_call = raise_on_call

    async def retrieve(self, **kwargs) -> RagRetrievalResult:
        self.called += 1
        if self.raise_on_call:
            raise RuntimeError("vector unavailable")
        raise AssertionError("vector retriever should not be called")


def _document(
    document_id: str,
    *,
    chunk_key: str | None = None,
    content: str = "retrieved document",
) -> RetrievedDocument:
    metadata = {"id": document_id, "source_type": "official_society", "status": "candidate_unreviewed"}
    if chunk_key:
        metadata["chunk_key"] = chunk_key
    return RetrievedDocument(
        content=content,
        title=document_id,
        source_name="source",
        url=f"https://example.test/{document_id}",
        metadata=metadata,
        score=0.8,
    )
