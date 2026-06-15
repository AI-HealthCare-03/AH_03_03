from __future__ import annotations

import pytest

from ai_runtime.llm.rag.embeddings import MockEmbeddingProvider, OpenAIEmbeddingProvider
from ai_runtime.llm.rag.tracing import build_embedding_rag_trace_metadata, build_vector_rag_trace_metadata
from ai_runtime.llm.rag.vector_retriever import VectorRagRetriever, _vector_to_pgvector_literal


class FakeVectorConnection:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.query: str | None = None
        self.values: list | None = None

    async def execute_query_dict(self, query: str, values: list | None = None) -> list[dict]:
        self.query = query
        self.values = values
        return self.rows


@pytest.mark.asyncio
async def test_vector_retriever_uses_pgvector_cosine_sql_and_binding() -> None:
    connection = FakeVectorConnection([_row("chunk-1")])
    retriever = VectorRagRetriever(
        embedding_provider=MockEmbeddingProvider(model_name="mock-model", dimension=8),
        connection=connection,
    )

    result = await retriever.retrieve(query="고혈압 저염 식단", top_k=3, disease_code="HTN", source_key="hypertension")

    assert result.strategy == "vector"
    assert result.documents
    assert connection.query is not None
    sql = " ".join(connection.query.split())
    assert "c.embedding <=> $1::vector" in sql
    assert "1 - (c.embedding <=> $1::vector) AS score" in sql
    assert "c.is_active = true" in sql
    assert "d.is_active = true" in sql
    assert "c.embedding IS NOT NULL" in sql
    assert "c.embedding_content_hash = c.content_hash" in sql
    assert "c.embedding_provider = $2" in sql
    assert "c.embedding_model = $3" in sql
    assert "c.embedding_dimension = $4" in sql
    assert "UPPER(d.disease_code) = $5" in sql
    assert "d.source_key = $6" in sql
    assert "LIMIT $7" in sql
    assert connection.values is not None
    assert isinstance(connection.values[0], str)
    assert connection.values[0].startswith("[")
    assert connection.values[0].endswith("]")
    assert connection.values[1:] == ["mock", "mock-model", 8, "HTN", "hypertension", 3]


@pytest.mark.asyncio
async def test_vector_retriever_returns_retrieval_result_schema() -> None:
    connection = FakeVectorConnection([_row("chunk-1", score=0.91)])
    retriever = VectorRagRetriever(
        embedding_provider=MockEmbeddingProvider(model_name="mock-model", dimension=8),
        connection=connection,
    )

    result = await retriever.retrieve(query="혈압 나트륨", top_k=2)

    assert result.strategy == "vector"
    assert result.fallback_reason is None
    assert result.documents[0].content == "저염 식습관 참고 문서 chunk"
    assert result.documents[0].title == "고혈압 식생활"
    assert result.documents[0].score == 0.91
    metadata = result.documents[0].metadata
    assert metadata["id"] == "chunk-1"
    assert metadata["chunk_key"] == "chunk-1"
    assert metadata["document_key"] == "rag:hypertension:hypertension.md"
    assert metadata["source_key"] == "hypertension"
    assert metadata["disease_code"] == "HTN"
    assert metadata["retriever_strategy"] == "vector"
    assert metadata["embedding_provider"] == "mock"
    assert metadata["embedding_model"] == "mock-model"
    assert metadata["embedding_dimension"] == 8
    assert result.contexts[0].metadata["retriever_strategy"] == "vector"
    assert result.reference_sources
    assert result.reference_summary


@pytest.mark.asyncio
async def test_vector_retriever_filters_issue_keys_from_metadata_without_extra_db_call() -> None:
    connection = FakeVectorConnection(
        [
            _row("chunk-1", issue_keys=["sodium_high"]),
            _row("chunk-2", issue_keys=["sugar_high"]),
        ]
    )
    retriever = VectorRagRetriever(
        embedding_provider=MockEmbeddingProvider(model_name="mock-model", dimension=8),
        connection=connection,
    )

    result = await retriever.retrieve(query="나트륨", top_k=2, issue_keys=["sodium_high"])

    assert [document.metadata["chunk_key"] for document in result.documents] == ["chunk-1"]


@pytest.mark.asyncio
async def test_vector_retriever_supports_openai_provider_with_injected_fake_client() -> None:
    connection = FakeVectorConnection([_row("chunk-1")])
    retriever = VectorRagRetriever(
        embedding_provider=OpenAIEmbeddingProvider(
            model_name="text-embedding-3-small",
            dimension=3,
            client=FakeOpenAIEmbeddingClient([[0.1, 0.2, 0.3]]),
        ),
        connection=connection,
    )

    result = await retriever.retrieve(query="fake client만 사용합니다.", top_k=2)

    assert result.documents
    assert result.fallback_reason is None
    assert connection.query is not None
    assert connection.values is not None
    assert connection.values[1:4] == ["openai", "text-embedding-3-small", 3]
    assert result.trace_metadata["embedding_provider"] == "openai"
    assert result.trace_metadata["fallback_used"] is False


@pytest.mark.asyncio
async def test_vector_retriever_returns_fallback_when_embedding_disabled() -> None:
    connection = FakeVectorConnection([_row("chunk-1")])
    retriever = VectorRagRetriever(embedding_provider=None, connection=connection)

    result = await retriever.retrieve(query="혈당", top_k=2)

    assert result.documents == []
    assert result.fallback_reason == "embedding_disabled"
    assert connection.query is None
    assert result.trace_metadata["embedding_search"] is True


@pytest.mark.asyncio
async def test_vector_retriever_returns_fallback_when_embedding_provider_fails() -> None:
    connection = FakeVectorConnection([_row("chunk-1")])
    retriever = VectorRagRetriever(embedding_provider=FailingEmbeddingProvider(), connection=connection)

    result = await retriever.retrieve(query="혈압", top_k=2)

    assert result.documents == []
    assert result.fallback_reason == "embedding_query_failed:RuntimeError"
    assert connection.query is None
    assert result.trace_metadata["embedding_provider"] == "openai"


def test_vector_literal_uses_pgvector_format() -> None:
    assert _vector_to_pgvector_literal([0.1, -0.2, 1.0]) == "[0.1,-0.2,1]"


def test_vector_trace_metadata_does_not_include_chunk_content() -> None:
    from ai_runtime.llm.rag.retriever import RetrievedDocument

    document = RetrievedDocument(
        content="문서 본문 전체가 trace에 들어가면 안 됩니다.",
        title="고혈압 식생활",
        source_name="대한고혈압학회",
        url="https://example.test/htn",
        metadata={"chunk_key": "chunk-1", "source_type": "official_society", "status": "candidate_unreviewed"},
        score=0.87,
    )

    metadata = build_vector_rag_trace_metadata(
        query="고혈압 나트륨 식단",
        top_k=3,
        documents=[document],
        candidate_count=5,
        disease_code="HTN",
        source_key="hypertension",
        issue_keys=["sodium_high"],
        embedding_provider="mock",
        embedding_model="mock-model",
        embedding_dimension=8,
        latency_ms=1.2,
        fallback_reason=None,
    )

    serialized = str(metadata)
    assert "문서 본문 전체" not in serialized
    assert metadata["query_preview"] == "고혈압 나트륨 식단"
    assert metadata["query_length"] == len("고혈압 나트륨 식단")
    assert metadata["chunk_keys"] == ["chunk-1"]
    assert metadata["retrieved_sources"][0]["score"] == 0.87
    assert "content" not in metadata["retrieved_sources"][0]


def test_embedding_trace_metadata_does_not_include_content_vector_or_api_key() -> None:
    metadata = build_embedding_rag_trace_metadata(
        provider="openai",
        model="text-embedding-3-small",
        dimension=1536,
        batch_size=64,
        chunk_count=24,
        failed_count=0,
        estimated_char_count=1234,
        apply=True,
        latency_ms=10.5,
        fallback_reason=None,
    )

    serialized = str(metadata)
    assert "sk-" not in serialized
    assert "embedding_vector" not in metadata
    assert "content" not in metadata
    assert metadata["embedding_vector_logged"] is False
    assert metadata["chunk_content_logged"] is False
    assert metadata["provider"] == "openai"


def _row(chunk_key: str, *, score: float = 0.82, issue_keys: list[str] | None = None) -> dict:
    return {
        "chunk_key": chunk_key,
        "content": "저염 식습관 참고 문서 chunk",
        "content_hash": "hash-1",
        "section_title": "식생활",
        "metadata": {
            "source_org": "대한고혈압학회",
            "source_type": "official_society",
            "topic_tags": ["나트륨", "저염"],
            "issue_keys": issue_keys or ["sodium_high"],
            "status": "candidate_unreviewed",
        },
        "embedding_model": "mock-model",
        "embedding_provider": "mock",
        "embedding_dimension": 8,
        "document_key": "rag:hypertension:hypertension.md",
        "source_key": "hypertension",
        "disease_code": "HTN",
        "title": "고혈압 식생활",
        "document_url": "https://example.test/htn",
        "score": score,
    }


class FakeOpenAIEmbeddingClient:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self._embeddings = embeddings
        self.embeddings = self
        self.calls: list[dict] = []

    def create(self, *, model: str, input: list[str]) -> dict:
        self.calls.append({"model": model, "input": input})
        return {"data": [{"index": index, "embedding": embedding} for index, embedding in enumerate(self._embeddings)]}


class FailingEmbeddingProvider:
    provider_name = "openai"
    model_name = "text-embedding-3-small"
    dimension = 1536

    def embed_text(self, text: str) -> list[float]:
        raise RuntimeError("embedding unavailable")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding unavailable")
