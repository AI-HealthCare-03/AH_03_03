from __future__ import annotations

from dataclasses import dataclass

import pytest

from ai_runtime.llm.rag.embeddings import MockEmbeddingProvider
from scripts.rag.embed_rag_chunks import (
    OrmEmbeddingWriteGateway,
    RagChunkEmbeddingCandidate,
    _vector_to_pgvector_literal,
    build_embedding_provider_from_options,
    dry_run_embed_rag_chunks,
    embed_rag_chunks,
)


@dataclass(frozen=True)
class FakeChunkRow:
    chunk_key: str
    content: str
    content_hash: str | None = "hash-1"
    source_key: str = "hypertension"
    document_key: str = "rag:hypertension:hypertension.md"
    is_active: bool = True
    document_active: bool = True
    embedding_is_null: bool = True
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    embedding_content_hash: str | None = None


class FakeEmbeddingGateway:
    def __init__(self, rows: list[FakeChunkRow]) -> None:
        self.rows = rows
        self.write_count = 0

    async def list_active_chunks(
        self,
        *,
        limit: int | None = None,
        only_source_key: str | None = None,
        only_document_key: str | None = None,
    ) -> list[RagChunkEmbeddingCandidate]:
        candidates: list[RagChunkEmbeddingCandidate] = []
        for row in self.rows:
            if not row.is_active or not row.document_active:
                continue
            if not row.content.strip():
                continue
            if only_source_key is not None and row.source_key != only_source_key:
                continue
            if only_document_key is not None and row.document_key != only_document_key:
                continue
            candidates.append(
                RagChunkEmbeddingCandidate(
                    chunk_key=row.chunk_key,
                    content=row.content,
                    content_hash=row.content_hash,
                    embedding_is_null=row.embedding_is_null,
                    embedding_provider=row.embedding_provider,
                    embedding_model=row.embedding_model,
                    embedding_dimension=row.embedding_dimension,
                    embedding_content_hash=row.embedding_content_hash,
                    document_key=row.document_key,
                    source_key=row.source_key,
                    metadata={"source_key": row.source_key},
                )
            )
            if limit is not None and len(candidates) >= limit:
                break
        return candidates


class FakeEmbeddingWriter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def write_embedding(
        self,
        *,
        chunk_key: str,
        vector: list[float],
        provider: str,
        model: str,
        dimension: int,
    ) -> int:
        self.calls.append(
            {
                "chunk_key": chunk_key,
                "vector": vector,
                "provider": provider,
                "model": model,
                "dimension": dimension,
            }
        )
        return 1


class FakeOpenAIEmbeddingProvider:
    provider_name = "openai"
    model_name = "text-embedding-3-small"

    def __init__(self, *, dimension: int) -> None:
        self.dimension = dimension

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1) / 10 for _ in range(self.dimension)] for index, _ in enumerate(texts)]


class FakeConnection:
    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self.query: str | None = None
        self.values: list[object] | None = None
        self.rows = rows if rows is not None else [{"chunk_key": "chunk-1"}]

    async def execute_query(self, query: str, values: list[object]) -> tuple[int, list[dict[str, object]]]:
        self.query = query
        self.values = values
        return 0, self.rows


@pytest.mark.asyncio
async def test_disabled_provider_returns_disabled_summary_without_db_write() -> None:
    gateway = FakeEmbeddingGateway([_row("chunk-1")])

    summary = await dry_run_embed_rag_chunks(
        gateway=gateway,
        provider=None,
        provider_name="disabled",
        model_name=None,
        dimension=None,
        batch_size=64,
    )

    assert summary.embedding_enabled is False
    assert summary.provider == "disabled"
    assert summary.total_candidate_chunks == 0
    assert summary.embedded_chunks == 0
    assert summary.db_write_performed is False
    assert summary.warnings == ["embedding disabled"]
    assert gateway.write_count == 0


@pytest.mark.asyncio
async def test_mock_provider_embeds_active_chunks_only() -> None:
    gateway = FakeEmbeddingGateway(
        [
            _row("chunk-1", content="첫 번째"),
            _row("chunk-2", content="두 번째"),
            _row("inactive", is_active=False),
            _row("inactive-document", document_active=False),
            _row("empty", content="   "),
        ]
    )

    summary = await dry_run_embed_rag_chunks(
        gateway=gateway,
        provider=MockEmbeddingProvider(dimension=8),
        provider_name="mock",
        model_name="mock-model",
        dimension=8,
        batch_size=64,
    )

    assert summary.embedding_enabled is True
    assert summary.total_candidate_chunks == 2
    assert summary.embedded_chunks == 2
    assert summary.skipped_chunks == 0
    assert summary.failed_chunks == 0
    assert summary.db_write_performed is False


@pytest.mark.asyncio
async def test_batch_size_controls_batch_count() -> None:
    gateway = FakeEmbeddingGateway([_row("chunk-1"), _row("chunk-2"), _row("chunk-3")])

    summary = await dry_run_embed_rag_chunks(
        gateway=gateway,
        provider=MockEmbeddingProvider(dimension=8),
        provider_name="mock",
        model_name="mock-model",
        dimension=8,
        batch_size=2,
    )

    assert summary.total_candidate_chunks == 3
    assert summary.batches == 2
    assert summary.embedded_chunks == 3


@pytest.mark.asyncio
async def test_limit_is_applied_before_embedding() -> None:
    gateway = FakeEmbeddingGateway([_row("chunk-1"), _row("chunk-2"), _row("chunk-3")])

    summary = await dry_run_embed_rag_chunks(
        gateway=gateway,
        provider=MockEmbeddingProvider(dimension=8),
        provider_name="mock",
        model_name="mock-model",
        dimension=8,
        batch_size=10,
        limit=2,
    )

    assert summary.total_candidate_chunks == 2
    assert summary.embedded_chunks == 2
    assert summary.sample_chunk_keys == ["chunk-1", "chunk-2"]


@pytest.mark.asyncio
async def test_only_source_key_filter_is_applied() -> None:
    gateway = FakeEmbeddingGateway(
        [
            _row("htn-1", source_key="hypertension"),
            _row("dm-1", source_key="diabetes", document_key="rag:diabetes:diabetes.md"),
        ]
    )

    summary = await dry_run_embed_rag_chunks(
        gateway=gateway,
        provider=MockEmbeddingProvider(dimension=8),
        provider_name="mock",
        model_name="mock-model",
        dimension=8,
        batch_size=10,
        only_source_key="diabetes",
    )

    assert summary.total_candidate_chunks == 1
    assert summary.sample_chunk_keys == ["dm-1"]


@pytest.mark.asyncio
async def test_only_document_key_filter_is_applied() -> None:
    gateway = FakeEmbeddingGateway(
        [
            _row("htn-1", document_key="rag:hypertension:hypertension.md"),
            _row("htn-2", document_key="rag:hypertension:other.md"),
        ]
    )

    summary = await dry_run_embed_rag_chunks(
        gateway=gateway,
        provider=MockEmbeddingProvider(dimension=8),
        provider_name="mock",
        model_name="mock-model",
        dimension=8,
        batch_size=10,
        only_document_key="rag:hypertension:other.md",
    )

    assert summary.total_candidate_chunks == 1
    assert summary.sample_chunk_keys == ["htn-2"]


@pytest.mark.asyncio
async def test_openai_provider_dry_run_uses_injected_fake_provider() -> None:
    gateway = FakeEmbeddingGateway([_row("chunk-1")])

    summary = await dry_run_embed_rag_chunks(
        gateway=gateway,
        provider=FakeOpenAIEmbeddingProvider(dimension=3),
        provider_name="openai",
        model_name="text-embedding-3-small",
        dimension=3,
        batch_size=64,
    )

    assert summary.total_candidate_chunks == 1
    assert summary.embedded_chunks == 1
    assert summary.failed_chunks == 0
    assert summary.db_write_performed is False


@pytest.mark.asyncio
async def test_apply_with_mock_provider_calls_fake_writer() -> None:
    gateway = FakeEmbeddingGateway([_row("chunk-1")])
    writer = FakeEmbeddingWriter()

    summary = await embed_rag_chunks(
        gateway=gateway,
        provider=MockEmbeddingProvider(model_name="mock-model", dimension=8),
        provider_name="mock",
        model_name="mock-model",
        dimension=8,
        batch_size=64,
        writer=writer,
        apply=True,
    )

    assert summary.planned_embedding_writes == 1
    assert summary.embedding_writes == 1
    assert summary.db_write_performed is True
    assert len(writer.calls) == 1
    assert writer.calls[0]["chunk_key"] == "chunk-1"
    assert writer.calls[0]["provider"] == "mock"


@pytest.mark.asyncio
async def test_current_embedding_is_skipped_by_default() -> None:
    gateway = FakeEmbeddingGateway(
        [
            _row(
                "chunk-1",
                embedding_is_null=False,
                embedding_provider="mock",
                embedding_model="mock-model",
                embedding_dimension=8,
                embedding_content_hash="hash-1",
            )
        ]
    )

    summary = await embed_rag_chunks(
        gateway=gateway,
        provider=MockEmbeddingProvider(model_name="mock-model", dimension=8),
        provider_name="mock",
        model_name="mock-model",
        dimension=8,
        batch_size=64,
    )

    assert summary.total_candidate_chunks == 1
    assert summary.planned_embedding_writes == 0
    assert summary.embedded_chunks == 0
    assert summary.skipped_existing_embeddings == 1
    assert summary.skipped_chunks == 1


@pytest.mark.asyncio
async def test_stale_embedding_content_hash_is_regenerated() -> None:
    gateway = FakeEmbeddingGateway(
        [
            _row(
                "chunk-1",
                embedding_is_null=False,
                embedding_provider="mock",
                embedding_model="mock-model",
                embedding_dimension=8,
                embedding_content_hash="old-hash",
            )
        ]
    )

    summary = await embed_rag_chunks(
        gateway=gateway,
        provider=MockEmbeddingProvider(model_name="mock-model", dimension=8),
        provider_name="mock",
        model_name="mock-model",
        dimension=8,
        batch_size=64,
    )

    assert summary.planned_embedding_writes == 1
    assert summary.stale_embeddings == 1
    assert summary.embedded_chunks == 1


@pytest.mark.asyncio
async def test_force_regenerates_current_embedding() -> None:
    gateway = FakeEmbeddingGateway(
        [
            _row(
                "chunk-1",
                embedding_is_null=False,
                embedding_provider="mock",
                embedding_model="mock-model",
                embedding_dimension=8,
                embedding_content_hash="hash-1",
            )
        ]
    )

    summary = await embed_rag_chunks(
        gateway=gateway,
        provider=MockEmbeddingProvider(model_name="mock-model", dimension=8),
        provider_name="mock",
        model_name="mock-model",
        dimension=8,
        batch_size=64,
        force=True,
    )

    assert summary.planned_embedding_writes == 1
    assert summary.forced_embeddings == 1
    assert summary.embedded_chunks == 1


@pytest.mark.asyncio
async def test_missing_content_hash_is_skipped_with_warning() -> None:
    gateway = FakeEmbeddingGateway([_row("chunk-1", content_hash=None)])

    summary = await embed_rag_chunks(
        gateway=gateway,
        provider=MockEmbeddingProvider(dimension=8),
        provider_name="mock",
        model_name="mock-model",
        dimension=8,
        batch_size=64,
    )

    assert summary.total_candidate_chunks == 1
    assert summary.planned_embedding_writes == 0
    assert summary.skipped_chunks == 1
    assert "content_hash is missing" in summary.warnings[0]


def test_vector_literal_uses_pgvector_array_format() -> None:
    assert _vector_to_pgvector_literal([0.1, -0.2, 1.0]) == "[0.1,-0.2,1]"


@pytest.mark.asyncio
async def test_raw_sql_writer_uses_parameter_binding_payload() -> None:
    connection = FakeConnection()
    writer = OrmEmbeddingWriteGateway(connection)

    affected = await writer.write_embedding(
        chunk_key="chunk-1",
        vector=[0.1, -0.2],
        provider="mock",
        model="mock-model",
        dimension=2,
    )

    assert affected == 1
    assert connection.query is not None
    assert "$1::vector" in connection.query
    assert "RETURNING chunk_key" in connection.query
    assert "0.1" not in connection.query
    assert connection.values == ["[0.1,-0.2]", "mock", "mock-model", 2, "chunk-1"]


@pytest.mark.asyncio
async def test_raw_sql_writer_returns_zero_when_returning_has_no_rows() -> None:
    connection = FakeConnection(rows=[])
    writer = OrmEmbeddingWriteGateway(connection)

    affected = await writer.write_embedding(
        chunk_key="missing-chunk",
        vector=[0.1, -0.2],
        provider="mock",
        model="mock-model",
        dimension=2,
    )

    assert affected == 0
    assert connection.query is not None
    assert "RETURNING chunk_key" in connection.query
    assert connection.values == ["[0.1,-0.2]", "mock", "mock-model", 2, "missing-chunk"]


@pytest.mark.asyncio
async def test_apply_with_raw_sql_writer_success_counts_returning_row() -> None:
    gateway = FakeEmbeddingGateway([_row("chunk-1")])
    writer = OrmEmbeddingWriteGateway(FakeConnection(rows=[{"chunk_key": "chunk-1"}]))

    summary = await embed_rag_chunks(
        gateway=gateway,
        provider=MockEmbeddingProvider(model_name="mock-model", dimension=8),
        provider_name="mock",
        model_name="mock-model",
        dimension=8,
        batch_size=64,
        writer=writer,
        apply=True,
    )

    assert summary.planned_embedding_writes == 1
    assert summary.embedding_writes == 1
    assert summary.failed_embedding_writes == 0
    assert summary.db_write_performed is True
    assert summary.warnings == []


@pytest.mark.asyncio
async def test_apply_with_raw_sql_writer_empty_returning_row_counts_failure() -> None:
    gateway = FakeEmbeddingGateway([_row("chunk-1")])
    writer = OrmEmbeddingWriteGateway(FakeConnection(rows=[]))

    summary = await embed_rag_chunks(
        gateway=gateway,
        provider=MockEmbeddingProvider(model_name="mock-model", dimension=8),
        provider_name="mock",
        model_name="mock-model",
        dimension=8,
        batch_size=64,
        writer=writer,
        apply=True,
    )

    assert summary.planned_embedding_writes == 1
    assert summary.embedding_writes == 0
    assert summary.failed_embedding_writes == 1
    assert summary.db_write_performed is False
    assert summary.warnings == ["chunk-1: embedding write affected 0 rows"]


@pytest.mark.asyncio
async def test_openai_provider_apply_uses_fake_provider_and_writer() -> None:
    gateway = FakeEmbeddingGateway([_row("chunk-1")])
    writer = FakeEmbeddingWriter()

    summary = await embed_rag_chunks(
        gateway=gateway,
        provider=FakeOpenAIEmbeddingProvider(dimension=3),
        provider_name="openai",
        model_name="text-embedding-3-small",
        dimension=3,
        batch_size=64,
        writer=writer,
        apply=True,
    )

    assert summary.total_candidate_chunks == 1
    assert summary.embedding_writes == 1
    assert summary.db_write_performed is True
    assert writer.calls[0]["provider"] == "openai"
    assert writer.calls[0]["model"] == "text-embedding-3-small"


@pytest.mark.asyncio
async def test_production_mock_apply_is_blocked() -> None:
    gateway = FakeEmbeddingGateway([_row("chunk-1")])
    writer = FakeEmbeddingWriter()

    summary = await embed_rag_chunks(
        gateway=gateway,
        provider=MockEmbeddingProvider(model_name="mock-model", dimension=8),
        provider_name="mock",
        model_name="mock-model",
        dimension=8,
        batch_size=64,
        writer=writer,
        apply=True,
        is_production=True,
    )

    assert summary.total_candidate_chunks == 0
    assert summary.embedding_writes == 0
    assert summary.db_write_performed is False
    assert writer.calls == []
    assert summary.warnings == ["Mock embedding apply is disabled in production"]


@pytest.mark.asyncio
async def test_mock_row_is_stale_for_openai_provider() -> None:
    gateway = FakeEmbeddingGateway(
        [
            _row(
                "chunk-1",
                embedding_is_null=False,
                embedding_provider="mock",
                embedding_model="mock-model",
                embedding_dimension=8,
                embedding_content_hash="hash-1",
            )
        ]
    )

    summary = await embed_rag_chunks(
        gateway=gateway,
        provider=FakeOpenAIEmbeddingProvider(dimension=3),
        provider_name="openai",
        model_name="text-embedding-3-small",
        dimension=3,
        batch_size=64,
    )

    assert summary.planned_embedding_writes == 1
    assert summary.stale_embeddings == 1
    assert summary.embedded_chunks == 1


def test_provider_override_mock_enables_embedding_without_env() -> None:
    provider, provider_name, model_name, dimension = build_embedding_provider_from_options(
        provider_override="mock",
        model_override="mock-model",
        dimension_override=11,
    )

    assert isinstance(provider, MockEmbeddingProvider)
    assert provider_name == "mock"
    assert model_name == "mock-model"
    assert dimension == 11


def test_provider_override_disabled_returns_no_provider() -> None:
    provider, provider_name, model_name, dimension = build_embedding_provider_from_options(provider_override="disabled")

    assert provider is None
    assert provider_name == "disabled"
    assert model_name is None
    assert dimension is None


def _row(
    chunk_key: str,
    *,
    content: str = "테스트 chunk 내용",
    content_hash: str | None = "hash-1",
    source_key: str = "hypertension",
    document_key: str = "rag:hypertension:hypertension.md",
    is_active: bool = True,
    document_active: bool = True,
    embedding_is_null: bool = True,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    embedding_dimension: int | None = None,
    embedding_content_hash: str | None = None,
) -> FakeChunkRow:
    return FakeChunkRow(
        chunk_key=chunk_key,
        content=content,
        content_hash=content_hash,
        source_key=source_key,
        document_key=document_key,
        is_active=is_active,
        document_active=document_active,
        embedding_is_null=embedding_is_null,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        embedding_content_hash=embedding_content_hash,
    )
