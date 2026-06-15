from __future__ import annotations

from dataclasses import dataclass

import pytest

from ai_runtime.llm.rag.embeddings import MockEmbeddingProvider, OpenAIEmbeddingProvider
from scripts.rag.embed_rag_chunks import (
    RagChunkEmbeddingCandidate,
    build_embedding_provider_from_options,
    dry_run_embed_rag_chunks,
)


@dataclass(frozen=True)
class FakeChunkRow:
    chunk_key: str
    content: str
    source_key: str = "hypertension"
    document_key: str = "rag:hypertension:hypertension.md"
    is_active: bool = True
    document_active: bool = True


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
                    document_key=row.document_key,
                    source_key=row.source_key,
                    metadata={"source_key": row.source_key},
                )
            )
            if limit is not None and len(candidates) >= limit:
                break
        return candidates


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
async def test_openai_provider_does_not_call_external_api_and_reports_failure() -> None:
    gateway = FakeEmbeddingGateway([_row("chunk-1")])

    summary = await dry_run_embed_rag_chunks(
        gateway=gateway,
        provider=OpenAIEmbeddingProvider(),
        provider_name="openai",
        model_name="text-embedding-3-small",
        dimension=1536,
        batch_size=64,
    )

    assert summary.total_candidate_chunks == 1
    assert summary.embedded_chunks == 0
    assert summary.failed_chunks == 1
    assert summary.db_write_performed is False
    assert "NotImplementedError" in summary.warnings[0]


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
    source_key: str = "hypertension",
    document_key: str = "rag:hypertension:hypertension.md",
    is_active: bool = True,
    document_active: bool = True,
) -> FakeChunkRow:
    return FakeChunkRow(
        chunk_key=chunk_key,
        content=content,
        source_key=source_key,
        document_key=document_key,
        is_active=is_active,
        document_active=document_active,
    )
