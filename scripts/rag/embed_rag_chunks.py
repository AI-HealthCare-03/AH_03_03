from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tortoise import Tortoise  # noqa: E402

from ai_runtime.llm.rag.embeddings import EmbeddingProvider, get_embedding_provider  # noqa: E402
from app.core import config  # noqa: E402
from app.core.db.databases import TORTOISE_ORM  # noqa: E402
from app.models.rag import RAGChunk  # noqa: E402


@dataclass(frozen=True)
class RagChunkEmbeddingCandidate:
    chunk_key: str
    content: str
    document_key: str | None = None
    source_key: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class RagChunkEmbeddingDryRunSummary:
    embedding_enabled: bool
    provider: str
    model: str | None
    dimension: int | None
    batch_size: int
    total_candidate_chunks: int = 0
    embedded_chunks: int = 0
    skipped_chunks: int = 0
    failed_chunks: int = 0
    batches: int = 0
    sample_chunk_keys: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    db_write_performed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RagChunkEmbeddingGateway(Protocol):
    async def list_active_chunks(
        self,
        *,
        limit: int | None = None,
        only_source_key: str | None = None,
        only_document_key: str | None = None,
    ) -> list[RagChunkEmbeddingCandidate]: ...


class OrmRagChunkEmbeddingGateway:
    async def list_active_chunks(
        self,
        *,
        limit: int | None = None,
        only_source_key: str | None = None,
        only_document_key: str | None = None,
    ) -> list[RagChunkEmbeddingCandidate]:
        rows = await RAGChunk.filter(is_active=True).select_related("document").order_by("document_id", "chunk_index")
        candidates: list[RagChunkEmbeddingCandidate] = []
        for row in rows:
            document = getattr(row, "document", None)
            if document is not None and not getattr(document, "is_active", True):
                continue
            candidate = _candidate_from_chunk(row)
            if not candidate.content.strip():
                continue
            if only_source_key is not None and candidate.source_key != only_source_key:
                continue
            if only_document_key is not None and candidate.document_key != only_document_key:
                continue
            candidates.append(candidate)
            if limit is not None and len(candidates) >= limit:
                break
        return candidates


async def dry_run_embed_rag_chunks(
    *,
    gateway: RagChunkEmbeddingGateway,
    provider: EmbeddingProvider | None,
    provider_name: str,
    model_name: str | None,
    dimension: int | None,
    batch_size: int,
    limit: int | None = None,
    only_source_key: str | None = None,
    only_document_key: str | None = None,
    fail_fast: bool = False,
) -> RagChunkEmbeddingDryRunSummary:
    normalized_batch_size = max(1, batch_size)
    summary = RagChunkEmbeddingDryRunSummary(
        embedding_enabled=provider is not None,
        provider=provider_name,
        model=model_name,
        dimension=dimension,
        batch_size=normalized_batch_size,
    )
    if provider is None:
        summary.warnings.append("embedding disabled")
        return summary

    candidates = await gateway.list_active_chunks(
        limit=limit,
        only_source_key=only_source_key,
        only_document_key=only_document_key,
    )
    summary.total_candidate_chunks = len(candidates)
    summary.sample_chunk_keys = [candidate.chunk_key for candidate in candidates[:5]]

    for batch in _batches(candidates, normalized_batch_size):
        if not batch:
            continue
        summary.batches += 1
        texts = [candidate.content for candidate in batch]
        try:
            vectors = provider.embed_texts(texts)
        except Exception as exc:  # noqa: BLE001 - dry-run should summarize provider failures.
            summary.failed_chunks += len(batch)
            summary.warnings.append(f"embedding batch failed: {type(exc).__name__}: {exc}")
            if fail_fast:
                raise
            continue
        for candidate, vector in zip(batch, vectors, strict=False):
            if len(vector) != provider.dimension:
                summary.failed_chunks += 1
                summary.warnings.append(
                    f"{candidate.chunk_key}: vector dimension {len(vector)} != provider dimension {provider.dimension}"
                )
                if fail_fast:
                    raise ValueError(summary.warnings[-1])
                continue
            summary.embedded_chunks += 1
    summary.skipped_chunks = max(0, summary.total_candidate_chunks - summary.embedded_chunks - summary.failed_chunks)
    return summary


def build_embedding_provider_from_options(
    *,
    provider_override: str | None = None,
    model_override: str | None = None,
    dimension_override: int | None = None,
) -> tuple[EmbeddingProvider | None, str, str | None, int | None]:
    provider_name = provider_override or config.RAG_EMBEDDING_PROVIDER
    enabled = config.RAG_EMBEDDING_ENABLED
    if provider_override is not None:
        enabled = provider_override.strip().lower() not in {"", "disabled", "none", "off"}
    settings = SimpleNamespace(
        RAG_EMBEDDING_ENABLED=enabled,
        RAG_EMBEDDING_PROVIDER=provider_name,
        RAG_EMBEDDING_MODEL=model_override or config.RAG_EMBEDDING_MODEL,
        RAG_EMBEDDING_DIMENSION=dimension_override or config.RAG_EMBEDDING_DIMENSION,
    )
    provider = get_embedding_provider(settings)
    if provider is None:
        return None, str(provider_name or "disabled"), None, None
    return provider, provider.provider_name, provider.model_name, provider.dimension


def _candidate_from_chunk(row: RAGChunk) -> RagChunkEmbeddingCandidate:
    metadata = row.metadata if isinstance(row.metadata, dict) else {}
    document = getattr(row, "document", None)
    document_key = getattr(document, "document_key", None)
    return RagChunkEmbeddingCandidate(
        chunk_key=row.chunk_key or f"rag_chunk:{row.id}",
        content=row.content or "",
        document_key=document_key or metadata.get("document_key"),
        source_key=metadata.get("source_key"),
        metadata=metadata,
    )


def _batches(candidates: list[RagChunkEmbeddingCandidate], batch_size: int) -> list[list[RagChunkEmbeddingCandidate]]:
    return [candidates[index : index + batch_size] for index in range(0, len(candidates), batch_size)]


def _print_summary(summary: RagChunkEmbeddingDryRunSummary, *, as_json: bool) -> None:
    payload = summary.to_dict()
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return
    print("RAG chunk embedding dry-run")
    print(f"- embedding_enabled: {summary.embedding_enabled}")
    print(f"- provider: {summary.provider}")
    print(f"- model: {summary.model}")
    print(f"- dimension: {summary.dimension}")
    print(f"- batch_size: {summary.batch_size}")
    print(f"- total_candidate_chunks: {summary.total_candidate_chunks}")
    print(f"- embedded_chunks: {summary.embedded_chunks}")
    print(f"- skipped_chunks: {summary.skipped_chunks}")
    print(f"- failed_chunks: {summary.failed_chunks}")
    print(f"- batches: {summary.batches}")
    print(f"- sample_chunk_keys: {summary.sample_chunk_keys}")
    print(f"- db_write_performed: {summary.db_write_performed}")
    if summary.warnings:
        print("- warnings:")
        for warning in summary.warnings:
            print(f"  - {warning}")


async def _run_cli(args: argparse.Namespace) -> RagChunkEmbeddingDryRunSummary:
    provider, provider_name, model_name, dimension = build_embedding_provider_from_options(
        provider_override=args.provider,
        model_override=args.model,
        dimension_override=args.dimension,
    )
    batch_size = args.batch_size or config.RAG_EMBEDDING_BATCH_SIZE
    if provider is None:
        return await dry_run_embed_rag_chunks(
            gateway=OrmRagChunkEmbeddingGateway(),
            provider=None,
            provider_name=provider_name,
            model_name=model_name,
            dimension=dimension,
            batch_size=batch_size,
        )

    await Tortoise.init(config=TORTOISE_ORM)
    try:
        return await dry_run_embed_rag_chunks(
            gateway=OrmRagChunkEmbeddingGateway(),
            provider=provider,
            provider_name=provider_name,
            model_name=model_name,
            dimension=dimension,
            batch_size=batch_size,
            limit=args.limit,
            only_source_key=args.only_source_key,
            only_document_key=args.only_document_key,
            fail_fast=args.fail_fast,
        )
    finally:
        await Tortoise.close_connections()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run embedding generation for active RAG chunks.")
    parser.add_argument("--json", action="store_true", help="Print summary as JSON.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum chunk count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Embedding batch size.")
    parser.add_argument("--provider", choices=["mock", "openai", "disabled"], default=None, help="Provider override.")
    parser.add_argument("--dimension", type=int, default=None, help="Embedding dimension override.")
    parser.add_argument("--model", default=None, help="Embedding model override.")
    parser.add_argument("--only-document-key", default=None, help="Only process chunks for this document_key.")
    parser.add_argument("--only-source-key", default=None, help="Only process chunks with this metadata.source_key.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first embedding failure.")
    args = parser.parse_args()

    summary = asyncio.run(_run_cli(args))
    _print_summary(summary, as_json=args.json)


if __name__ == "__main__":
    main()
