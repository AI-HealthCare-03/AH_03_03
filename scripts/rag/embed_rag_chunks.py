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
from tortoise.transactions import in_transaction  # noqa: E402

from ai_runtime.llm.rag.embeddings import EmbeddingProvider, get_embedding_provider  # noqa: E402
from app.core import config  # noqa: E402
from app.core.db.databases import TORTOISE_ORM  # noqa: E402


@dataclass(frozen=True)
class RagChunkEmbeddingCandidate:
    chunk_key: str
    content: str
    content_hash: str | None = None
    embedding_is_null: bool = True
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    embedding_content_hash: str | None = None
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
    planned_embedding_writes: int = 0
    embedding_writes: int = 0
    skipped_existing_embeddings: int = 0
    stale_embeddings: int = 0
    forced_embeddings: int = 0
    failed_embedding_writes: int = 0
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


class EmbeddingWriteGateway(Protocol):
    async def write_embedding(
        self,
        *,
        chunk_key: str,
        vector: list[float],
        provider: str,
        model: str,
        dimension: int,
    ) -> int: ...


class OrmRagChunkEmbeddingGateway:
    async def list_active_chunks(
        self,
        *,
        limit: int | None = None,
        only_source_key: str | None = None,
        only_document_key: str | None = None,
    ) -> list[RagChunkEmbeddingCandidate]:
        query = """
            SELECT
                c.chunk_key,
                c.content,
                c.content_hash,
                c.embedding_provider,
                c.embedding_model,
                c.embedding_dimension,
                c.embedding_content_hash,
                c.embedding IS NULL AS embedding_is_null,
                c.metadata,
                d.document_key
            FROM rag_chunks c
            JOIN rag_documents d ON d.id = c.document_id
            WHERE c.is_active = TRUE
              AND d.is_active = TRUE
              AND COALESCE(BTRIM(c.content), '') <> ''
            ORDER BY c.document_id, c.chunk_index
        """
        rows = await Tortoise.get_connection("default").execute_query_dict(query)
        candidates: list[RagChunkEmbeddingCandidate] = []
        for row in rows:
            candidate = _candidate_from_row(row)
            if only_source_key is not None and candidate.source_key != only_source_key:
                continue
            if only_document_key is not None and candidate.document_key != only_document_key:
                continue
            candidates.append(candidate)
            if limit is not None and len(candidates) >= limit:
                break
        return candidates


class OrmEmbeddingWriteGateway:
    def __init__(self, connection: Any) -> None:
        self.connection = connection

    async def write_embedding(
        self,
        *,
        chunk_key: str,
        vector: list[float],
        provider: str,
        model: str,
        dimension: int,
    ) -> int:
        vector_payload = _vector_to_pgvector_literal(vector)
        affected_rows, _ = await self.connection.execute_query(
            """
            UPDATE rag_chunks
            SET
                embedding = $1::vector,
                embedding_provider = $2,
                embedding_model = $3,
                embedding_dimension = $4,
                embedding_content_hash = content_hash,
                embedded_at = NOW()
            WHERE chunk_key = $5
            """,
            [vector_payload, provider, model, dimension, chunk_key],
        )
        return int(affected_rows or 0)


async def dry_run_embed_rag_chunks(
    **kwargs: Any,
) -> RagChunkEmbeddingDryRunSummary:
    return await embed_rag_chunks(**kwargs, apply=False)


async def embed_rag_chunks(
    *,
    gateway: RagChunkEmbeddingGateway,
    provider: EmbeddingProvider | None,
    provider_name: str,
    model_name: str | None,
    dimension: int | None,
    batch_size: int,
    writer: EmbeddingWriteGateway | None = None,
    apply: bool = False,
    force: bool = False,
    skip_existing: bool = True,
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
    if apply and provider.provider_name != "mock":
        summary.warnings.append("OpenAI embedding apply is disabled in this stage")
        return summary
    if apply and writer is None:
        summary.warnings.append("embedding apply requested but no writer was provided")
        if fail_fast:
            raise ValueError(summary.warnings[-1])
        return summary

    candidates = await gateway.list_active_chunks(
        limit=limit,
        only_source_key=only_source_key,
        only_document_key=only_document_key,
    )
    summary.total_candidate_chunks = len(candidates)
    summary.sample_chunk_keys = [candidate.chunk_key for candidate in candidates[:5]]
    targets = _select_embedding_targets(
        candidates,
        provider=provider,
        force=force,
        skip_existing=skip_existing,
        summary=summary,
    )
    summary.planned_embedding_writes = len(targets)

    for batch in _batches(targets, normalized_batch_size):
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
            if apply:
                try:
                    written = await writer.write_embedding(
                        chunk_key=candidate.chunk_key,
                        vector=vector,
                        provider=provider.provider_name,
                        model=provider.model_name,
                        dimension=provider.dimension,
                    )
                except Exception as exc:  # noqa: BLE001 - script should summarize per-chunk write failures.
                    summary.failed_embedding_writes += 1
                    summary.warnings.append(
                        f"{candidate.chunk_key}: embedding write failed: {type(exc).__name__}: {exc}"
                    )
                    if fail_fast:
                        raise
                    continue
                if written > 0:
                    summary.embedding_writes += written
                else:
                    summary.failed_embedding_writes += 1
                    summary.warnings.append(f"{candidate.chunk_key}: embedding write affected 0 rows")
                    if fail_fast:
                        raise RuntimeError(summary.warnings[-1])
    summary.skipped_chunks = max(0, summary.total_candidate_chunks - summary.embedded_chunks - summary.failed_chunks)
    summary.db_write_performed = apply and summary.embedding_writes > 0
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


def _candidate_from_row(row: dict[str, Any]) -> RagChunkEmbeddingCandidate:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return RagChunkEmbeddingCandidate(
        chunk_key=str(row.get("chunk_key") or ""),
        content=str(row.get("content") or ""),
        content_hash=row.get("content_hash"),
        embedding_is_null=bool(row.get("embedding_is_null", True)),
        embedding_provider=row.get("embedding_provider"),
        embedding_model=row.get("embedding_model"),
        embedding_dimension=row.get("embedding_dimension"),
        embedding_content_hash=row.get("embedding_content_hash"),
        document_key=row.get("document_key") or metadata.get("document_key"),
        source_key=metadata.get("source_key"),
        metadata=metadata,
    )


def _batches(candidates: list[RagChunkEmbeddingCandidate], batch_size: int) -> list[list[RagChunkEmbeddingCandidate]]:
    return [candidates[index : index + batch_size] for index in range(0, len(candidates), batch_size)]


def _select_embedding_targets(
    candidates: list[RagChunkEmbeddingCandidate],
    *,
    provider: EmbeddingProvider,
    force: bool,
    skip_existing: bool,
    summary: RagChunkEmbeddingDryRunSummary,
) -> list[RagChunkEmbeddingCandidate]:
    targets: list[RagChunkEmbeddingCandidate] = []
    for candidate in candidates:
        if not candidate.chunk_key:
            summary.skipped_chunks += 1
            summary.warnings.append("chunk_key is missing; embedding skipped")
            continue
        if not candidate.content_hash:
            summary.skipped_chunks += 1
            summary.warnings.append(f"{candidate.chunk_key}: content_hash is missing; embedding skipped")
            continue
        current = _embedding_is_current(candidate, provider)
        if force:
            if current:
                summary.forced_embeddings += 1
            targets.append(candidate)
            continue
        if current and skip_existing:
            summary.skipped_existing_embeddings += 1
            summary.skipped_chunks += 1
            continue
        if not candidate.embedding_is_null:
            summary.stale_embeddings += 1
        targets.append(candidate)
    return targets


def _embedding_is_current(candidate: RagChunkEmbeddingCandidate, provider: EmbeddingProvider) -> bool:
    return (
        not candidate.embedding_is_null
        and candidate.embedding_provider == provider.provider_name
        and candidate.embedding_model == provider.model_name
        and candidate.embedding_dimension == provider.dimension
        and candidate.embedding_content_hash == candidate.content_hash
    )


def _vector_to_pgvector_literal(vector: list[float]) -> str:
    return "[" + ",".join(_format_vector_float(value) for value in vector) + "]"


def _format_vector_float(value: float) -> str:
    return format(float(value), ".12g")


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
    print(f"- planned_embedding_writes: {summary.planned_embedding_writes}")
    print(f"- embedding_writes: {summary.embedding_writes}")
    print(f"- skipped_existing_embeddings: {summary.skipped_existing_embeddings}")
    print(f"- stale_embeddings: {summary.stale_embeddings}")
    print(f"- forced_embeddings: {summary.forced_embeddings}")
    print(f"- failed_embedding_writes: {summary.failed_embedding_writes}")
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
    if args.apply and provider_name != "mock":
        return await embed_rag_chunks(
            gateway=OrmRagChunkEmbeddingGateway(),
            provider=provider,
            provider_name=provider_name,
            model_name=model_name,
            dimension=dimension,
            batch_size=batch_size,
            apply=True,
            fail_fast=args.fail_fast,
        )
    if provider is None:
        return await embed_rag_chunks(
            gateway=OrmRagChunkEmbeddingGateway(),
            provider=None,
            provider_name=provider_name,
            model_name=model_name,
            dimension=dimension,
            batch_size=batch_size,
        )

    await Tortoise.init(config=TORTOISE_ORM)
    try:
        if args.apply:
            async with in_transaction() as connection:
                return await embed_rag_chunks(
                    gateway=OrmRagChunkEmbeddingGateway(),
                    provider=provider,
                    provider_name=provider_name,
                    model_name=model_name,
                    dimension=dimension,
                    batch_size=batch_size,
                    writer=OrmEmbeddingWriteGateway(connection),
                    apply=True,
                    force=args.force,
                    skip_existing=args.skip_existing,
                    limit=args.limit,
                    only_source_key=args.only_source_key,
                    only_document_key=args.only_document_key,
                    fail_fast=args.fail_fast,
                )
        return await embed_rag_chunks(
            gateway=OrmRagChunkEmbeddingGateway(),
            provider=provider,
            provider_name=provider_name,
            model_name=model_name,
            dimension=dimension,
            batch_size=batch_size,
            force=args.force,
            skip_existing=args.skip_existing,
            limit=args.limit,
            only_source_key=args.only_source_key,
            only_document_key=args.only_document_key,
            fail_fast=args.fail_fast,
        )
    finally:
        await Tortoise.close_connections()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run embedding generation for active RAG chunks.")
    parser.add_argument("--apply", action="store_true", help="Write mock embeddings to rag_chunks. Default is dry-run.")
    parser.add_argument("--force", action="store_true", help="Regenerate embeddings even if current metadata matches.")
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip chunks whose embedding metadata is current.",
    )
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
