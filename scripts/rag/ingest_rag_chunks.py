from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tortoise import Tortoise  # noqa: E402
from tortoise.transactions import in_transaction  # noqa: E402

from ai_runtime.llm.rag.chunker import RagChunkDraft, build_rag_chunks_from_index  # noqa: E402
from ai_runtime.llm.rag.source_loader import (  # noqa: E402
    DEFAULT_RAG_SOURCE_DIR,
    INDEX_FILE_NAME,
    load_rag_source_index,
)
from ai_runtime.llm.rag.source_trust import source_trust_level_for_type  # noqa: E402
from app.core.db.databases import TORTOISE_ORM  # noqa: E402
from app.models.rag import RAGChunk, RAGDiseaseType, RAGDocument, RAGSource  # noqa: E402

DEFAULT_NAMESPACE = "docs/rag_sources"

LEGACY_DISEASE_TYPE_BY_CODE: dict[str, RAGDiseaseType] = {
    "DM": RAGDiseaseType.DIABETES,
    "OBE": RAGDiseaseType.OBESITY,
    "DL": RAGDiseaseType.DYSLIPIDEMIA,
    "HTN": RAGDiseaseType.COMMON,
    "DIET_NUTRITION": RAGDiseaseType.COMMON,
    "DIET_CAUTION": RAGDiseaseType.COMMON,
    "CKD": RAGDiseaseType.COMMON,
    "ANEM": RAGDiseaseType.COMMON,
    "FL": RAGDiseaseType.COMMON,
    "DIET_FAQ": RAGDiseaseType.COMMON,
}


@dataclass(frozen=True)
class ExcludedRagSource:
    source_key: str
    disease_code: str
    reason: str


@dataclass(frozen=True)
class RagDocumentIngestItem:
    source_key: str
    document_key: str
    source_payload: dict[str, Any]
    document_payload: dict[str, Any]
    chunk_payloads: list[dict[str, Any]]


@dataclass
class RagIngestSummary:
    mode: str
    ingest_namespace: str
    enabled_documents: int = 0
    disabled_documents: int = 0
    total_chunks: int = 0
    planned_source_creates: int = 0
    planned_source_updates: int = 0
    planned_document_creates: int = 0
    planned_document_updates: int = 0
    planned_chunk_creates: int = 0
    planned_chunk_updates: int = 0
    planned_chunk_unchanged: int = 0
    planned_document_deactivations: int = 0
    planned_chunk_deactivations: int = 0
    excluded_sources: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RagIngestGateway(Protocol):
    async def get_source_by_key(self, source_key: str) -> Any | None: ...

    async def create_source(self, payload: dict[str, Any]) -> Any: ...

    async def update_source(self, source: Any, payload: dict[str, Any]) -> None: ...

    async def get_document_by_key(self, document_key: str) -> Any | None: ...

    async def create_document(self, source_id: int, payload: dict[str, Any]) -> Any: ...

    async def update_document(self, document: Any, payload: dict[str, Any]) -> None: ...

    async def get_chunk_by_key(self, chunk_key: str) -> Any | None: ...

    async def create_chunk(self, document_id: int, payload: dict[str, Any]) -> Any: ...

    async def update_chunk(self, chunk: Any, payload: dict[str, Any]) -> None: ...

    async def list_managed_documents(self, namespace: str, source_keys: set[str]) -> list[Any]: ...

    async def list_managed_chunks(self, namespace: str, source_keys: set[str]) -> list[Any]: ...

    async def deactivate_document(self, document: Any) -> None: ...

    async def deactivate_chunks_for_document(self, document_id: int) -> int: ...

    async def deactivate_chunk(self, chunk: Any) -> None: ...


class OrmRagIngestGateway:
    async def get_source_by_key(self, source_key: str) -> RAGSource | None:
        return await RAGSource.get_or_none(source_key=source_key)

    async def create_source(self, payload: dict[str, Any]) -> RAGSource:
        return await RAGSource.create(**payload)

    async def update_source(self, source: RAGSource, payload: dict[str, Any]) -> None:
        _assign_model_fields(source, payload)
        await source.save()

    async def get_document_by_key(self, document_key: str) -> RAGDocument | None:
        return await RAGDocument.get_or_none(document_key=document_key)

    async def create_document(self, source_id: int, payload: dict[str, Any]) -> RAGDocument:
        return await RAGDocument.create(source_id=source_id, **payload)

    async def update_document(self, document: RAGDocument, payload: dict[str, Any]) -> None:
        _assign_model_fields(document, payload)
        await document.save()

    async def get_chunk_by_key(self, chunk_key: str) -> RAGChunk | None:
        return await RAGChunk.get_or_none(chunk_key=chunk_key)

    async def create_chunk(self, document_id: int, payload: dict[str, Any]) -> RAGChunk:
        return await RAGChunk.create(document_id=document_id, **payload)

    async def update_chunk(self, chunk: RAGChunk, payload: dict[str, Any]) -> None:
        _assign_model_fields(chunk, payload)
        await chunk.save()

    async def list_managed_documents(self, namespace: str, source_keys: set[str]) -> list[RAGDocument]:
        documents = await RAGDocument.filter(is_active=True)
        return [
            document
            for document in documents
            if _metadata_matches_namespace(document.metadata, namespace)
            and _metadata_source_key(document.metadata, document.source_key) in source_keys
        ]

    async def list_managed_chunks(self, namespace: str, source_keys: set[str]) -> list[RAGChunk]:
        chunks = await RAGChunk.filter(is_active=True)
        return [
            chunk
            for chunk in chunks
            if _metadata_matches_namespace(chunk.metadata, namespace)
            and _metadata_source_key(chunk.metadata, None) in source_keys
        ]

    async def deactivate_document(self, document: RAGDocument) -> None:
        document.is_active = False
        await document.save()

    async def deactivate_chunks_for_document(self, document_id: int) -> int:
        return await RAGChunk.filter(document_id=document_id, is_active=True).update(is_active=False)

    async def deactivate_chunk(self, chunk: RAGChunk) -> None:
        chunk.is_active = False
        await chunk.save()


@asynccontextmanager
async def _null_transaction() -> Any:
    yield None


def load_rag_ingest_items(
    *,
    index_path: Path | None = None,
    source_dir: Path = DEFAULT_RAG_SOURCE_DIR,
    namespace: str = DEFAULT_NAMESPACE,
    limit: int | None = None,
    only_source_key: str | None = None,
) -> tuple[list[RagDocumentIngestItem], list[ExcludedRagSource], set[str]]:
    index_path = index_path or source_dir / INDEX_FILE_NAME
    source_metadata = load_rag_source_index(source_dir)
    source_file_exists = {metadata.id: (source_dir / metadata.filename).exists() for metadata in source_metadata}
    target_source_keys = {
        metadata.id for metadata in source_metadata if only_source_key is None or metadata.id == only_source_key
    }
    excluded_sources: list[ExcludedRagSource] = []
    for metadata in source_metadata:
        if only_source_key is not None and metadata.id != only_source_key:
            excluded_sources.append(
                ExcludedRagSource(metadata.id, metadata.disease_code, "filtered_by_only_source_key")
            )
            continue
        if not metadata.enabled:
            excluded_sources.append(ExcludedRagSource(metadata.id, metadata.disease_code, "disabled"))
            continue
        if metadata.review_status == "missing_source":
            excluded_sources.append(ExcludedRagSource(metadata.id, metadata.disease_code, "missing_source"))
            continue
        if not source_file_exists[metadata.id]:
            excluded_sources.append(ExcludedRagSource(metadata.id, metadata.disease_code, "missing_file"))

    chunks = build_rag_chunks_from_index(index_path=index_path, source_dir=source_dir)
    if only_source_key is not None:
        chunks = [chunk for chunk in chunks if chunk.source_key == only_source_key]
    items = _build_ingest_items(chunks, namespace=namespace)
    if limit is not None:
        limited_items = items[:limit]
        for item in items[limit:]:
            excluded_sources.append(
                ExcludedRagSource(item.source_key, item.document_payload["disease_code"], "limited")
            )
        items = limited_items
        if limit < len(target_source_keys):
            target_source_keys = {item.source_key for item in items}
    return items, excluded_sources, target_source_keys


async def ingest_rag_chunks(
    items: list[RagDocumentIngestItem],
    *,
    gateway: RagIngestGateway,
    namespace: str = DEFAULT_NAMESPACE,
    apply: bool = False,
    deactivate_missing: bool = False,
    fail_fast: bool = False,
    excluded_sources: list[ExcludedRagSource] | None = None,
    target_source_keys: set[str] | None = None,
    transaction_factory: Callable[[], Any] = _null_transaction,
) -> RagIngestSummary:
    summary = RagIngestSummary(
        mode="apply" if apply else "dry-run",
        ingest_namespace=namespace,
        enabled_documents=len(items),
        disabled_documents=len(excluded_sources or []),
        total_chunks=sum(len(item.chunk_payloads) for item in items),
        excluded_sources=[asdict(source) for source in excluded_sources or []],
    )
    target_source_keys = target_source_keys or {item.source_key for item in items}

    for item in items:
        try:
            async with transaction_factory() if apply else _null_transaction():
                await _plan_or_apply_document(item, gateway=gateway, summary=summary, apply=apply)
        except Exception as exc:  # noqa: BLE001 - ingest summary should record per-document failures.
            message = f"{item.source_key}: {type(exc).__name__}: {exc}"
            summary.warnings.append(message)
            if fail_fast:
                raise

    if deactivate_missing:
        try:
            await _plan_or_apply_deactivations(
                items,
                gateway=gateway,
                summary=summary,
                namespace=namespace,
                target_source_keys=target_source_keys,
                apply=apply,
            )
        except Exception as exc:  # noqa: BLE001
            summary.warnings.append(f"deactivate_missing: {type(exc).__name__}: {exc}")
            if fail_fast:
                raise

    return summary


async def _plan_or_apply_document(
    item: RagDocumentIngestItem,
    *,
    gateway: RagIngestGateway,
    summary: RagIngestSummary,
    apply: bool,
) -> None:
    source = await gateway.get_source_by_key(item.source_key)
    if source is None:
        summary.planned_source_creates += 1
        if apply:
            source = await gateway.create_source(item.source_payload)
    elif _payload_changed(source, item.source_payload):
        summary.planned_source_updates += 1
        if apply:
            await gateway.update_source(source, item.source_payload)

    source_id = getattr(source, "id", 0) if source is not None else 0
    document = await gateway.get_document_by_key(item.document_key)
    if document is None:
        summary.planned_document_creates += 1
        if apply:
            document = await gateway.create_document(source_id, item.document_payload)
    elif _payload_changed(document, item.document_payload):
        summary.planned_document_updates += 1
        if apply:
            await gateway.update_document(document, item.document_payload)

    document_id = getattr(document, "id", 0) if document is not None else 0
    for payload in item.chunk_payloads:
        chunk = await gateway.get_chunk_by_key(payload["chunk_key"])
        if chunk is None:
            summary.planned_chunk_creates += 1
            if apply:
                await gateway.create_chunk(document_id, payload)
            continue
        if not _payload_changed(chunk, payload):
            summary.planned_chunk_unchanged += 1
            continue
        summary.planned_chunk_updates += 1
        if apply:
            await gateway.update_chunk(chunk, payload)


async def _plan_or_apply_deactivations(
    items: list[RagDocumentIngestItem],
    *,
    gateway: RagIngestGateway,
    summary: RagIngestSummary,
    namespace: str,
    target_source_keys: set[str],
    apply: bool,
) -> None:
    manifest_document_keys = {item.document_key for item in items}
    manifest_chunk_keys = {payload["chunk_key"] for item in items for payload in item.chunk_payloads}

    managed_documents = await gateway.list_managed_documents(namespace, target_source_keys)
    for document in managed_documents:
        if getattr(document, "document_key", None) not in manifest_document_keys:
            summary.planned_document_deactivations += 1
            if apply:
                await gateway.deactivate_document(document)
                summary.planned_chunk_deactivations += await gateway.deactivate_chunks_for_document(document.id)

    managed_chunks = await gateway.list_managed_chunks(namespace, target_source_keys)
    for chunk in managed_chunks:
        if getattr(chunk, "chunk_key", None) not in manifest_chunk_keys and getattr(chunk, "is_active", True):
            summary.planned_chunk_deactivations += 1
            if apply:
                await gateway.deactivate_chunk(chunk)


def _build_ingest_items(chunks: list[RagChunkDraft], *, namespace: str) -> list[RagDocumentIngestItem]:
    grouped: dict[str, list[RagChunkDraft]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.source_key, []).append(chunk)

    items: list[RagDocumentIngestItem] = []
    for source_key in sorted(grouped):
        source_chunks = sorted(grouped[source_key], key=lambda chunk: chunk.chunk_index)
        first = source_chunks[0]
        document_key = _document_key(first.source_key, first.filename)
        source_payload = _source_payload(first, namespace=namespace)
        document_payload = _document_payload(first, document_key=document_key, namespace=namespace)
        chunk_payloads = [_chunk_payload(chunk, namespace=namespace) for chunk in source_chunks]
        items.append(
            RagDocumentIngestItem(
                source_key=source_key,
                document_key=document_key,
                source_payload=source_payload,
                document_payload=document_payload,
                chunk_payloads=chunk_payloads,
            )
        )
    return items


def _source_payload(chunk: RagChunkDraft, *, namespace: str) -> dict[str, Any]:
    return {
        "source_key": chunk.source_key,
        "name": chunk.title,
        "organization": chunk.source_org,
        "source_type": chunk.source_type,
        "base_url": chunk.source_url,
        "description": chunk.usage_scope or chunk.notes,
        "is_active": True,
        "metadata": {
            "ingest_namespace": namespace,
            "topic_tags": list(chunk.topic_tags),
            "issue_keys": list(chunk.issue_keys),
            "review_status": chunk.review_status,
            "safety_level": chunk.safety_level,
            "source_trust_level": _source_trust_level(chunk),
            "year": chunk.year,
            "filename": chunk.filename,
            "raw_source_metadata": _raw_source_metadata(chunk),
        },
    }


def _document_payload(chunk: RagChunkDraft, *, document_key: str, namespace: str) -> dict[str, Any]:
    return {
        "document_key": document_key,
        "source_key": chunk.source_key,
        "title": chunk.title,
        "disease_type": _legacy_disease_type(chunk.disease_code),
        "disease_code": chunk.disease_code,
        "filename": chunk.filename,
        "document_url": chunk.source_url,
        "review_status": chunk.review_status,
        "usage_scope": chunk.usage_scope,
        "is_active": True,
        "metadata": {
            "ingest_namespace": namespace,
            "topic_tags": list(chunk.topic_tags),
            "issue_keys": list(chunk.issue_keys),
            "notes": chunk.notes,
            "source_org": chunk.source_org,
            "source_type": chunk.source_type,
            "source_trust_level": _source_trust_level(chunk),
            "safety_level": chunk.safety_level,
            "year": chunk.year,
            "raw_source_metadata": _raw_source_metadata(chunk),
        },
    }


def _chunk_payload(chunk: RagChunkDraft, *, namespace: str) -> dict[str, Any]:
    return {
        "chunk_key": chunk.chunk_key,
        "chunk_index": chunk.chunk_index,
        "section_title": chunk.section_title,
        "content": chunk.content,
        "content_hash": chunk.content_hash,
        "content_length": chunk.content_length,
        "token_estimate": chunk.token_estimate,
        "disease_type": _legacy_disease_type(chunk.disease_code),
        "keywords": ", ".join(dict.fromkeys((*chunk.topic_tags, *chunk.issue_keys))),
        "embedding_model": None,
        "is_active": True,
        "metadata": {
            "ingest_namespace": namespace,
            "heading_path": list(chunk.heading_path),
            "section_index": chunk.section_index,
            "source_key": chunk.source_key,
            "disease_code": chunk.disease_code,
            "review_status": chunk.review_status,
            "usage_scope": chunk.usage_scope,
            "source_org": chunk.source_org,
            "source_url": chunk.source_url,
            "source_type": chunk.source_type,
            "source_trust_level": _source_trust_level(chunk),
            "topic_tags": list(chunk.topic_tags),
            "issue_keys": list(chunk.issue_keys),
            "safety_level": chunk.safety_level,
            "raw_chunk_metadata": {
                "source_id": chunk.source_id,
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "source_type": chunk.source_type,
                "source_trust_level": _source_trust_level(chunk),
                "year": chunk.year,
                "enabled": chunk.enabled,
            },
        },
    }


def _raw_source_metadata(chunk: RagChunkDraft) -> dict[str, Any]:
    return {
        "id": chunk.source_id,
        "disease_code": chunk.disease_code,
        "title": chunk.title,
        "filename": chunk.filename,
        "source_org": chunk.source_org,
        "source_url": chunk.source_url,
        "source_type": chunk.source_type,
        "source_trust_level": _source_trust_level(chunk),
        "topic_tags": list(chunk.topic_tags),
        "issue_keys": list(chunk.issue_keys),
        "usage_scope": chunk.usage_scope,
        "review_status": chunk.review_status,
        "enabled": chunk.enabled,
        "notes": chunk.notes,
        "safety_level": chunk.safety_level,
        "year": chunk.year,
    }


def _source_trust_level(chunk: RagChunkDraft) -> str:
    return chunk.source_trust_level or source_trust_level_for_type(chunk.source_type)


def _legacy_disease_type(disease_code: str) -> RAGDiseaseType:
    return LEGACY_DISEASE_TYPE_BY_CODE.get(disease_code, RAGDiseaseType.COMMON)


def _document_key(source_key: str, filename: str) -> str:
    return f"rag:{source_key}:{filename}"


def _payload_changed(model: Any, payload: dict[str, Any]) -> bool:
    return any(getattr(model, key, None) != value for key, value in payload.items())


def _assign_model_fields(model: Any, payload: dict[str, Any]) -> None:
    for key, value in payload.items():
        setattr(model, key, value)


def _metadata_matches_namespace(metadata: Any, namespace: str) -> bool:
    return isinstance(metadata, dict) and metadata.get("ingest_namespace") == namespace


def _metadata_source_key(metadata: Any, fallback: str | None) -> str | None:
    if isinstance(metadata, dict):
        return metadata.get("source_key") or metadata.get("raw_source_metadata", {}).get("id") or fallback
    return fallback


def _print_summary(summary: RagIngestSummary, *, as_json: bool) -> None:
    payload = summary.to_dict()
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return
    print("RAG chunk ingest")
    print(f"- mode: {summary.mode}")
    print(f"- namespace: {summary.ingest_namespace}")
    print(f"- enabled_documents: {summary.enabled_documents}")
    print(f"- disabled_documents: {summary.disabled_documents}")
    print(f"- total_chunks: {summary.total_chunks}")
    print(f"- planned_source_creates: {summary.planned_source_creates}")
    print(f"- planned_source_updates: {summary.planned_source_updates}")
    print(f"- planned_document_creates: {summary.planned_document_creates}")
    print(f"- planned_document_updates: {summary.planned_document_updates}")
    print(f"- planned_chunk_creates: {summary.planned_chunk_creates}")
    print(f"- planned_chunk_updates: {summary.planned_chunk_updates}")
    print(f"- planned_chunk_unchanged: {summary.planned_chunk_unchanged}")
    print(f"- planned_document_deactivations: {summary.planned_document_deactivations}")
    print(f"- planned_chunk_deactivations: {summary.planned_chunk_deactivations}")
    if summary.excluded_sources:
        print("- excluded_sources:")
        for source in summary.excluded_sources:
            print(f"  - {source['source_key']} ({source['disease_code']}): {source['reason']}")
    if summary.warnings:
        print("- warnings:")
        for warning in summary.warnings:
            print(f"  - {warning}")


async def _run_cli(args: argparse.Namespace) -> RagIngestSummary:
    items, excluded_sources, target_source_keys = load_rag_ingest_items(
        index_path=args.index_path,
        source_dir=args.source_dir,
        namespace=args.namespace,
        limit=args.limit,
        only_source_key=args.only_source_key,
    )
    await Tortoise.init(config=TORTOISE_ORM)
    try:
        return await ingest_rag_chunks(
            items,
            gateway=OrmRagIngestGateway(),
            namespace=args.namespace,
            apply=args.apply,
            deactivate_missing=args.deactivate_missing,
            fail_fast=args.fail_fast,
            excluded_sources=excluded_sources,
            target_source_keys=target_source_keys,
            transaction_factory=in_transaction,
        )
    finally:
        await Tortoise.close_connections()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run or apply RAG markdown chunk ingest.")
    parser.add_argument("--index-path", type=Path, default=None, help="RAG source index path.")
    parser.add_argument(
        "--source-dir", type=Path, default=DEFAULT_RAG_SOURCE_DIR, help="RAG markdown source directory."
    )
    parser.add_argument("--apply", action="store_true", help="Apply changes to DB. Default is dry-run only.")
    parser.add_argument("--json", action="store_true", help="Print summary as JSON.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of source documents to process.")
    parser.add_argument("--only-source-key", default=None, help="Only ingest one source_key.")
    parser.add_argument(
        "--deactivate-missing",
        action="store_true",
        help="Deactivate managed documents/chunks missing from the current manifest.",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first document failure.")
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE, help="Managed ingest namespace.")
    args = parser.parse_args()

    summary = asyncio.run(_run_cli(args))
    _print_summary(summary, as_json=args.json)


if __name__ == "__main__":
    main()
