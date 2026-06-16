from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ai_runtime.llm.rag.source_loader import (
    DEFAULT_RAG_SOURCE_DIR,
    INDEX_FILE_NAME,
    RagSourceDocument,
    RagSourceMetadata,
    load_rag_source_document,
    load_rag_source_index,
)

DEFAULT_CHUNK_MAX_CHARS = 1000
DEFAULT_CHUNK_OVERLAP_CHARS = 150
MIN_CHUNK_CHARS = 40
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class RagChunkDraft:
    source_id: str
    document_id: str
    source_key: str
    disease_code: str
    title: str
    source_org: str
    source_url: str
    review_status: str
    usage_scope: str
    topic_tags: tuple[str, ...]
    issue_keys: tuple[str, ...]
    safety_level: str
    section_title: str | None
    chunk_index: int
    chunk_key: str
    content_hash: str
    content: str
    content_length: int
    filename: str
    source_type: str
    source_trust_level: str
    year: int | None
    enabled: bool
    notes: str | None
    heading_path: tuple[str, ...]
    section_index: int
    token_estimate: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["topic_tags"] = list(self.topic_tags)
        payload["issue_keys"] = list(self.issue_keys)
        payload["heading_path"] = list(self.heading_path)
        return payload


@dataclass(frozen=True)
class RagChunkDryRunSummary:
    total_chunks: int
    enabled_documents: int
    disabled_documents: int
    chunks_by_disease_code: dict[str, int]
    sources: list[dict[str, Any]]
    chunk_fields: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def source_count(self) -> int:
        return self.enabled_documents

    @property
    def chunk_count(self) -> int:
        return self.total_chunks


def build_rag_chunks_from_index(
    index_path: Path | str | None = None,
    source_dir: Path | str = DEFAULT_RAG_SOURCE_DIR,
    *,
    max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> list[RagChunkDraft]:
    """Build chunk drafts before pgvector ingest; no DB writes or embeddings."""
    source_dir = Path(source_dir)
    index_path = Path(index_path) if index_path is not None else source_dir / INDEX_FILE_NAME
    documents = _load_enabled_chunkable_documents(index_path=index_path, source_dir=source_dir)
    chunks: list[RagChunkDraft] = []
    for document in documents:
        chunks.extend(_chunk_document(document, start_index=0, max_chars=max_chars, overlap_chars=overlap_chars))
    return chunks


def summarize_rag_chunks(
    chunks: list[RagChunkDraft],
    *,
    index_path: Path | str | None = None,
    source_dir: Path | str = DEFAULT_RAG_SOURCE_DIR,
) -> RagChunkDryRunSummary:
    source_dir = Path(source_dir)
    index_path = Path(index_path) if index_path is not None else source_dir / INDEX_FILE_NAME
    index = _load_index_from_path(index_path)
    enabled_documents = sum(
        1
        for metadata in index
        if metadata.enabled and metadata.review_status != "missing_source" and (source_dir / metadata.filename).exists()
    )
    disabled_documents = len(index) - enabled_documents
    grouped: dict[str, list[RagChunkDraft]] = {}
    chunks_by_disease_code: dict[str, int] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.source_id, []).append(chunk)
        chunks_by_disease_code[chunk.disease_code] = chunks_by_disease_code.get(chunk.disease_code, 0) + 1

    sources = []
    for source_id, source_chunks in sorted(grouped.items()):
        content_lengths = [chunk.content_length for chunk in source_chunks]
        sources.append(
            {
                "source_id": source_id,
                "document_id": source_chunks[0].document_id,
                "source_key": source_chunks[0].source_key,
                "disease_code": source_chunks[0].disease_code,
                "title": source_chunks[0].title,
                "review_status": source_chunks[0].review_status,
                "chunk_count": len(source_chunks),
                "min_content_length": min(content_lengths),
                "max_content_length": max(content_lengths),
                "chunk_keys": [chunk.chunk_key for chunk in source_chunks],
            }
        )

    return RagChunkDryRunSummary(
        total_chunks=len(chunks),
        enabled_documents=enabled_documents,
        disabled_documents=disabled_documents,
        chunks_by_disease_code=dict(sorted(chunks_by_disease_code.items())),
        sources=sources,
        chunk_fields=list(RagChunkDraft.__dataclass_fields__.keys()),
    )


def build_rag_chunk_drafts(
    source_dir: Path = DEFAULT_RAG_SOURCE_DIR,
    *,
    enabled_only: bool = True,
    max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> list[RagChunkDraft]:
    if enabled_only:
        return build_rag_chunks_from_index(source_dir=source_dir, max_chars=max_chars, overlap_chars=overlap_chars)
    documents = _load_existing_documents(source_dir)
    chunks: list[RagChunkDraft] = []
    for document in documents:
        chunks.extend(_chunk_document(document, start_index=0, max_chars=max_chars, overlap_chars=overlap_chars))
    return chunks


def build_rag_chunk_dry_run_summary(chunks: list[RagChunkDraft]) -> RagChunkDryRunSummary:
    return summarize_rag_chunks(chunks)


def _load_enabled_chunkable_documents(*, index_path: Path, source_dir: Path) -> list[RagSourceDocument]:
    index = _load_index_from_path(index_path)
    documents: list[RagSourceDocument] = []
    for metadata in index:
        if not metadata.enabled or metadata.review_status == "missing_source":
            continue
        if not (source_dir / metadata.filename).exists():
            continue
        documents.append(load_rag_source_document(metadata.id, source_dir=source_dir, index=index))
    return documents


def _load_existing_documents(source_dir: Path) -> list[RagSourceDocument]:
    index = load_rag_source_index(source_dir)
    return [
        load_rag_source_document(metadata.id, source_dir=source_dir, index=index)
        for metadata in index
        if (source_dir / metadata.filename).exists()
    ]


def _load_index_from_path(index_path: Path) -> list[RagSourceMetadata]:
    with index_path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"RAG source index must be a list: {index_path}")
    return [RagSourceMetadata.from_dict(item) for item in payload]


def _chunk_document(
    document: RagSourceDocument,
    *,
    start_index: int,
    max_chars: int,
    overlap_chars: int,
) -> list[RagChunkDraft]:
    sections = _markdown_sections(document.content)
    chunks: list[RagChunkDraft] = []
    chunk_index = start_index
    for section_index, section in enumerate(sections):
        for content in _split_long_content(section["content"], max_chars=max_chars, overlap_chars=overlap_chars):
            if len(content) < MIN_CHUNK_CHARS:
                continue
            source_key = document.id
            chunks.append(
                RagChunkDraft(
                    source_id=document.id,
                    document_id=document.id,
                    source_key=source_key,
                    disease_code=document.metadata.disease_code,
                    title=document.metadata.title,
                    source_org=document.metadata.source_org,
                    source_url=document.metadata.source_url,
                    review_status=document.metadata.review_status,
                    usage_scope=document.metadata.usage_scope,
                    topic_tags=document.metadata.topic_tags,
                    issue_keys=document.metadata.issue_keys,
                    safety_level=document.metadata.safety_level or "normal",
                    section_title=section["section_title"],
                    chunk_index=chunk_index,
                    chunk_key=_chunk_key(source_key, section_index, chunk_index),
                    content_hash=_content_hash(content),
                    content=content,
                    content_length=len(content),
                    filename=document.metadata.filename,
                    source_type=document.metadata.source_type,
                    source_trust_level=document.metadata.source_trust_level or "unknown",
                    year=document.metadata.year,
                    enabled=document.metadata.enabled,
                    notes=document.metadata.notes,
                    heading_path=tuple(section["heading_path"]),
                    section_index=section_index,
                    token_estimate=max(1, len(content) // 3),
                )
            )
            chunk_index += 1
    return chunks


def _markdown_sections(content: str) -> list[dict[str, Any]]:
    lines = content.splitlines()
    headings: list[tuple[int, str]] = []
    sections: list[dict[str, Any]] = []
    current_lines: list[str] = []
    current_title: str | None = None
    current_path: list[str] = []

    def flush() -> None:
        normalized = _normalize_content("\n".join(current_lines))
        if normalized:
            sections.append(
                {
                    "section_title": current_title,
                    "heading_path": list(current_path),
                    "content": normalized,
                }
            )

    for line in lines:
        match = HEADING_PATTERN.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            headings[:] = [
                (heading_level, heading_title) for heading_level, heading_title in headings if heading_level < level
            ]
            headings.append((level, title))
            current_title = title
            current_path = [heading_title for _, heading_title in headings]
            current_lines = [line]
            continue
        current_lines.append(line)
    flush()
    return sections


def _split_long_content(content: str, *, max_chars: int, overlap_chars: int) -> list[str]:
    if len(content) <= max_chars:
        return [content]
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", content) if paragraph.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = paragraph
        while len(current) > max_chars:
            chunks.append(current[:max_chars].strip())
            start = max(0, max_chars - overlap_chars)
            current = current[start:].strip()
    if current:
        chunks.append(current)
    return [_normalize_content(chunk) for chunk in chunks if _normalize_content(chunk)]


def _normalize_content(content: str) -> str:
    lines = [line.rstrip() for line in content.strip().splitlines()]
    return "\n".join(lines).strip()


def _content_hash(content: str) -> str:
    return hashlib.sha256(_normalize_content(content).encode("utf-8")).hexdigest()


def _chunk_key(source_key: str, section_index: int, chunk_index: int) -> str:
    return f"rag:{source_key}:section:{section_index:03d}:chunk:{chunk_index:04d}"
