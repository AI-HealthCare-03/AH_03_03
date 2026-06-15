from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAG_SOURCE_DIR = REPO_ROOT / "docs" / "rag_sources"
INDEX_FILE_NAME = "index.json"


@dataclass(frozen=True)
class RagSourceMetadata:
    id: str
    disease_code: str
    title: str
    filename: str
    source_org: str
    source_url: str
    year: int | None
    source_type: str
    topic_tags: tuple[str, ...]
    issue_keys: tuple[str, ...]
    usage_scope: str
    review_status: str
    enabled: bool
    notes: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RagSourceMetadata:
        disease_code = str(payload.get("disease_code") or payload.get("disease_type") or "")
        review_status = str(payload.get("review_status") or payload.get("status") or "")
        return cls(
            id=str(payload["id"]),
            disease_code=disease_code,
            title=str(payload["title"]),
            filename=str(payload.get("filename") or f"{payload['id']}.md"),
            source_org=str(payload["source_org"]),
            source_url=str(payload["source_url"]),
            year=payload.get("year"),
            source_type=str(payload["source_type"]),
            topic_tags=tuple(str(item) for item in payload.get("topic_tags", [])),
            issue_keys=tuple(str(item) for item in payload.get("issue_keys", [])),
            usage_scope=str(payload.get("usage_scope") or payload.get("runtime_use") or ""),
            review_status=review_status,
            enabled=bool(payload.get("enabled", True)),
            notes=payload.get("notes"),
        )

    @property
    def disease_type(self) -> str:
        return self.disease_code

    @property
    def runtime_use(self) -> str:
        return self.usage_scope

    @property
    def status(self) -> str:
        return self.review_status


@dataclass(frozen=True)
class RagSourceDocument:
    metadata: RagSourceMetadata
    content: str
    path: Path

    @property
    def id(self) -> str:
        return self.metadata.id

    @property
    def title(self) -> str:
        return self.metadata.title

    @property
    def status(self) -> str:
        return self.metadata.status


def load_rag_source_index(source_dir: Path = DEFAULT_RAG_SOURCE_DIR) -> list[RagSourceMetadata]:
    index_path = source_dir / INDEX_FILE_NAME
    with index_path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"RAG source index must be a list: {index_path}")
    return [RagSourceMetadata.from_dict(item) for item in payload]


def load_rag_source_document(
    source_id: str,
    source_dir: Path = DEFAULT_RAG_SOURCE_DIR,
    index: list[RagSourceMetadata] | None = None,
) -> RagSourceDocument:
    metadata_by_id = {metadata.id: metadata for metadata in (index or load_rag_source_index(source_dir))}
    metadata = metadata_by_id.get(source_id)
    if metadata is None:
        raise KeyError(f"Unknown RAG source id: {source_id}")

    source_path = source_dir / metadata.filename
    content = source_path.read_text(encoding="utf-8")
    return RagSourceDocument(metadata=metadata, content=content, path=source_path)


def load_all_rag_source_documents(
    source_dir: Path = DEFAULT_RAG_SOURCE_DIR,
    *,
    enabled_only: bool = True,
) -> list[RagSourceDocument]:
    index = load_rag_source_index(source_dir)
    return [
        load_rag_source_document(metadata.id, source_dir=source_dir, index=index)
        for metadata in index
        if not enabled_only or metadata.enabled
    ]
