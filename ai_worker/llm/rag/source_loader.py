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
    disease_type: str
    title: str
    source_org: str
    source_url: str
    year: int | None
    source_type: str
    runtime_use: str
    status: str
    notes: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RagSourceMetadata:
        return cls(
            id=str(payload["id"]),
            disease_type=str(payload["disease_type"]),
            title=str(payload["title"]),
            source_org=str(payload["source_org"]),
            source_url=str(payload["source_url"]),
            year=payload.get("year"),
            source_type=str(payload["source_type"]),
            runtime_use=str(payload["runtime_use"]),
            status=str(payload["status"]),
            notes=payload.get("notes"),
        )


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

    source_path = source_dir / f"{source_id}.md"
    content = source_path.read_text(encoding="utf-8")
    return RagSourceDocument(metadata=metadata, content=content, path=source_path)


def load_all_rag_source_documents(source_dir: Path = DEFAULT_RAG_SOURCE_DIR) -> list[RagSourceDocument]:
    index = load_rag_source_index(source_dir)
    return [load_rag_source_document(metadata.id, source_dir=source_dir, index=index) for metadata in index]
