from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ai_runtime.llm.rag.chunker import RagChunkDraft, build_rag_chunk_drafts
from app.models.rag import RAGDiseaseType
from scripts.rag.ingest_rag_chunks import (
    LEGACY_DISEASE_TYPE_BY_CODE,
    RagDocumentIngestItem,
    _build_ingest_items,
    ingest_rag_chunks,
    load_rag_ingest_items,
)


@dataclass
class FakeRow:
    id: int
    is_active: bool = True
    metadata: dict[str, Any] | None = None
    source_key: str | None = None
    document_key: str | None = None
    chunk_key: str | None = None
    content_hash: str | None = None
    document_id: int | None = None
    values: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class FakeRagGateway:
    def __init__(self) -> None:
        self.sources: dict[str, FakeRow] = {}
        self.documents: dict[str, FakeRow] = {}
        self.chunks: dict[str, FakeRow] = {}
        self.next_id = 1

    async def get_source_by_key(self, source_key: str) -> FakeRow | None:
        return self.sources.get(source_key)

    async def create_source(self, payload: dict[str, Any]) -> FakeRow:
        row = self._row(payload, source_key=payload["source_key"])
        self.sources[payload["source_key"]] = row
        return row

    async def update_source(self, source: FakeRow, payload: dict[str, Any]) -> None:
        self._update_row(source, payload)

    async def get_document_by_key(self, document_key: str) -> FakeRow | None:
        return self.documents.get(document_key)

    async def create_document(self, source_id: int, payload: dict[str, Any]) -> FakeRow:
        row = self._row(payload, document_key=payload["document_key"])
        row.values["source_id"] = source_id
        self.documents[payload["document_key"]] = row
        return row

    async def update_document(self, document: FakeRow, payload: dict[str, Any]) -> None:
        self._update_row(document, payload)

    async def get_chunk_by_key(self, chunk_key: str) -> FakeRow | None:
        return self.chunks.get(chunk_key)

    async def create_chunk(self, document_id: int, payload: dict[str, Any]) -> FakeRow:
        row = self._row(payload, chunk_key=payload["chunk_key"], content_hash=payload["content_hash"])
        row.document_id = document_id
        self.chunks[payload["chunk_key"]] = row
        return row

    async def update_chunk(self, chunk: FakeRow, payload: dict[str, Any]) -> None:
        self._update_row(chunk, payload)
        chunk.content_hash = payload["content_hash"]

    async def list_managed_documents(self, namespace: str, source_keys: set[str]) -> list[FakeRow]:
        return [
            document
            for document in self.documents.values()
            if document.is_active
            and document.metadata
            and document.metadata.get("ingest_namespace") == namespace
            and document.source_key in source_keys
        ]

    async def list_managed_chunks(self, namespace: str, source_keys: set[str]) -> list[FakeRow]:
        return [
            chunk
            for chunk in self.chunks.values()
            if chunk.is_active
            and chunk.metadata
            and chunk.metadata.get("ingest_namespace") == namespace
            and chunk.metadata.get("source_key") in source_keys
        ]

    async def deactivate_document(self, document: FakeRow) -> None:
        document.is_active = False

    async def deactivate_chunks_for_document(self, document_id: int) -> int:
        count = 0
        for chunk in self.chunks.values():
            if chunk.document_id == document_id and chunk.is_active:
                chunk.is_active = False
                count += 1
        return count

    async def deactivate_chunk(self, chunk: FakeRow) -> None:
        chunk.is_active = False

    def _row(self, payload: dict[str, Any], **overrides: Any) -> FakeRow:
        row = FakeRow(id=self.next_id, values=dict(payload), **overrides)
        self.next_id += 1
        self._update_row(row, payload)
        return row

    @staticmethod
    def _update_row(row: FakeRow, payload: dict[str, Any]) -> None:
        row.values.update(payload)
        row.metadata = payload.get("metadata")
        row.is_active = payload.get("is_active", row.is_active)
        row.source_key = payload.get("source_key", row.source_key)
        row.document_key = payload.get("document_key", row.document_key)
        row.chunk_key = payload.get("chunk_key", row.chunk_key)
        row.content_hash = payload.get("content_hash", row.content_hash)


@pytest.mark.asyncio
async def test_dry_run_does_not_write_to_gateway() -> None:
    gateway = FakeRagGateway()
    items = [_item("hypertension", "HTN")]

    summary = await ingest_rag_chunks(items, gateway=gateway, apply=False)

    assert summary.mode == "dry-run"
    assert summary.planned_source_creates == 1
    assert summary.planned_document_creates == 1
    assert summary.planned_chunk_creates == 1
    assert gateway.sources == {}
    assert gateway.documents == {}
    assert gateway.chunks == {}


@pytest.mark.asyncio
async def test_apply_creates_rows_and_second_apply_is_idempotent() -> None:
    gateway = FakeRagGateway()
    items = [_item("diabetes", "DM")]

    first = await ingest_rag_chunks(items, gateway=gateway, apply=True)
    second = await ingest_rag_chunks(items, gateway=gateway, apply=True)

    assert first.planned_source_creates == 1
    assert first.planned_document_creates == 1
    assert first.planned_chunk_creates == 1
    assert second.planned_source_creates == 0
    assert second.planned_document_creates == 0
    assert second.planned_chunk_creates == 0
    assert second.planned_chunk_unchanged == 1
    assert len(gateway.sources) == 1
    assert len(gateway.documents) == 1
    assert len(gateway.chunks) == 1


@pytest.mark.asyncio
async def test_apply_updates_chunk_when_content_hash_changes() -> None:
    gateway = FakeRagGateway()
    original = [_item("obesity", "OBE", content="첫 번째 내용입니다.", content_hash="a" * 64)]
    changed = [_item("obesity", "OBE", content="수정된 내용입니다.", content_hash="b" * 64)]

    await ingest_rag_chunks(original, gateway=gateway, apply=True)
    summary = await ingest_rag_chunks(changed, gateway=gateway, apply=True)

    assert summary.planned_chunk_updates == 1
    assert gateway.chunks["rag:obesity:section:000:chunk:0000"].content_hash == "b" * 64
    assert gateway.chunks["rag:obesity:section:000:chunk:0000"].values["content"] == "수정된 내용입니다."


@pytest.mark.asyncio
async def test_dry_run_detects_metadata_only_chunk_update() -> None:
    gateway = FakeRagGateway()
    items = [_item("dyslipidemia", "DL")]

    await ingest_rag_chunks(items, gateway=gateway, apply=True)
    chunk = gateway.chunks["rag:dyslipidemia:section:000:chunk:0000"]
    chunk.metadata = {
        key: value
        for key, value in chunk.metadata.items()
        if key not in {"source_org", "source_url", "source_type", "source_trust_level"}
    }
    chunk.values["metadata"] = chunk.metadata

    summary = await ingest_rag_chunks(items, gateway=gateway, apply=False)

    assert summary.planned_chunk_updates == 1
    assert summary.planned_chunk_unchanged == 0


@pytest.mark.asyncio
async def test_metadata_only_chunk_update_preserves_existing_embedding_payload() -> None:
    gateway = FakeRagGateway()
    items = [_item("dyslipidemia", "DL")]

    await ingest_rag_chunks(items, gateway=gateway, apply=True)
    chunk = gateway.chunks["rag:dyslipidemia:section:000:chunk:0000"]
    chunk.values["embedding"] = "[0.1,0.2]"
    chunk.values["embedding_provider"] = "openai"
    chunk.metadata = {
        key: value
        for key, value in chunk.metadata.items()
        if key not in {"source_org", "source_url", "source_type", "source_trust_level"}
    }
    chunk.values["metadata"] = chunk.metadata

    summary = await ingest_rag_chunks(items, gateway=gateway, apply=True)

    assert summary.planned_chunk_updates == 1
    assert gateway.chunks["rag:dyslipidemia:section:000:chunk:0000"].values["embedding"] == "[0.1,0.2]"
    assert gateway.chunks["rag:dyslipidemia:section:000:chunk:0000"].values["embedding_provider"] == "openai"
    assert gateway.chunks["rag:dyslipidemia:section:000:chunk:0000"].metadata["source_trust_level"] == "unknown"


@pytest.mark.asyncio
async def test_deactivate_missing_only_affects_managed_namespace() -> None:
    gateway = FakeRagGateway()
    first_items = [_item("hypertension", "HTN"), _item("diabetes", "DM")]
    next_items = [_item("hypertension", "HTN")]

    await ingest_rag_chunks(first_items, gateway=gateway, apply=True)
    summary = await ingest_rag_chunks(
        next_items,
        gateway=gateway,
        apply=True,
        deactivate_missing=True,
        target_source_keys={"hypertension", "diabetes"},
    )

    assert summary.planned_document_deactivations == 1
    assert summary.planned_chunk_deactivations == 1
    assert gateway.documents["rag:diabetes:diabetes.md"].is_active is False
    assert gateway.chunks["rag:diabetes:section:000:chunk:0000"].is_active is False
    assert gateway.documents["rag:hypertension:hypertension.md"].is_active is True


def test_disabled_and_missing_sources_are_excluded_from_manifest() -> None:
    items, excluded_sources, _target_source_keys = load_rag_ingest_items()
    source_keys = {item.source_key for item in items}
    excluded_by_key = {source.source_key: source.reason for source in excluded_sources}

    assert {"hypertension", "diabetes", "dyslipidemia", "obesity", "diet_nutrition", "diet_caution"} <= source_keys
    assert "ckd" not in source_keys
    assert "anemia" not in source_keys
    assert "fatty_liver" not in source_keys
    assert "diet_faq" not in source_keys
    assert excluded_by_key["ckd"] == "disabled"
    assert excluded_by_key["anemia"] == "disabled"
    assert excluded_by_key["fatty_liver"] == "disabled"
    assert excluded_by_key["diet_faq"] == "disabled"


def test_legacy_disease_type_mapping_uses_existing_enum_only() -> None:
    assert LEGACY_DISEASE_TYPE_BY_CODE["DM"] == RAGDiseaseType.DIABETES
    assert LEGACY_DISEASE_TYPE_BY_CODE["OBE"] == RAGDiseaseType.OBESITY
    assert LEGACY_DISEASE_TYPE_BY_CODE["DL"] == RAGDiseaseType.DYSLIPIDEMIA
    assert LEGACY_DISEASE_TYPE_BY_CODE["HTN"] == RAGDiseaseType.COMMON
    assert LEGACY_DISEASE_TYPE_BY_CODE["DIET_NUTRITION"] == RAGDiseaseType.COMMON


def test_ingest_payload_preserves_disease_code_and_chunk_metadata() -> None:
    item = _item("diet_nutrition", "DIET_NUTRITION")

    assert item.document_payload["disease_code"] == "DIET_NUTRITION"
    assert item.document_payload["disease_type"] == RAGDiseaseType.COMMON
    assert item.document_payload["document_url"] == "https://example.com/diet_nutrition"
    chunk_payload = item.chunk_payloads[0]
    assert chunk_payload["metadata"]["heading_path"] == ["요약"]
    assert chunk_payload["metadata"]["source_key"] == "diet_nutrition"
    assert chunk_payload["metadata"]["disease_code"] == "DIET_NUTRITION"
    assert chunk_payload["metadata"]["source_org"] == "테스트 기관"
    assert chunk_payload["metadata"]["source_url"] == "https://example.com/diet_nutrition"
    assert chunk_payload["metadata"]["source_type"] == "official"
    assert chunk_payload["metadata"]["source_trust_level"] == "unknown"
    assert chunk_payload["metadata"]["issue_keys"] == ["fiber_support"]
    assert "fiber_support" in chunk_payload["keywords"]


def test_ingest_payload_preserves_source_metadata_for_all_chunks_in_source() -> None:
    dyslipidemia_chunks = [chunk for chunk in build_rag_chunk_drafts() if chunk.source_id == "dyslipidemia"]
    item = _build_ingest_items(dyslipidemia_chunks, namespace="docs/rag_sources")[0]

    assert len(item.chunk_payloads) == 4
    for payload in item.chunk_payloads:
        metadata = payload["metadata"]
        assert metadata["source_org"] == "대한지질·동맥경화학회, 질병관리청 국가건강정보포털"
        assert metadata["source_url"].startswith("https://lipid.or.kr/")
        assert metadata["source_type"] == "clinical_guideline"
        assert metadata["source_trust_level"] == "official_guideline"
        assert "콜레스테롤" in metadata["topic_tags"]
        assert "cholesterol_management" in metadata["issue_keys"]
        assert metadata["usage_scope"]
        assert metadata["review_status"] == "candidate_unreviewed"
        assert metadata["safety_level"] == "normal"


def test_ingest_script_does_not_reference_vector_or_openai_runtime_calls() -> None:
    script = (Path(__file__).resolve().parents[2] / "scripts/rag/ingest_rag_chunks.py").read_text(encoding="utf-8")
    lowered = script.lower()

    assert "openai" not in lowered
    assert "create extension" not in lowered
    assert "pgvector" not in lowered
    assert "vector(" not in lowered
    assert "embedding_provider" not in lowered


def _item(
    source_key: str, disease_code: str, *, content: str = "테스트 chunk 내용입니다.", content_hash: str = "a" * 64
) -> RagDocumentIngestItem:
    return _build_ingest_items(
        [_chunk(source_key, disease_code, content=content, content_hash=content_hash)], namespace="docs/rag_sources"
    )[0]


def _chunk(source_key: str, disease_code: str, *, content: str, content_hash: str) -> RagChunkDraft:
    return RagChunkDraft(
        source_id=source_key,
        document_id=source_key,
        source_key=source_key,
        disease_code=disease_code,
        title=f"{source_key} 문서",
        source_org="테스트 기관",
        source_url=f"https://example.com/{source_key}",
        review_status="candidate_unreviewed",
        usage_scope="테스트용",
        topic_tags=("식단",),
        issue_keys=("fiber_support",),
        safety_level="normal",
        section_title="요약",
        chunk_index=0,
        chunk_key=f"rag:{source_key}:section:000:chunk:0000",
        content_hash=content_hash,
        content=content,
        content_length=len(content),
        filename=f"{source_key}.md",
        source_type="official",
        source_trust_level="unknown",
        year=2026,
        enabled=True,
        notes="테스트 notes",
        heading_path=("요약",),
        section_index=0,
        token_estimate=10,
    )
