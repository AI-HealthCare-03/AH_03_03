from pathlib import Path

from app.models.rag import RAGChunk, RAGDiseaseType, RAGDocument, RAGSource

MIGRATION_PATH = (
    Path(__file__).resolve().parents[2] / "app/core/db/migrations/models/12_20260615030000_add_rag_metadata_schema.py"
)


def test_rag_metadata_model_fields_are_optional_and_creatable() -> None:
    source = RAGSource(
        name="공식 출처",
        source_type="official",
        source_key="diet_nutrition",
        metadata={"source_org": "테스트 기관"},
    )
    document = RAGDocument(
        source_id=1,
        document_key="diet_nutrition.md",
        source_key="diet_nutrition",
        title="건강한 식생활",
        disease_type=RAGDiseaseType.COMMON,
        disease_code="DIET_NUTRITION",
        filename="diet_nutrition.md",
        review_status="candidate_unreviewed",
        usage_scope="diet_recommendation",
        metadata={"topic_tags": ["식단"], "issue_keys": ["fiber_support"]},
    )
    chunk = RAGChunk(
        document_id=1,
        chunk_key="rag:diet_nutrition:section:000:chunk:0000",
        chunk_index=0,
        section_title="요약",
        content="실제 섭취량이 확정되지 않은 식단 조언은 참고용으로 제공한다.",
        content_hash="a" * 64,
        content_length=36,
        token_estimate=12,
        disease_type=RAGDiseaseType.COMMON,
        metadata={"heading_path": ["요약"], "safety_level": "normal"},
    )

    assert source.source_key == "diet_nutrition"
    assert source.metadata == {"source_org": "테스트 기관"}
    assert document.disease_code == "DIET_NUTRITION"
    assert document.metadata == {"topic_tags": ["식단"], "issue_keys": ["fiber_support"]}
    assert chunk.chunk_key == "rag:diet_nutrition:section:000:chunk:0000"
    assert chunk.is_active is True
    assert chunk.metadata == {"heading_path": ["요약"], "safety_level": "normal"}


def test_rag_legacy_disease_type_enum_is_unchanged() -> None:
    assert {item.value for item in RAGDiseaseType} == {
        "DIABETES",
        "OBESITY",
        "DYSLIPIDEMIA",
        "COMMON",
    }


def test_rag_chunk_model_keeps_metadata_out_of_extra_columns() -> None:
    prohibited_chunk_columns = {
        "source_key",
        "disease_code",
        "review_status",
        "usage_scope",
        "safety_level",
        "heading_path",
        "topic_tags",
        "issue_keys",
        "section_index",
    }

    assert prohibited_chunk_columns.isdisjoint(RAGChunk._meta.fields_map)
    assert "metadata" in RAGChunk._meta.fields_map


def test_rag_key_fields_are_unique_candidates() -> None:
    assert getattr(RAGSource._meta.fields_map["source_key"], "unique", False) is True
    assert getattr(RAGDocument._meta.fields_map["document_key"], "unique", False) is True
    assert getattr(RAGChunk._meta.fields_map["chunk_key"], "unique", False) is True


def test_rag_metadata_migration_excludes_vector_and_embedding_sql() -> None:
    migration_sql = MIGRATION_PATH.read_text(encoding="utf-8").lower()

    assert "create extension" not in migration_sql
    assert "pgvector" not in migration_sql
    assert " vector" not in migration_sql
    assert "embedding_provider" not in migration_sql
    assert "embedding_version" not in migration_sql
    assert "embedding(" not in migration_sql
    assert '"source_key"' not in _chunk_column_additions(migration_sql)
    assert '"disease_code"' not in _chunk_column_additions(migration_sql)
    assert '"topic_tags"' not in _chunk_column_additions(migration_sql)
    assert '"issue_keys"' not in _chunk_column_additions(migration_sql)


def _chunk_column_additions(migration_sql: str) -> str:
    additions = [line for line in migration_sql.splitlines() if 'alter table "rag_chunks" add column' in line.lower()]
    return "\n".join(additions)
