from pathlib import Path

from app.models.rag import RAGChunk

MIGRATION_12_PATH = (
    Path(__file__).resolve().parents[2] / "app/core/db/migrations/models/12_20260615030000_add_rag_metadata_schema.py"
)
MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "app/core/db/migrations/models/13_20260615031000_add_rag_chunk_embedding_vector.py"
)


def _normalized_migration_sql() -> str:
    migration_sql = MIGRATION_PATH.read_text(encoding="utf-8").lower()
    return " ".join(migration_sql.split())


def test_recent_rag_migrations_include_aerich_models_state() -> None:
    assert "MODELS_STATE = (" in MIGRATION_12_PATH.read_text(encoding="utf-8")
    assert "MODELS_STATE = (" in MIGRATION_PATH.read_text(encoding="utf-8")


def test_pgvector_migration_adds_extension_and_vector_column() -> None:
    migration_sql = _normalized_migration_sql()

    assert "create extension if not exists vector" in migration_sql
    assert 'alter table "rag_chunks" add column if not exists "embedding" vector(1536)' in migration_sql


def test_pgvector_migration_adds_embedding_metadata_columns() -> None:
    migration_sql = _normalized_migration_sql()

    assert '"embedding_provider" varchar(100)' in migration_sql
    assert '"embedding_dimension" int' in migration_sql
    assert '"embedding_content_hash" varchar(64)' in migration_sql
    assert '"embedded_at" timestamptz' in migration_sql


def test_pgvector_migration_does_not_create_vector_index_or_drop_extension() -> None:
    migration_sql = _normalized_migration_sql()

    assert "create index" not in migration_sql
    assert "using hnsw" not in migration_sql
    assert "using ivfflat" not in migration_sql
    assert "drop extension" not in migration_sql


def test_rag_chunk_model_keeps_embedding_model_and_scalar_metadata_fields() -> None:
    fields = RAGChunk._meta.fields_map

    assert "embedding_model" in fields
    assert "embedding_provider" in fields
    assert "embedding_dimension" in fields
    assert "embedding_content_hash" in fields
    assert "embedded_at" in fields
    assert "embedding" not in fields
