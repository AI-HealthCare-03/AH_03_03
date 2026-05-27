from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "rag_sources" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "name" VARCHAR(100) NOT NULL,
    "organization" VARCHAR(100),
    "source_type" VARCHAR(30) NOT NULL,
    "base_url" VARCHAR(500),
    "description" TEXT,
    "is_active" BOOL NOT NULL DEFAULT TRUE,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS "rag_documents" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "source_id" BIGINT NOT NULL REFERENCES "rag_sources" ("id") ON DELETE CASCADE,
    "title" VARCHAR(255) NOT NULL,
    "disease_type" VARCHAR(20) NOT NULL,
    "document_url" VARCHAR(500),
    "published_at" DATE,
    "fetched_at" TIMESTAMPTZ,
    "version" VARCHAR(50),
    "is_active" BOOL NOT NULL DEFAULT TRUE,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_rag_documents_source"
    ON "rag_documents" ("source_id");
CREATE INDEX IF NOT EXISTS "idx_rag_documents_disease_active"
    ON "rag_documents" ("disease_type", "is_active");

CREATE TABLE IF NOT EXISTS "rag_chunks" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "document_id" BIGINT NOT NULL REFERENCES "rag_documents" ("id") ON DELETE CASCADE,
    "chunk_index" INT NOT NULL,
    "section_title" VARCHAR(255),
    "content" TEXT NOT NULL,
    "disease_type" VARCHAR(20) NOT NULL,
    "keywords" TEXT,
    "embedding_model" VARCHAR(100),
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_rag_chunks_document_index"
    ON "rag_chunks" ("document_id", "chunk_index");
CREATE INDEX IF NOT EXISTS "idx_rag_chunks_disease"
    ON "rag_chunks" ("disease_type");

CREATE TABLE IF NOT EXISTS "rag_retrieval_logs" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "user_id" BIGINT REFERENCES "users" ("id") ON DELETE SET NULL,
    "analysis_result_id" BIGINT REFERENCES "analysis_results" ("id") ON DELETE SET NULL,
    "query_text" TEXT NOT NULL,
    "disease_type" VARCHAR(20),
    "retrieved_chunk_ids" JSONB,
    "top_k" INT,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_rag_retrieval_logs_user_created"
    ON "rag_retrieval_logs" ("user_id", "created_at");
CREATE INDEX IF NOT EXISTS "idx_rag_retrieval_logs_analysis_result"
    ON "rag_retrieval_logs" ("analysis_result_id");
"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "rag_retrieval_logs";
DROP TABLE IF EXISTS "rag_chunks";
DROP TABLE IF EXISTS "rag_documents";
DROP TABLE IF EXISTS "rag_sources";
"""
