from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "rag_sources" ADD COLUMN IF NOT EXISTS "source_key" VARCHAR(100);
        ALTER TABLE "rag_sources" ADD COLUMN IF NOT EXISTS "metadata" JSONB;
        CREATE UNIQUE INDEX IF NOT EXISTS "uid_rag_sources_source_key" ON "rag_sources" ("source_key");

        ALTER TABLE "rag_documents" ADD COLUMN IF NOT EXISTS "document_key" VARCHAR(200);
        ALTER TABLE "rag_documents" ADD COLUMN IF NOT EXISTS "source_key" VARCHAR(100);
        ALTER TABLE "rag_documents" ADD COLUMN IF NOT EXISTS "disease_code" VARCHAR(50);
        ALTER TABLE "rag_documents" ADD COLUMN IF NOT EXISTS "filename" VARCHAR(255);
        ALTER TABLE "rag_documents" ADD COLUMN IF NOT EXISTS "review_status" VARCHAR(50);
        ALTER TABLE "rag_documents" ADD COLUMN IF NOT EXISTS "usage_scope" TEXT;
        ALTER TABLE "rag_documents" ADD COLUMN IF NOT EXISTS "metadata" JSONB;
        CREATE UNIQUE INDEX IF NOT EXISTS "uid_rag_documents_document_key" ON "rag_documents" ("document_key");
        CREATE INDEX IF NOT EXISTS "idx_rag_documents_source_key" ON "rag_documents" ("source_key");
        CREATE INDEX IF NOT EXISTS "idx_rag_documents_disease_code_active" ON "rag_documents" ("disease_code", "is_active");
        CREATE INDEX IF NOT EXISTS "idx_rag_documents_review_status_active" ON "rag_documents" ("review_status", "is_active");

        ALTER TABLE "rag_chunks" ADD COLUMN IF NOT EXISTS "chunk_key" VARCHAR(200);
        ALTER TABLE "rag_chunks" ADD COLUMN IF NOT EXISTS "content_hash" VARCHAR(64);
        ALTER TABLE "rag_chunks" ADD COLUMN IF NOT EXISTS "content_length" INT;
        ALTER TABLE "rag_chunks" ADD COLUMN IF NOT EXISTS "token_estimate" INT;
        ALTER TABLE "rag_chunks" ADD COLUMN IF NOT EXISTS "is_active" BOOL NOT NULL DEFAULT TRUE;
        ALTER TABLE "rag_chunks" ADD COLUMN IF NOT EXISTS "metadata" JSONB;
        CREATE UNIQUE INDEX IF NOT EXISTS "uid_rag_chunks_chunk_key" ON "rag_chunks" ("chunk_key");
        CREATE INDEX IF NOT EXISTS "idx_rag_chunks_content_hash" ON "rag_chunks" ("content_hash");
        CREATE INDEX IF NOT EXISTS "idx_rag_chunks_is_active" ON "rag_chunks" ("is_active");
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP INDEX IF EXISTS "idx_rag_chunks_is_active";
        DROP INDEX IF EXISTS "idx_rag_chunks_content_hash";
        DROP INDEX IF EXISTS "uid_rag_chunks_chunk_key";
        ALTER TABLE "rag_chunks" DROP COLUMN IF EXISTS "metadata";
        ALTER TABLE "rag_chunks" DROP COLUMN IF EXISTS "is_active";
        ALTER TABLE "rag_chunks" DROP COLUMN IF EXISTS "token_estimate";
        ALTER TABLE "rag_chunks" DROP COLUMN IF EXISTS "content_length";
        ALTER TABLE "rag_chunks" DROP COLUMN IF EXISTS "content_hash";
        ALTER TABLE "rag_chunks" DROP COLUMN IF EXISTS "chunk_key";

        DROP INDEX IF EXISTS "idx_rag_documents_review_status_active";
        DROP INDEX IF EXISTS "idx_rag_documents_disease_code_active";
        DROP INDEX IF EXISTS "idx_rag_documents_source_key";
        DROP INDEX IF EXISTS "uid_rag_documents_document_key";
        ALTER TABLE "rag_documents" DROP COLUMN IF EXISTS "metadata";
        ALTER TABLE "rag_documents" DROP COLUMN IF EXISTS "usage_scope";
        ALTER TABLE "rag_documents" DROP COLUMN IF EXISTS "review_status";
        ALTER TABLE "rag_documents" DROP COLUMN IF EXISTS "filename";
        ALTER TABLE "rag_documents" DROP COLUMN IF EXISTS "disease_code";
        ALTER TABLE "rag_documents" DROP COLUMN IF EXISTS "source_key";
        ALTER TABLE "rag_documents" DROP COLUMN IF EXISTS "document_key";

        DROP INDEX IF EXISTS "uid_rag_sources_source_key";
        ALTER TABLE "rag_sources" DROP COLUMN IF EXISTS "metadata";
        ALTER TABLE "rag_sources" DROP COLUMN IF EXISTS "source_key";
    """


MODELS_STATE = (
    "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"
)
