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
