from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE EXTENSION IF NOT EXISTS vector;
        ALTER TABLE "rag_chunks" ADD COLUMN IF NOT EXISTS "embedding" vector(1536);
        ALTER TABLE "rag_chunks" ADD COLUMN IF NOT EXISTS "embedding_provider" VARCHAR(100);
        ALTER TABLE "rag_chunks" ADD COLUMN IF NOT EXISTS "embedding_dimension" INT;
        ALTER TABLE "rag_chunks" ADD COLUMN IF NOT EXISTS "embedding_content_hash" VARCHAR(64);
        ALTER TABLE "rag_chunks" ADD COLUMN IF NOT EXISTS "embedded_at" TIMESTAMPTZ;
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "rag_chunks" DROP COLUMN IF EXISTS "embedded_at";
        ALTER TABLE "rag_chunks" DROP COLUMN IF EXISTS "embedding_content_hash";
        ALTER TABLE "rag_chunks" DROP COLUMN IF EXISTS "embedding_dimension";
        ALTER TABLE "rag_chunks" DROP COLUMN IF EXISTS "embedding_provider";
        ALTER TABLE "rag_chunks" DROP COLUMN IF EXISTS "embedding";
    """
