from typing import Any

from app.models.rag import RAGChunk, RAGDocument, RAGRetrievalLog, RAGSource


async def create_rag_source(data: dict[str, Any]) -> RAGSource:
    return await RAGSource.create(**data)


async def list_active_rag_sources() -> list[RAGSource]:
    return await RAGSource.filter(is_active=True).order_by("name")


async def get_rag_source_by_id(source_id: int) -> RAGSource | None:
    return await RAGSource.get_or_none(id=source_id)


async def create_rag_document(source_id: int, data: dict[str, Any]) -> RAGDocument:
    return await RAGDocument.create(source_id=source_id, **data)


async def get_rag_document_by_id(document_id: int) -> RAGDocument | None:
    return await RAGDocument.get_or_none(id=document_id)


async def list_active_rag_documents(
    disease_type: str | None = None, limit: int = 50, offset: int = 0
) -> list[RAGDocument]:
    query = RAGDocument.filter(is_active=True)
    if disease_type is not None:
        query = query.filter(disease_type=disease_type)
    return await query.order_by("-published_at", "-created_at").offset(offset).limit(limit)


async def create_rag_chunk(document_id: int, data: dict[str, Any]) -> RAGChunk:
    return await RAGChunk.create(document_id=document_id, **data)


async def create_rag_chunks(document_id: int, chunks: list[dict[str, Any]]) -> list[RAGChunk]:
    objects = [RAGChunk(document_id=document_id, **chunk) for chunk in chunks]
    if not objects:
        return []
    await RAGChunk.bulk_create(objects)
    return objects


async def list_rag_chunks_by_document(document_id: int) -> list[RAGChunk]:
    return await RAGChunk.filter(document_id=document_id).order_by("chunk_index")


async def list_rag_chunks_by_disease_type(disease_type: str, limit: int = 20, offset: int = 0) -> list[RAGChunk]:
    return (
        await RAGChunk.filter(disease_type=disease_type)
        .order_by("document_id", "chunk_index")
        .offset(offset)
        .limit(limit)
    )


async def create_rag_retrieval_log(
    user_id: int | None, analysis_result_id: int | None, data: dict[str, Any]
) -> RAGRetrievalLog:
    return await RAGRetrievalLog.create(user_id=user_id, analysis_result_id=analysis_result_id, **data)
