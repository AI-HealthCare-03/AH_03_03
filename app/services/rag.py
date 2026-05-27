from app.dtos.rag import (
    RAGChunkCreateRequest,
    RAGDocumentCreateRequest,
    RAGRetrievalLogCreateRequest,
    RAGSourceCreateRequest,
)
from app.models.rag import RAGChunk, RAGDocument, RAGRetrievalLog, RAGSource
from app.repositories import rag_repository

# 이 서비스는 RAG 메타데이터 CRUD 전용이다.
# 임베딩 생성, vector search, retrieval, context 구성, LLM 호출은 ai_runtime/llm/rag/에서 처리한다.


async def create_rag_source(request: RAGSourceCreateRequest) -> RAGSource:
    return await rag_repository.create_rag_source(request.model_dump())


async def list_active_rag_sources() -> list[RAGSource]:
    return await rag_repository.list_active_rag_sources()


async def get_rag_source(source_id: int) -> RAGSource | None:
    return await rag_repository.get_rag_source_by_id(source_id)


async def create_rag_document(source_id: int, request: RAGDocumentCreateRequest) -> RAGDocument:
    return await rag_repository.create_rag_document(source_id, request.model_dump())


async def get_rag_document(document_id: int) -> RAGDocument | None:
    return await rag_repository.get_rag_document_by_id(document_id)


async def list_active_rag_documents(
    disease_type: str | None = None, limit: int = 50, offset: int = 0
) -> list[RAGDocument]:
    return await rag_repository.list_active_rag_documents(disease_type=disease_type, limit=limit, offset=offset)


async def create_rag_chunk(document_id: int, request: RAGChunkCreateRequest) -> RAGChunk:
    return await rag_repository.create_rag_chunk(document_id, request.model_dump())


async def create_rag_chunks(document_id: int, chunks: list[RAGChunkCreateRequest]) -> list[RAGChunk]:
    data = [chunk.model_dump() for chunk in chunks]
    return await rag_repository.create_rag_chunks(document_id, data)


async def list_rag_chunks_by_document(document_id: int) -> list[RAGChunk]:
    return await rag_repository.list_rag_chunks_by_document(document_id)


async def list_rag_chunks_by_disease_type(disease_type: str, limit: int = 20, offset: int = 0) -> list[RAGChunk]:
    return await rag_repository.list_rag_chunks_by_disease_type(disease_type, limit=limit, offset=offset)


async def create_rag_retrieval_log(
    user_id: int | None, analysis_result_id: int | None, request: RAGRetrievalLogCreateRequest
) -> RAGRetrievalLog:
    return await rag_repository.create_rag_retrieval_log(
        user_id=user_id,
        analysis_result_id=analysis_result_id,
        data=request.model_dump(),
    )
