from datetime import date, datetime
from typing import Any

from pydantic import BaseModel

from app.dtos.base import BaseSerializerModel
from app.models.rag import RAGDiseaseType


class RAGSourceCreateRequest(BaseModel):
    name: str
    organization: str | None = None
    source_type: str
    base_url: str | None = None
    description: str | None = None
    is_active: bool = True


class RAGSourceResponse(BaseSerializerModel):
    id: int
    name: str
    organization: str | None
    source_type: str
    base_url: str | None
    description: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class RAGDocumentCreateRequest(BaseModel):
    title: str
    disease_type: RAGDiseaseType
    document_url: str | None = None
    published_at: date | None = None
    fetched_at: datetime | None = None
    version: str | None = None
    is_active: bool = True


class RAGDocumentResponse(BaseSerializerModel):
    id: int
    source_id: int
    title: str
    disease_type: RAGDiseaseType
    document_url: str | None
    published_at: date | None
    fetched_at: datetime | None
    version: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class RAGChunkCreateRequest(BaseModel):
    chunk_index: int
    section_title: str | None = None
    content: str
    disease_type: RAGDiseaseType
    keywords: str | None = None
    embedding_model: str | None = None


class RAGChunkResponse(BaseSerializerModel):
    id: int
    document_id: int
    chunk_index: int
    section_title: str | None
    content: str
    disease_type: RAGDiseaseType
    keywords: str | None
    embedding_model: str | None
    created_at: datetime
    updated_at: datetime


class RAGRetrievalLogCreateRequest(BaseModel):
    query_text: str
    disease_type: RAGDiseaseType | None = None
    retrieved_chunk_ids: list[int] | dict[str, Any] | None = None
    top_k: int | None = None


class RAGRetrievalLogResponse(BaseSerializerModel):
    id: int
    user_id: int | None
    analysis_result_id: int | None
    query_text: str
    disease_type: RAGDiseaseType | None
    retrieved_chunk_ids: list[int] | dict[str, Any] | None
    top_k: int | None
    created_at: datetime
