from datetime import date, datetime
from typing import Any

from pydantic import BaseModel

from app.dtos.base import BaseSerializerModel
from app.models.rag import RAGDiseaseType


class RAGSourceCreateRequest(BaseModel):
    source_key: str | None = None
    name: str
    organization: str | None = None
    source_type: str
    base_url: str | None = None
    description: str | None = None
    is_active: bool = True
    metadata: dict[str, Any] | None = None


class RAGSourceResponse(BaseSerializerModel):
    id: int
    source_key: str | None
    name: str
    organization: str | None
    source_type: str
    base_url: str | None
    description: str | None
    is_active: bool
    metadata: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime


class RAGDocumentCreateRequest(BaseModel):
    document_key: str | None = None
    source_key: str | None = None
    title: str
    disease_type: RAGDiseaseType
    disease_code: str | None = None
    filename: str | None = None
    document_url: str | None = None
    review_status: str | None = None
    usage_scope: str | None = None
    published_at: date | None = None
    fetched_at: datetime | None = None
    version: str | None = None
    is_active: bool = True
    metadata: dict[str, Any] | None = None


class RAGDocumentResponse(BaseSerializerModel):
    id: int
    source_id: int
    document_key: str | None
    source_key: str | None
    title: str
    disease_type: RAGDiseaseType
    disease_code: str | None
    filename: str | None
    document_url: str | None
    review_status: str | None
    usage_scope: str | None
    published_at: date | None
    fetched_at: datetime | None
    version: str | None
    is_active: bool
    metadata: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime


class RAGChunkCreateRequest(BaseModel):
    chunk_key: str | None = None
    chunk_index: int
    section_title: str | None = None
    content: str
    content_hash: str | None = None
    content_length: int | None = None
    token_estimate: int | None = None
    disease_type: RAGDiseaseType
    keywords: str | None = None
    embedding_model: str | None = None
    is_active: bool = True
    metadata: dict[str, Any] | None = None


class RAGChunkResponse(BaseSerializerModel):
    id: int
    document_id: int
    chunk_key: str | None
    chunk_index: int
    section_title: str | None
    content: str
    content_hash: str | None
    content_length: int | None
    token_estimate: int | None
    disease_type: RAGDiseaseType
    keywords: str | None
    embedding_model: str | None
    is_active: bool
    metadata: dict[str, Any] | None
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
