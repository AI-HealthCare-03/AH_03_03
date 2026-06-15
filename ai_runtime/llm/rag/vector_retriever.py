from __future__ import annotations

import time
from typing import Any, Protocol

from tortoise import Tortoise

from ai_runtime.llm.rag.embeddings import EmbeddingProvider
from ai_runtime.llm.rag.rag_context_builder import build_reference_sources, build_reference_summary
from ai_runtime.llm.rag.retriever import RagRetrievalResult, RetrievedDocument
from ai_runtime.llm.rag.source_trust import source_trust_level_for_metadata
from ai_runtime.llm.rag.tracing import trace_vector_rag_retrieval

VECTOR_RETRIEVER_STRATEGY = "vector"
MAX_TRACE_QUERY_LENGTH = 500


class VectorSearchConnection(Protocol):
    async def execute_query_dict(self, query: str, values: list[Any] | None = None) -> list[dict[str, Any]]: ...


class VectorRagRetriever:
    strategy = VECTOR_RETRIEVER_STRATEGY

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider | None,
        connection: VectorSearchConnection | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.connection = connection

    async def retrieve(
        self,
        *,
        query: str,
        disease_type: str | None = None,
        disease_code: str | None = None,
        source_key: str | None = None,
        issue_keys: list[str] | tuple[str, ...] | None = None,
        topic_tags: list[str] | tuple[str, ...] | None = None,
        top_k: int = 3,
        include_safety_disclaimer: bool = False,
    ) -> RagRetrievalResult:
        started_at = time.monotonic()
        provider = self.embedding_provider
        effective_disease_code = disease_code or disease_type
        if provider is None:
            return self._fallback_result(
                query=query,
                top_k=top_k,
                disease_code=effective_disease_code,
                source_key=source_key,
                issue_keys=issue_keys,
                provider=None,
                fallback_reason="embedding_disabled",
                latency_ms=_elapsed_ms(started_at),
            )
        if provider.provider_name != "mock":
            return self._fallback_result(
                query=query,
                top_k=top_k,
                disease_code=effective_disease_code,
                source_key=source_key,
                issue_keys=issue_keys,
                provider=provider,
                fallback_reason="openai_embedding_disabled",
                latency_ms=_elapsed_ms(started_at),
            )

        query_vector = provider.embed_text(query)
        rows = await self._search_rows(
            query_vector=query_vector,
            top_k=max(top_k, 0),
            disease_code=effective_disease_code,
            source_key=source_key,
        )
        documents = [
            _row_to_document(row)
            for row in rows
            if _metadata_matches_filters(row.get("metadata"), issue_keys=issue_keys, topic_tags=topic_tags)
        ][: max(top_k, 0)]
        contexts = [document.to_context() for document in documents]
        fallback_reason = "no_result" if not documents else None
        latency_ms = _elapsed_ms(started_at)
        trace_vector_rag_retrieval(
            query=query,
            top_k=top_k,
            documents=documents,
            candidate_count=len(rows),
            disease_code=effective_disease_code,
            source_key=source_key,
            issue_keys=list(issue_keys or []),
            embedding_provider=provider.provider_name,
            embedding_model=provider.model_name,
            embedding_dimension=provider.dimension,
            latency_ms=latency_ms,
            fallback_reason=fallback_reason,
        )
        return RagRetrievalResult(
            documents=documents,
            reference_sources=build_reference_sources(contexts),
            reference_summary=build_reference_summary(contexts),
            strategy=self.strategy,
            fallback_reason=fallback_reason,
            trace_metadata=_build_vector_result_trace_metadata(
                query=query,
                top_k=top_k,
                documents=documents,
                candidate_count=len(rows),
                disease_code=effective_disease_code,
                source_key=source_key,
                issue_keys=list(issue_keys or []),
                provider=provider,
                latency_ms=latency_ms,
                fallback_reason=fallback_reason,
            ),
        )

    async def _search_rows(
        self,
        *,
        query_vector: list[float],
        top_k: int,
        disease_code: str | None,
        source_key: str | None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []

        conditions = [
            "c.is_active = true",
            "d.is_active = true",
            "c.embedding IS NOT NULL",
            "c.embedding_content_hash = c.content_hash",
        ]
        values: list[Any] = [_vector_to_pgvector_literal(query_vector)]
        if disease_code:
            values.append(str(disease_code).upper())
            conditions.append(f"UPPER(d.disease_code) = ${len(values)}")
        if source_key:
            values.append(str(source_key))
            conditions.append(f"d.source_key = ${len(values)}")
        values.append(top_k)
        limit_placeholder = f"${len(values)}"
        query = _build_vector_search_sql(conditions=conditions, limit_placeholder=limit_placeholder)
        return await self._connection().execute_query_dict(query, values)

    def _connection(self) -> VectorSearchConnection:
        return self.connection or Tortoise.get_connection("default")

    def _fallback_result(
        self,
        *,
        query: str,
        top_k: int,
        disease_code: str | None,
        source_key: str | None,
        issue_keys: list[str] | tuple[str, ...] | None,
        provider: EmbeddingProvider | None,
        fallback_reason: str,
        latency_ms: float,
    ) -> RagRetrievalResult:
        trace_metadata = _build_vector_result_trace_metadata(
            query=query,
            top_k=top_k,
            documents=[],
            candidate_count=0,
            disease_code=disease_code,
            source_key=source_key,
            issue_keys=list(issue_keys or []),
            provider=provider,
            latency_ms=latency_ms,
            fallback_reason=fallback_reason,
        )
        trace_vector_rag_retrieval(
            query=query,
            top_k=top_k,
            documents=[],
            candidate_count=0,
            disease_code=disease_code,
            source_key=source_key,
            issue_keys=list(issue_keys or []),
            embedding_provider=getattr(provider, "provider_name", None),
            embedding_model=getattr(provider, "model_name", None),
            embedding_dimension=getattr(provider, "dimension", None),
            latency_ms=latency_ms,
            fallback_reason=fallback_reason,
        )
        return RagRetrievalResult(
            documents=[],
            reference_sources=[],
            reference_summary=None,
            strategy=self.strategy,
            fallback_reason=fallback_reason,
            trace_metadata=trace_metadata,
        )


def _build_vector_search_sql(*, conditions: list[str], limit_placeholder: str) -> str:
    where_clause = "\n  AND ".join(conditions)
    return f"""
        SELECT
          c.chunk_key,
          c.content,
          c.content_hash,
          c.section_title,
          c.metadata,
          c.embedding_model,
          c.embedding_provider,
          c.embedding_dimension,
          d.document_key,
          d.source_key,
          d.disease_code,
          d.title,
          d.document_url,
          1 - (c.embedding <=> $1::vector) AS score
        FROM rag_chunks c
        JOIN rag_documents d ON c.document_id = d.id
        WHERE {where_clause}
        ORDER BY c.embedding <=> $1::vector
        LIMIT {limit_placeholder}
    """


def _row_to_document(row: dict[str, Any]) -> RetrievedDocument:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    merged_metadata = {
        **metadata,
        "id": row.get("chunk_key"),
        "chunk_key": row.get("chunk_key"),
        "document_key": row.get("document_key"),
        "source_key": row.get("source_key"),
        "disease_code": row.get("disease_code"),
        "section_title": row.get("section_title"),
        "source_url": row.get("document_url"),
        "retriever_strategy": VECTOR_RETRIEVER_STRATEGY,
        "embedding_model": row.get("embedding_model"),
        "embedding_provider": row.get("embedding_provider"),
        "embedding_dimension": row.get("embedding_dimension"),
        "score": _float_or_none(row.get("score")),
    }
    merged_metadata["source_trust_level"] = merged_metadata.get(
        "source_trust_level"
    ) or source_trust_level_for_metadata(merged_metadata)
    return RetrievedDocument(
        content=str(row.get("content") or ""),
        title=row.get("title"),
        source_name=metadata.get("source_org") or row.get("source_key"),
        url=row.get("document_url"),
        metadata=merged_metadata,
        score=_float_or_none(row.get("score")),
    )


def _metadata_matches_filters(
    metadata: Any,
    *,
    issue_keys: list[str] | tuple[str, ...] | None,
    topic_tags: list[str] | tuple[str, ...] | None,
) -> bool:
    if not issue_keys and not topic_tags:
        return True
    if not isinstance(metadata, dict):
        return False
    metadata_issue_keys = {str(value) for value in metadata.get("issue_keys") or []}
    metadata_topic_tags = {str(value) for value in metadata.get("topic_tags") or []}
    if issue_keys and not set(map(str, issue_keys)).intersection(metadata_issue_keys):
        return False
    if topic_tags and not set(map(str, topic_tags)).intersection(metadata_topic_tags):
        return False
    return True


def _build_vector_result_trace_metadata(
    *,
    query: str,
    top_k: int,
    documents: list[RetrievedDocument],
    candidate_count: int,
    disease_code: str | None,
    source_key: str | None,
    issue_keys: list[str],
    provider: EmbeddingProvider | None,
    latency_ms: float,
    fallback_reason: str | None,
) -> dict[str, Any]:
    return {
        "retriever_strategy": VECTOR_RETRIEVER_STRATEGY,
        "query_preview": _trace_query_preview(query),
        "query_length": len(query),
        "top_k": top_k,
        "candidate_count": candidate_count,
        "returned_count": len(documents),
        "disease_code": disease_code,
        "source_key": source_key,
        "issue_keys": issue_keys,
        "embedding_provider": getattr(provider, "provider_name", None),
        "embedding_model": getattr(provider, "model_name", None),
        "embedding_dimension": getattr(provider, "dimension", None),
        "scores": [document.score for document in documents],
        "chunk_keys": [str(document.metadata.get("chunk_key")) for document in documents],
        "latency_ms": latency_ms,
        "fallback_used": fallback_reason is not None,
        "fallback_reason": fallback_reason,
        "vector_rag": True,
        "embedding_search": True,
        "documents": [document.to_trace_metadata() for document in documents],
    }


def _vector_to_pgvector_literal(vector: list[float]) -> str:
    return "[" + ",".join(_format_vector_float(value) for value in vector) + "]"


def _format_vector_float(value: float) -> str:
    return format(float(value), ".12g")


def _trace_query_preview(query: str) -> str:
    return query[:MAX_TRACE_QUERY_LENGTH]


def _elapsed_ms(started_at: float) -> float:
    return round((time.monotonic() - started_at) * 1000, 3)


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None
