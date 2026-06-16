from __future__ import annotations

from typing import Any

from ai_runtime.llm.llm_client import record_langfuse_event
from ai_runtime.llm.rag.retriever import RetrievedDocument
from ai_runtime.llm.schemas import RetrievedContext
from app.core import config

KEYWORD_RAG_TRACE_VERSION = "keyword_rag_poc_v1"
KEYWORD_RAG_PROMPT_VERSION = "keyword_rag_context_v1"
MAX_TRACE_QUERY_LENGTH = 500


def build_keyword_rag_trace_metadata(
    *,
    query: str,
    disease_type: str | None,
    contexts: list[RetrievedContext],
    top_k: int,
    include_safety_disclaimer: bool,
) -> dict[str, Any]:
    retrieved_source_ids = [str(context.metadata.get("id")) for context in contexts if context.metadata.get("id")]
    source_status = {
        str(context.metadata.get("id")): context.metadata.get("status")
        for context in contexts
        if context.metadata.get("id")
    }
    fallback_reason = _fallback_reason(contexts)
    return {
        "trace_version": KEYWORD_RAG_TRACE_VERSION,
        "prompt_version": KEYWORD_RAG_PROMPT_VERSION,
        "source": "keyword_rag_poc",
        "retrieval_strategy": "local_markdown_keyword_match",
        "disease_type": disease_type,
        "query_preview": _trace_query_preview(query),
        "query_length": len(query),
        "query_truncated": len(query) > MAX_TRACE_QUERY_LENGTH,
        "top_k": top_k,
        "include_safety_disclaimer": include_safety_disclaimer,
        "retrieved_source_count": len(contexts),
        "retrieved_source_ids": retrieved_source_ids,
        "source_status": source_status,
        "retrieved_sources": [build_context_source_metadata(context) for context in contexts],
        "fallback": fallback_reason is not None,
        "fallback_reason": fallback_reason,
        "llm_call": False,
        "vector_rag": False,
        "embedding_search": False,
    }


def build_context_source_metadata(context: RetrievedContext) -> dict[str, Any]:
    return {
        "id": context.metadata.get("id"),
        "title": context.title,
        "source_org": context.source_name,
        "source_url": context.url,
        "year": context.metadata.get("year"),
        "status": context.metadata.get("status"),
        "match_reason": context.metadata.get("match_reason"),
        "matched_keywords": context.metadata.get("matched_keywords", []),
    }


def trace_keyword_rag_retrieval(
    *,
    query: str,
    disease_type: str | None,
    contexts: list[RetrievedContext],
    top_k: int,
    include_safety_disclaimer: bool,
) -> bool:
    if not config.RAG_ENABLED:
        return False

    metadata = build_keyword_rag_trace_metadata(
        query=query,
        disease_type=disease_type,
        contexts=contexts,
        top_k=top_k,
        include_safety_disclaimer=include_safety_disclaimer,
    )
    return record_langfuse_event(
        name="rag.keyword_retrieval",
        input_payload={
            "query_preview": _trace_query_preview(query),
            "query_length": len(query),
            "disease_type": disease_type,
            "top_k": top_k,
        },
        output_payload={
            "retrieved_source_count": len(contexts),
            "retrieved_source_ids": metadata["retrieved_source_ids"],
            "fallback": metadata["fallback"],
        },
        metadata=metadata,
    )


def build_vector_rag_trace_metadata(
    *,
    query: str,
    top_k: int,
    documents: list[RetrievedDocument],
    candidate_count: int,
    disease_code: str | None = None,
    source_key: str | None = None,
    issue_keys: list[str] | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    embedding_dimension: int | None = None,
    latency_ms: float | None = None,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "trace_version": "vector_rag_poc_v1",
        "source": "vector_rag_poc",
        "retriever_strategy": "vector",
        "query_preview": _trace_query_preview(query),
        "query_length": len(query),
        "query_truncated": len(query) > MAX_TRACE_QUERY_LENGTH,
        "top_k": top_k,
        "candidate_count": candidate_count,
        "returned_count": len(documents),
        "disease_code": disease_code,
        "source_key": source_key,
        "issue_keys": issue_keys or [],
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "scores": [document.score for document in documents],
        "chunk_keys": [str(document.metadata.get("chunk_key")) for document in documents],
        "latency_ms": latency_ms,
        "fallback_used": fallback_reason is not None,
        "fallback_reason": fallback_reason,
        "vector_rag": True,
        "embedding_search": True,
        "retrieved_sources": [document.to_trace_metadata() for document in documents],
    }


def trace_vector_rag_retrieval(
    *,
    query: str,
    top_k: int,
    documents: list[RetrievedDocument],
    candidate_count: int,
    disease_code: str | None = None,
    source_key: str | None = None,
    issue_keys: list[str] | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    embedding_dimension: int | None = None,
    latency_ms: float | None = None,
    fallback_reason: str | None = None,
) -> bool:
    if not config.RAG_ENABLED:
        return False

    metadata = build_vector_rag_trace_metadata(
        query=query,
        top_k=top_k,
        documents=documents,
        candidate_count=candidate_count,
        disease_code=disease_code,
        source_key=source_key,
        issue_keys=issue_keys,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        latency_ms=latency_ms,
        fallback_reason=fallback_reason,
    )
    return record_langfuse_event(
        name="rag.vector_retrieval",
        input_payload={
            "query_preview": metadata["query_preview"],
            "query_length": metadata["query_length"],
            "disease_code": disease_code,
            "source_key": source_key,
            "top_k": top_k,
        },
        output_payload={
            "candidate_count": candidate_count,
            "returned_count": len(documents),
            "chunk_keys": metadata["chunk_keys"],
            "fallback_used": metadata["fallback_used"],
        },
        metadata=metadata,
    )


def build_hybrid_rag_trace_metadata(
    *,
    query: str,
    strategy: str,
    top_k: int,
    documents: list[RetrievedDocument],
    keyword_returned_count: int,
    vector_returned_count: int,
    fallback_used: bool,
    fallback_reason: str | None,
    disease_code: str | None = None,
    source_key: str | None = None,
    issue_keys: list[str] | None = None,
    latency_ms: float | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    embedding_dimension: int | None = None,
    keyword_weight: float | None = None,
    vector_weight: float | None = None,
) -> dict[str, Any]:
    return {
        "trace_version": "hybrid_rag_poc_v1",
        "source": "hybrid_rag_poc",
        "retriever_strategy": strategy,
        "query_preview": _trace_query_preview(query),
        "query_length": len(query),
        "query_truncated": len(query) > MAX_TRACE_QUERY_LENGTH,
        "top_k": top_k,
        "keyword_returned_count": keyword_returned_count,
        "vector_returned_count": vector_returned_count,
        "merged_count": len(documents),
        "final_count": len(documents),
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "keyword_weight": keyword_weight,
        "vector_weight": vector_weight,
        "disease_code": disease_code,
        "source_key": source_key,
        "issue_keys": issue_keys or [],
        "selected_chunk_keys": [
            str(document.metadata.get("chunk_key") or document.metadata.get("id")) for document in documents
        ],
        "latency_ms": latency_ms,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "vector_rag": vector_returned_count > 0,
        "embedding_search": vector_returned_count > 0,
        "retrieved_sources": [document.to_trace_metadata() for document in documents],
    }


def trace_hybrid_rag_retrieval(
    *,
    query: str,
    strategy: str,
    top_k: int,
    documents: list[RetrievedDocument],
    keyword_returned_count: int,
    vector_returned_count: int,
    fallback_used: bool,
    fallback_reason: str | None,
    disease_code: str | None = None,
    source_key: str | None = None,
    issue_keys: list[str] | None = None,
    latency_ms: float | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    embedding_dimension: int | None = None,
    keyword_weight: float | None = None,
    vector_weight: float | None = None,
) -> bool:
    if not config.RAG_ENABLED:
        return False

    metadata = build_hybrid_rag_trace_metadata(
        query=query,
        strategy=strategy,
        top_k=top_k,
        documents=documents,
        keyword_returned_count=keyword_returned_count,
        vector_returned_count=vector_returned_count,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        disease_code=disease_code,
        source_key=source_key,
        issue_keys=issue_keys,
        latency_ms=latency_ms,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        keyword_weight=keyword_weight,
        vector_weight=vector_weight,
    )
    return record_langfuse_event(
        name="rag.hybrid_retrieval",
        input_payload={
            "query_preview": metadata["query_preview"],
            "query_length": metadata["query_length"],
            "strategy": strategy,
            "top_k": top_k,
        },
        output_payload={
            "keyword_returned_count": keyword_returned_count,
            "vector_returned_count": vector_returned_count,
            "merged_count": len(documents),
            "final_count": len(documents),
            "fallback_used": fallback_used,
        },
        metadata=metadata,
    )


def build_embedding_rag_trace_metadata(
    *,
    provider: str,
    model: str | None,
    dimension: int | None,
    batch_size: int,
    chunk_count: int,
    failed_count: int,
    estimated_char_count: int,
    apply: bool,
    latency_ms: float | None = None,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "trace_version": "rag_embedding_apply_v1",
        "source": "rag_embedding",
        "provider": provider,
        "model": model,
        "dimension": dimension,
        "batch_size": batch_size,
        "chunk_count": chunk_count,
        "failed_count": failed_count,
        "estimated_char_count": estimated_char_count,
        "apply": apply,
        "latency_ms": latency_ms,
        "fallback_reason": fallback_reason,
        "embedding_vector_logged": False,
        "chunk_content_logged": False,
    }


def _trace_query_preview(query: str) -> str:
    return query[:MAX_TRACE_QUERY_LENGTH]


def _fallback_reason(contexts: list[RetrievedContext]) -> str | None:
    if not contexts:
        return "no_keyword_match"
    source_ids = {str(context.metadata.get("id")) for context in contexts}
    if source_ids <= {"safety_disclaimer", "diet_caution"}:
        return "safety_disclaimer_only"
    return None
