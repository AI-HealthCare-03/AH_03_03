from __future__ import annotations

from typing import Any

from ai_worker.llm.llm_client import record_langfuse_event
from ai_worker.llm.schemas import RetrievedContext

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


def _trace_query_preview(query: str) -> str:
    return query[:MAX_TRACE_QUERY_LENGTH]


def _fallback_reason(contexts: list[RetrievedContext]) -> str | None:
    if not contexts:
        return "no_keyword_match"
    source_ids = {str(context.metadata.get("id")) for context in contexts}
    if source_ids == {"safety_disclaimer"}:
        return "safety_disclaimer_only"
    return None
