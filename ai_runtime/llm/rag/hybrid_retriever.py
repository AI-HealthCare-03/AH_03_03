from __future__ import annotations

import inspect
import time
from typing import Any

from ai_runtime.llm.rag.rag_context_builder import build_reference_sources, build_reference_summary
from ai_runtime.llm.rag.retriever import KeywordRagRetriever, RagRetrievalResult, RetrievedDocument
from ai_runtime.llm.rag.tracing import trace_hybrid_rag_retrieval

HYBRID_STRATEGY_KEYWORD_ONLY = "keyword_only"
HYBRID_STRATEGY_KEYWORD_FIRST_VECTOR_FALLBACK = "keyword_first_vector_fallback"
HYBRID_STRATEGY_VECTOR_DISABLED = "vector_disabled"
SUPPORTED_HYBRID_STRATEGIES = {
    HYBRID_STRATEGY_KEYWORD_ONLY,
    HYBRID_STRATEGY_KEYWORD_FIRST_VECTOR_FALLBACK,
    HYBRID_STRATEGY_VECTOR_DISABLED,
}


class HybridRagRetriever:
    """Strategy helper for keyword/vector retrieval.

    This helper is intentionally not wired to product APIs yet. It lets tests and
    future callers exercise fallback policy before vector retrieval becomes a
    user-facing path.
    """

    def __init__(
        self,
        *,
        keyword_retriever: Any | None = None,
        vector_retriever: Any | None = None,
    ) -> None:
        self.keyword_retriever = keyword_retriever or KeywordRagRetriever()
        self.vector_retriever = vector_retriever

    async def retrieve(
        self,
        *,
        query_text: str | None = None,
        query: str | None = None,
        top_k: int = 3,
        disease_code: str | None = None,
        disease_type: str | None = None,
        source_key: str | None = None,
        issue_keys: list[str] | tuple[str, ...] | None = None,
        topic_tags: list[str] | tuple[str, ...] | None = None,
        strategy: str = HYBRID_STRATEGY_KEYWORD_ONLY,
        min_keyword_results: int = 1,
        include_safety_disclaimer: bool = False,
    ) -> RagRetrievalResult:
        started_at = time.monotonic()
        normalized_strategy = _normalize_strategy(strategy)
        effective_query = query_text if query_text is not None else query or ""
        effective_disease_code = disease_code or disease_type
        keyword_result = await _call_keyword_retriever(
            self.keyword_retriever,
            query=effective_query,
            disease_type=effective_disease_code,
            top_k=top_k,
            include_safety_disclaimer=include_safety_disclaimer,
        )
        keyword_count = len(keyword_result.documents)
        vector_result: RagRetrievalResult | None = None
        vector_error: str | None = None
        fallback_used = False
        fallback_reason: str | None = None

        if normalized_strategy == HYBRID_STRATEGY_KEYWORD_ONLY:
            documents = keyword_result.documents
            fallback_reason = "keyword_only"
        elif normalized_strategy == HYBRID_STRATEGY_VECTOR_DISABLED:
            documents = keyword_result.documents
            fallback_reason = "vector_disabled"
        elif keyword_count >= max(min_keyword_results, 0):
            documents = keyword_result.documents
            fallback_reason = None
        else:
            fallback_used = True
            fallback_reason = "no_keyword_result" if keyword_count == 0 else "insufficient_keyword_results"
            if self.vector_retriever is None:
                vector_error = "vector_retriever_unavailable"
                documents = keyword_result.documents
            else:
                try:
                    vector_result = await _call_vector_retriever(
                        self.vector_retriever,
                        query=effective_query,
                        disease_code=effective_disease_code,
                        source_key=source_key,
                        issue_keys=list(issue_keys or []),
                        topic_tags=list(topic_tags or []),
                        top_k=top_k,
                        include_safety_disclaimer=include_safety_disclaimer,
                    )
                    documents = deduplicate_retrieved_documents([*keyword_result.documents, *vector_result.documents])[
                        : max(top_k, 0)
                    ]
                    if not vector_result.documents:
                        vector_error = vector_result.fallback_reason or "vector_no_result"
                except Exception as exc:  # pragma: no cover - exact exception depends on injected retriever
                    vector_error = f"vector_retrieval_failed:{type(exc).__name__}"
                    documents = keyword_result.documents
        contexts = [document.to_context() for document in documents]
        result_fallback_reason = _result_fallback_reason(documents)
        latency_ms = _elapsed_ms(started_at)
        trace_metadata = build_hybrid_result_trace_metadata(
            query=effective_query,
            strategy=normalized_strategy,
            keyword_result=keyword_result,
            vector_result=vector_result,
            documents=documents,
            top_k=top_k,
            disease_code=effective_disease_code,
            source_key=source_key,
            issue_keys=list(issue_keys or []),
            fallback_used=fallback_used,
            fallback_reason=vector_error or fallback_reason,
            latency_ms=latency_ms,
        )
        trace_hybrid_rag_retrieval(
            query=effective_query,
            strategy=normalized_strategy,
            top_k=top_k,
            documents=documents,
            keyword_returned_count=keyword_count,
            vector_returned_count=len(vector_result.documents) if vector_result else 0,
            fallback_used=fallback_used,
            fallback_reason=vector_error or fallback_reason,
            disease_code=effective_disease_code,
            source_key=source_key,
            issue_keys=list(issue_keys or []),
            latency_ms=latency_ms,
            embedding_provider=trace_metadata.get("embedding_provider") if isinstance(trace_metadata, dict) else None,
            embedding_model=trace_metadata.get("embedding_model") if isinstance(trace_metadata, dict) else None,
            embedding_dimension=trace_metadata.get("embedding_dimension") if isinstance(trace_metadata, dict) else None,
        )
        return RagRetrievalResult(
            documents=documents,
            reference_sources=build_reference_sources(contexts),
            reference_summary=build_reference_summary(contexts),
            strategy=normalized_strategy,
            fallback_reason=result_fallback_reason,
            trace_metadata=trace_metadata,
        )


def deduplicate_retrieved_documents(documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
    seen: set[str] = set()
    deduplicated: list[RetrievedDocument] = []
    for document in documents:
        key = retrieval_dedup_key(document)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(document)
    return deduplicated


def retrieval_dedup_key(document: RetrievedDocument) -> str:
    metadata = document.metadata or {}
    chunk_key = metadata.get("chunk_key")
    if chunk_key:
        return f"chunk:{chunk_key}"
    document_id = metadata.get("id")
    if document_id:
        return f"id:{document_id}"
    return f"fallback:{document.url or ''}:{document.title or ''}:{document.content[:80]}"


def clamp_vector_score(score: float | None) -> float | None:
    """Normalize vector scores for future merge/rerank work.

    Parallel keyword/vector reranking is intentionally deferred. This helper
    documents the expected vector score range without changing keyword scores.
    """

    if score is None:
        return None
    return max(0.0, min(float(score), 1.0))


def build_hybrid_result_trace_metadata(
    *,
    query: str,
    strategy: str,
    keyword_result: RagRetrievalResult,
    vector_result: RagRetrievalResult | None,
    documents: list[RetrievedDocument],
    top_k: int,
    disease_code: str | None,
    source_key: str | None,
    issue_keys: list[str],
    fallback_used: bool,
    fallback_reason: str | None,
    latency_ms: float,
) -> dict[str, Any]:
    vector_trace = vector_result.trace_metadata if vector_result else {}
    return {
        "retriever_strategy": strategy,
        "query_preview": query[:500],
        "query_length": len(query),
        "top_k": top_k,
        "keyword_returned_count": len(keyword_result.documents),
        "vector_returned_count": len(vector_result.documents) if vector_result else 0,
        "merged_count": len(documents),
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "disease_code": disease_code,
        "source_key": source_key,
        "issue_keys": issue_keys,
        "selected_chunk_keys": [_document_trace_key(document) for document in documents],
        "latency_ms": latency_ms,
        "embedding_provider": vector_trace.get("embedding_provider"),
        "embedding_model": vector_trace.get("embedding_model"),
        "embedding_dimension": vector_trace.get("embedding_dimension"),
        "documents": [document.to_trace_metadata() for document in documents],
    }


async def _call_keyword_retriever(
    retriever: Any,
    *,
    query: str,
    disease_type: str | None,
    top_k: int,
    include_safety_disclaimer: bool,
) -> RagRetrievalResult:
    result = retriever.retrieve(
        query=query,
        disease_type=disease_type,
        top_k=top_k,
        include_safety_disclaimer=include_safety_disclaimer,
    )
    return await _maybe_await_result(result)


async def _call_vector_retriever(
    retriever: Any,
    *,
    query: str,
    disease_code: str | None,
    source_key: str | None,
    issue_keys: list[str],
    topic_tags: list[str],
    top_k: int,
    include_safety_disclaimer: bool,
) -> RagRetrievalResult:
    result = retriever.retrieve(
        query=query,
        disease_type=disease_code,
        disease_code=disease_code,
        source_key=source_key,
        issue_keys=issue_keys,
        topic_tags=topic_tags,
        top_k=top_k,
        include_safety_disclaimer=include_safety_disclaimer,
    )
    return await _maybe_await_result(result)


async def _maybe_await_result(result: Any) -> RagRetrievalResult:
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, RagRetrievalResult):
        raise TypeError("retriever must return RagRetrievalResult")
    return result


def _normalize_strategy(strategy: str) -> str:
    if strategy not in SUPPORTED_HYBRID_STRATEGIES:
        raise ValueError(f"unsupported hybrid RAG strategy: {strategy}")
    return strategy


def _document_trace_key(document: RetrievedDocument) -> str:
    metadata = document.metadata or {}
    return str(metadata.get("chunk_key") or metadata.get("id") or document.title or "")


def _result_fallback_reason(documents: list[RetrievedDocument]) -> str | None:
    if not documents:
        return "no_result"
    source_ids = {str(document.metadata.get("id")) for document in documents}
    if source_ids <= {"safety_disclaimer", "diet_caution"}:
        return "safety_disclaimer_only"
    return None


def _elapsed_ms(started_at: float) -> float:
    return round((time.monotonic() - started_at) * 1000, 3)
