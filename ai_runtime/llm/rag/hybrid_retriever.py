from __future__ import annotations

import inspect
import time
from typing import Any

from ai_runtime.llm.rag.rag_context_builder import build_reference_sources, build_reference_summary
from ai_runtime.llm.rag.retriever import KeywordRagRetriever, RagRetrievalResult, RetrievedDocument
from ai_runtime.llm.rag.tracing import trace_hybrid_rag_retrieval

HYBRID_STRATEGY_KEYWORD_ONLY = "keyword_only"
HYBRID_STRATEGY_KEYWORD_FIRST_VECTOR_FALLBACK = "keyword_first_vector_fallback"
HYBRID_STRATEGY_HYBRID_PARALLEL = "hybrid_parallel"
HYBRID_STRATEGY_VECTOR_DISABLED = "vector_disabled"
DEFAULT_KEYWORD_WEIGHT = 0.5
DEFAULT_VECTOR_WEIGHT = 0.5
SUPPORTED_HYBRID_STRATEGIES = {
    HYBRID_STRATEGY_KEYWORD_ONLY,
    HYBRID_STRATEGY_KEYWORD_FIRST_VECTOR_FALLBACK,
    HYBRID_STRATEGY_HYBRID_PARALLEL,
    HYBRID_STRATEGY_VECTOR_DISABLED,
}


class HybridRagRetriever:
    """Strategy helper for keyword/vector retrieval.

    Strategies:
    - keyword_only: keyword retriever only.
    - keyword_first_vector_fallback: call vector only when keyword is insufficient.
    - hybrid_parallel: call keyword and vector, then deduplicate and rerank.
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
        keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
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
        elif normalized_strategy == HYBRID_STRATEGY_HYBRID_PARALLEL:
            if self.vector_retriever is None:
                vector_error = "vector_retriever_unavailable"
                documents = keyword_result.documents
                fallback_used = True
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
                    documents = merge_and_rerank_hybrid_documents(
                        keyword_documents=keyword_result.documents,
                        vector_documents=vector_result.documents,
                        top_k=top_k,
                        disease_code=effective_disease_code,
                        issue_keys=list(issue_keys or []),
                        keyword_weight=keyword_weight,
                        vector_weight=vector_weight,
                    )
                    if not vector_result.documents:
                        fallback_used = True
                        vector_error = vector_result.fallback_reason or "vector_no_result"
                except Exception as exc:  # pragma: no cover - exact exception depends on injected retriever
                    fallback_used = True
                    vector_error = f"vector_retrieval_failed:{type(exc).__name__}"
                    documents = keyword_result.documents
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
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
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
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
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


def merge_and_rerank_hybrid_documents(
    *,
    keyword_documents: list[RetrievedDocument],
    vector_documents: list[RetrievedDocument],
    top_k: int,
    disease_code: str | None = None,
    issue_keys: list[str] | tuple[str, ...] | None = None,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
) -> list[RetrievedDocument]:
    ranked: dict[str, tuple[RetrievedDocument, float, int]] = {}
    order = 0
    for document in keyword_documents:
        key = retrieval_dedup_key(document)
        score = _hybrid_document_score(
            document,
            keyword_score=document.score,
            vector_score=None,
            disease_code=disease_code,
            issue_keys=list(issue_keys or []),
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
        )
        ranked[key] = (document, score, order)
        order += 1
    for document in vector_documents:
        key = retrieval_dedup_key(document)
        existing = ranked.get(key)
        keyword_score = existing[0].score if existing else None
        score = _hybrid_document_score(
            document,
            keyword_score=keyword_score,
            vector_score=document.score,
            disease_code=disease_code,
            issue_keys=list(issue_keys or []),
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
        )
        if existing is None or score > existing[1]:
            ranked[key] = (document if existing is None else existing[0], score, existing[2] if existing else order)
        order += 1

    sorted_documents = sorted(ranked.values(), key=lambda item: (-item[1], item[2]))
    return [_with_hybrid_score(document, score) for document, score, _ in sorted_documents[: max(top_k, 0)]]


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


def normalize_keyword_score(score: float | int | None) -> float:
    if score is None:
        return 0.0
    numeric = float(score)
    if numeric > 1:
        numeric = numeric / 100
    return max(0.0, min(numeric, 1.0))


def _hybrid_document_score(
    document: RetrievedDocument,
    *,
    keyword_score: float | int | None,
    vector_score: float | int | None,
    disease_code: str | None,
    issue_keys: list[str],
    keyword_weight: float,
    vector_weight: float,
) -> float:
    score = normalize_keyword_score(keyword_score) * keyword_weight
    score += (clamp_vector_score(float(vector_score)) or 0.0) * vector_weight if vector_score is not None else 0.0
    score += _review_status_boost(document)
    score += _disease_code_boost(document, disease_code)
    score += _issue_key_boost(document, issue_keys)
    return round(score, 6)


def _review_status_boost(document: RetrievedDocument) -> float:
    metadata = document.metadata or {}
    status = str(metadata.get("review_status") or metadata.get("status") or "").lower()
    if status in {"reviewed", "approved"}:
        return 0.08
    if status == "reference":
        return 0.04
    return 0.0


def _disease_code_boost(document: RetrievedDocument, disease_code: str | None) -> float:
    if not disease_code:
        return 0.0
    metadata = document.metadata or {}
    return 0.05 if str(metadata.get("disease_code") or "").upper() == str(disease_code).upper() else 0.0


def _issue_key_boost(document: RetrievedDocument, issue_keys: list[str]) -> float:
    if not issue_keys:
        return 0.0
    metadata = document.metadata or {}
    document_issue_keys = {str(value) for value in metadata.get("issue_keys") or []}
    return 0.05 if document_issue_keys.intersection(map(str, issue_keys)) else 0.0


def _with_hybrid_score(document: RetrievedDocument, hybrid_score: float) -> RetrievedDocument:
    return RetrievedDocument(
        content=document.content,
        title=document.title,
        source_name=document.source_name,
        url=document.url,
        metadata={
            **(document.metadata or {}),
            "retriever_strategy": HYBRID_STRATEGY_HYBRID_PARALLEL,
            "hybrid_score": hybrid_score,
        },
        score=document.score,
    )


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
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
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
        "final_count": len(documents),
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "keyword_weight": keyword_weight,
        "vector_weight": vector_weight,
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
