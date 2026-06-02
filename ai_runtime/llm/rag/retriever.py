from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ai_runtime.llm.rag.keyword_retriever import retrieve_keyword_rag_contexts
from ai_runtime.llm.rag.rag_context_builder import build_reference_sources, build_reference_summary
from ai_runtime.llm.rag.source_trust import source_trust_level_for_metadata
from ai_runtime.llm.schemas import RetrievedContext


@dataclass(frozen=True)
class RetrievedDocument:
    """Retriever-neutral document shape.

    Keep raw document content inside the runtime only. Trace metadata should use
    `to_trace_metadata()` so future vector retrievers do not leak source text or
    sensitive user input to observability tools.
    """

    content: str
    title: str | None = None
    source_name: str | None = None
    url: str | None = None
    metadata: dict = field(default_factory=dict)
    score: float | None = None

    @classmethod
    def from_context(cls, context: RetrievedContext) -> RetrievedDocument:
        score = context.metadata.get("score")
        return cls(
            title=context.title,
            content=context.content,
            source_name=context.source_name,
            url=context.url,
            metadata=dict(context.metadata),
            score=float(score) if isinstance(score, int | float) else None,
        )

    def to_context(self) -> RetrievedContext:
        return RetrievedContext(
            title=self.title,
            content=self.content,
            source_name=self.source_name,
            url=self.url,
            metadata=self.metadata,
        )

    def to_trace_metadata(self) -> dict[str, object]:
        source_trust_level = self.metadata.get("source_trust_level") or source_trust_level_for_metadata(self.metadata)
        return {
            "id": self.metadata.get("id"),
            "title": self.title,
            "source_type": self.metadata.get("source_type") or self.source_name,
            "source_org": self.source_name,
            "source_trust_level": source_trust_level,
            "score": self.score,
            "status": self.metadata.get("status"),
            "match_reason": self.metadata.get("match_reason"),
        }


@dataclass(frozen=True)
class RagRetrievalResult:
    documents: list[RetrievedDocument] = field(default_factory=list)
    reference_sources: list[dict[str, object]] = field(default_factory=list)
    reference_summary: str | None = None
    strategy: str = "none"
    fallback_reason: str | None = None
    trace_metadata: dict[str, object] = field(default_factory=dict)

    @property
    def contexts(self) -> list[RetrievedContext]:
        return [document.to_context() for document in self.documents]


class RagRetriever(Protocol):
    def retrieve(
        self,
        *,
        query: str,
        disease_type: str | None = None,
        top_k: int = 3,
        include_safety_disclaimer: bool = False,
    ) -> RagRetrievalResult:
        """Return retrieval results without external provider side effects."""


class KeywordRagRetriever:
    """Adapter for the current local markdown keyword retriever.

    Future vector/pgvector retrievers should implement `RagRetriever` and return
    the same `RagRetrievalResult` shape with similarity scores and source trust
    metadata. Do not call embedding APIs or vector stores from this adapter.
    """

    strategy = "local_markdown_keyword_match"

    def retrieve(
        self,
        *,
        query: str,
        disease_type: str | None = None,
        top_k: int = 3,
        include_safety_disclaimer: bool = False,
    ) -> RagRetrievalResult:
        contexts = retrieve_keyword_rag_contexts(
            user_message=query,
            disease_type=disease_type,
            top_k=top_k,
            include_safety_disclaimer=include_safety_disclaimer,
        )
        documents = [RetrievedDocument.from_context(context) for context in contexts]
        reference_sources = build_reference_sources(contexts)
        reference_summary = build_reference_summary(contexts)
        fallback_reason = _fallback_reason(documents)
        return RagRetrievalResult(
            documents=documents,
            reference_sources=reference_sources,
            reference_summary=reference_summary,
            strategy=self.strategy,
            fallback_reason=fallback_reason,
            trace_metadata={
                "strategy": self.strategy,
                "document_count": len(documents),
                "document_ids": [source.get("id") for source in reference_sources],
                "source_types": [source.get("source_org") for source in reference_sources],
                "source_trust_levels": [source.get("source_trust_level") for source in reference_sources],
                "documents": [document.to_trace_metadata() for document in documents],
                "fallback": fallback_reason is not None,
                "fallback_reason": fallback_reason,
                "vector_rag": False,
                "embedding_search": False,
            },
        )


def get_default_rag_retriever() -> RagRetriever:
    return KeywordRagRetriever()


def disabled_rag_retrieval_result(reason: str = "rag_disabled") -> RagRetrievalResult:
    return RagRetrievalResult(
        strategy="disabled",
        fallback_reason=reason,
        trace_metadata={
            "enabled": False,
            "reason": reason,
            "document_count": 0,
            "document_ids": [],
            "source_types": [],
            "source_trust_levels": [],
        },
    )


def _fallback_reason(documents: list[RetrievedDocument]) -> str | None:
    if not documents:
        return "no_result"
    source_ids = {str(document.metadata.get("id")) for document in documents}
    if source_ids == {"safety_disclaimer"}:
        return "safety_disclaimer_only"
    return None
