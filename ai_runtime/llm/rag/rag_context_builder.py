from __future__ import annotations

from ai_runtime.llm.rag.keyword_retriever import KeywordRagMatch
from ai_runtime.llm.schemas import RetrievedContext


def build_retrieved_contexts(matches: list[KeywordRagMatch]) -> list[RetrievedContext]:
    return [
        RetrievedContext(
            title=match.document.metadata.title,
            content=match.document.content,
            source_name=match.document.metadata.source_org,
            url=match.document.metadata.source_url,
            metadata=match.to_metadata(),
        )
        for match in matches
    ]


def build_retrieved_context_text(matches: list[KeywordRagMatch]) -> str:
    chunks = []
    for match in matches:
        metadata = match.document.metadata
        chunks.append(
            "\n".join(
                [
                    f"[{metadata.id}] {metadata.title}",
                    f"source_org: {metadata.source_org}",
                    f"status: {metadata.status}",
                    match.document.content.strip(),
                ]
            )
        )
    return "\n\n---\n\n".join(chunks)


def build_reference_sources(contexts: list[RetrievedContext]) -> list[dict[str, object]]:
    return [
        {
            "id": context.metadata.get("id"),
            "title": context.title,
            "source_org": context.source_name,
            "source_url": context.url,
            "year": context.metadata.get("year"),
            "status": context.metadata.get("status"),
            "source_type": context.metadata.get("source_type"),
            "source_trust_level": context.metadata.get("source_trust_level"),
        }
        for context in contexts
    ]


def build_reference_summary(contexts: list[RetrievedContext], limit: int = 3) -> str | None:
    sources = build_reference_sources(contexts[:limit])
    source_labels = [str(source["title"]) for source in sources if source.get("title")]
    if not source_labels:
        return None
    return f"참고 정보: {', '.join(source_labels)} 후보 문서를 함께 확인했습니다."
