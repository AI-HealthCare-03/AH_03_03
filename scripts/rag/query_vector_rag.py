from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tortoise import Tortoise  # noqa: E402

from ai_runtime.llm.rag.embeddings import EmbeddingProvider, get_embedding_provider  # noqa: E402
from ai_runtime.llm.rag.retriever import RagRetrievalResult  # noqa: E402
from ai_runtime.llm.rag.vector_retriever import VectorRagRetriever  # noqa: E402
from app.core import config  # noqa: E402
from app.core.db.databases import TORTOISE_ORM  # noqa: E402

DEFAULT_TOP_K = 3
DEFAULT_PREVIEW_CHARS = 180


class VectorRetrieverLike(Protocol):
    async def retrieve(
        self,
        *,
        query: str,
        disease_code: str | None = None,
        source_key: str | None = None,
        issue_keys: list[str] | None = None,
        topic_tags: list[str] | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> RagRetrievalResult: ...


@dataclass
class VectorRagQuerySummary:
    query_preview: str
    provider: str
    model: str | None
    dimension: int | None
    top_k: int
    returned_count: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fallback_reason: str | None = None
    db_write_performed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def query_vector_rag(
    *,
    query_text: str,
    provider: EmbeddingProvider | None,
    provider_name: str,
    model_name: str | None,
    dimension: int | None,
    top_k: int = DEFAULT_TOP_K,
    disease_code: str | None = None,
    source_key: str | None = None,
    issue_keys: list[str] | None = None,
    topic_tags: list[str] | None = None,
    include_content: bool = False,
    retriever: VectorRetrieverLike | None = None,
) -> VectorRagQuerySummary:
    summary = VectorRagQuerySummary(
        query_preview=_preview(query_text),
        provider=provider_name,
        model=model_name,
        dimension=dimension,
        top_k=max(0, top_k),
    )
    if provider is None:
        summary.warnings.append("embedding provider disabled")
        summary.fallback_reason = "embedding_disabled"
        return summary

    vector_retriever = retriever or VectorRagRetriever(embedding_provider=provider)
    result = await vector_retriever.retrieve(
        query=query_text,
        disease_code=disease_code,
        source_key=source_key,
        issue_keys=issue_keys or None,
        topic_tags=topic_tags or None,
        top_k=max(0, top_k),
    )
    summary.fallback_reason = result.fallback_reason
    summary.results = [
        _document_to_payload(rank=index + 1, document=document, include_content=include_content)
        for index, document in enumerate(result.documents)
    ]
    summary.returned_count = len(summary.results)
    return summary


def build_embedding_provider_from_options(
    *,
    provider_override: str | None = None,
    model_override: str | None = None,
    dimension_override: int | None = None,
) -> tuple[EmbeddingProvider | None, str, str | None, int | None]:
    provider_name = provider_override or config.RAG_EMBEDDING_PROVIDER
    enabled = config.RAG_EMBEDDING_ENABLED
    if provider_override is not None:
        enabled = provider_override.strip().lower() not in {"", "disabled", "none", "off"}
    settings = SimpleNamespace(
        RAG_EMBEDDING_ENABLED=enabled,
        RAG_EMBEDDING_PROVIDER=provider_name,
        RAG_EMBEDDING_MODEL=model_override or config.RAG_EMBEDDING_MODEL,
        RAG_EMBEDDING_DIMENSION=dimension_override or config.RAG_EMBEDDING_DIMENSION,
        OPENAI_API_KEY=config.OPENAI_API_KEY,
    )
    provider = get_embedding_provider(settings)
    if provider is None:
        return None, str(provider_name or "disabled"), None, None
    return provider, provider.provider_name, provider.model_name, provider.dimension


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only pgvector RAG query helper.")
    parser.add_argument("--query", required=True, help="Query text to embed and search.")
    parser.add_argument("--provider", choices=["mock", "openai", "disabled"], default=None, help="Provider override.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of vector search results.")
    parser.add_argument("--disease-code", default=None, help="Filter by rag_documents.disease_code.")
    parser.add_argument("--source-key", default=None, help="Filter by rag_documents.source_key.")
    parser.add_argument("--issue-key", action="append", default=[], help="Filter by chunk metadata issue_keys.")
    parser.add_argument("--topic-tag", action="append", default=[], help="Filter by chunk metadata topic_tags.")
    parser.add_argument("--dimension", type=int, default=None, help="Embedding dimension override.")
    parser.add_argument("--model", default=None, help="Embedding model override.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument(
        "--include-content",
        action="store_true",
        help="Include full chunk content. Default output only includes content_preview.",
    )
    return parser.parse_args(argv)


async def _run_cli(args: argparse.Namespace) -> VectorRagQuerySummary:
    try:
        provider, provider_name, model_name, dimension = build_embedding_provider_from_options(
            provider_override=args.provider,
            model_override=args.model,
            dimension_override=args.dimension,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should print a safe summary, not secrets.
        return VectorRagQuerySummary(
            query_preview=_preview(args.query),
            provider=args.provider or config.RAG_EMBEDDING_PROVIDER,
            model=args.model or config.RAG_EMBEDDING_MODEL,
            dimension=args.dimension or config.RAG_EMBEDDING_DIMENSION,
            top_k=max(0, args.top_k),
            warnings=[f"embedding provider unavailable: {type(exc).__name__}: {exc}"],
            fallback_reason="embedding_provider_unavailable",
        )

    if provider is None:
        return await query_vector_rag(
            query_text=args.query,
            provider=None,
            provider_name=provider_name,
            model_name=model_name,
            dimension=dimension,
            top_k=args.top_k,
        )

    await Tortoise.init(config=TORTOISE_ORM)
    try:
        return await query_vector_rag(
            query_text=args.query,
            provider=provider,
            provider_name=provider_name,
            model_name=model_name,
            dimension=dimension,
            top_k=args.top_k,
            disease_code=args.disease_code,
            source_key=args.source_key,
            issue_keys=args.issue_key,
            topic_tags=args.topic_tag,
            include_content=args.include_content,
        )
    finally:
        await Tortoise.close_connections()


def _document_to_payload(*, rank: int, document: Any, include_content: bool) -> dict[str, Any]:
    metadata = document.metadata if isinstance(document.metadata, dict) else {}
    content = str(document.content or "")
    payload = {
        "rank": rank,
        "chunk_key": metadata.get("chunk_key") or metadata.get("id"),
        "document_key": metadata.get("document_key"),
        "source_key": metadata.get("source_key"),
        "disease_code": metadata.get("disease_code"),
        "title": document.title,
        "section_title": metadata.get("section_title"),
        "score": document.score,
        "content_preview": _preview(content),
        "source_url": metadata.get("source_url") or document.url,
    }
    if include_content:
        payload["content"] = content
    return payload


def _preview(text: str, *, max_chars: int = DEFAULT_PREVIEW_CHARS) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "…"


def _print_summary(summary: VectorRagQuerySummary, *, as_json: bool) -> None:
    payload = summary.to_dict()
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return
    print("RAG vector query")
    print(f"- query_preview: {summary.query_preview}")
    print(f"- provider: {summary.provider}")
    print(f"- model: {summary.model}")
    print(f"- dimension: {summary.dimension}")
    print(f"- top_k: {summary.top_k}")
    print(f"- returned_count: {summary.returned_count}")
    if summary.fallback_reason:
        print(f"- fallback_reason: {summary.fallback_reason}")
    if summary.warnings:
        print("- warnings:")
        for warning in summary.warnings:
            print(f"  - {warning}")
    for result in summary.results:
        print(f"{result['rank']}. {result.get('title') or result.get('chunk_key')} score={result.get('score')}")
        print(f"   chunk_key={result.get('chunk_key')} disease_code={result.get('disease_code')}")
        print(f"   preview={result.get('content_preview')}")


def main() -> None:
    args = parse_args()
    summary = asyncio.run(_run_cli(args))
    _print_summary(summary, as_json=args.json)


if __name__ == "__main__":
    main()
