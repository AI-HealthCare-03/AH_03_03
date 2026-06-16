from __future__ import annotations

import json

import pytest

from ai_runtime.llm.rag.embeddings import MockEmbeddingProvider
from ai_runtime.llm.rag.retriever import RagRetrievalResult, RetrievedDocument
from scripts.rag.query_vector_rag import _print_summary, parse_args, query_vector_rag


class FakeVectorRetriever:
    def __init__(self, result: RagRetrievalResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def retrieve(
        self,
        *,
        query: str,
        disease_code: str | None = None,
        source_key: str | None = None,
        issue_keys: list[str] | None = None,
        topic_tags: list[str] | None = None,
        top_k: int = 3,
    ) -> RagRetrievalResult:
        self.calls.append(
            {
                "query": query,
                "disease_code": disease_code,
                "source_key": source_key,
                "issue_keys": issue_keys,
                "topic_tags": topic_tags,
                "top_k": top_k,
            }
        )
        return self.result


def test_vector_query_cli_parser_handles_options() -> None:
    args = parse_args(
        [
            "--query",
            "당뇨 식단 주의사항",
            "--provider",
            "mock",
            "--top-k",
            "5",
            "--disease-code",
            "DM",
            "--source-key",
            "diabetes",
            "--issue-key",
            "sugar_high",
            "--issue-key",
            "fiber_support",
            "--topic-tag",
            "혈당",
            "--json",
        ]
    )

    assert args.query == "당뇨 식단 주의사항"
    assert args.provider == "mock"
    assert args.top_k == 5
    assert args.disease_code == "DM"
    assert args.source_key == "diabetes"
    assert args.issue_key == ["sugar_high", "fiber_support"]
    assert args.topic_tag == ["혈당"]
    assert args.json is True


@pytest.mark.asyncio
async def test_vector_query_summary_uses_preview_without_full_content() -> None:
    retriever = FakeVectorRetriever(
        RagRetrievalResult(
            documents=[
                RetrievedDocument(
                    content="당뇨 식단 주의사항 " * 40,
                    title="당뇨 식생활",
                    source_name="공식 출처",
                    url="https://example.test/dm",
                    metadata={
                        "chunk_key": "rag:diabetes:section:000:chunk:0000",
                        "document_key": "rag:diabetes:diabetes.md",
                        "source_key": "diabetes",
                        "disease_code": "DM",
                        "section_title": "식사요법",
                        "source_url": "https://example.test/dm",
                    },
                    score=0.82,
                )
            ],
            strategy="vector",
        )
    )

    summary = await query_vector_rag(
        query_text="당뇨 식단 주의사항",
        provider=MockEmbeddingProvider(model_name="mock-model", dimension=8),
        provider_name="mock",
        model_name="mock-model",
        dimension=8,
        top_k=3,
        disease_code="DM",
        source_key="diabetes",
        issue_keys=["sugar_high"],
        topic_tags=["혈당"],
        retriever=retriever,
    )

    assert retriever.calls == [
        {
            "query": "당뇨 식단 주의사항",
            "disease_code": "DM",
            "source_key": "diabetes",
            "issue_keys": ["sugar_high"],
            "topic_tags": ["혈당"],
            "top_k": 3,
        }
    ]
    assert summary.returned_count == 1
    assert summary.db_write_performed is False
    result = summary.results[0]
    assert result["rank"] == 1
    assert result["chunk_key"] == "rag:diabetes:section:000:chunk:0000"
    assert result["document_key"] == "rag:diabetes:diabetes.md"
    assert result["source_key"] == "diabetes"
    assert result["disease_code"] == "DM"
    assert result["score"] == 0.82
    assert result["source_url"] == "https://example.test/dm"
    assert "content_preview" in result
    assert "content" not in result


@pytest.mark.asyncio
async def test_vector_query_can_include_full_content_when_requested() -> None:
    retriever = FakeVectorRetriever(
        RagRetrievalResult(
            documents=[
                RetrievedDocument(
                    content="짧은 chunk",
                    title="문서",
                    metadata={"chunk_key": "chunk-1"},
                    score=0.7,
                )
            ]
        )
    )

    summary = await query_vector_rag(
        query_text="질문",
        provider=MockEmbeddingProvider(model_name="mock-model", dimension=8),
        provider_name="mock",
        model_name="mock-model",
        dimension=8,
        retriever=retriever,
        include_content=True,
    )

    assert summary.results[0]["content"] == "짧은 chunk"
    assert summary.results[0]["content_preview"] == "짧은 chunk"


@pytest.mark.asyncio
async def test_vector_query_disabled_provider_returns_safe_summary() -> None:
    summary = await query_vector_rag(
        query_text="혈압 식단",
        provider=None,
        provider_name="disabled",
        model_name=None,
        dimension=None,
    )

    assert summary.returned_count == 0
    assert summary.db_write_performed is False
    assert summary.fallback_reason == "embedding_disabled"
    assert summary.warnings == ["embedding provider disabled"]


@pytest.mark.asyncio
async def test_vector_query_json_output_omits_vector_api_key_and_full_content_by_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = RagRetrievalResult(
        documents=[
            RetrievedDocument(
                content="본문 전문은 JSON 기본 출력에 들어가면 안 됩니다. " * 20,
                title="문서",
                metadata={"chunk_key": "chunk-1"},
                score=0.5,
            )
        ]
    )
    query_summary = await async_query_for_test(summary)

    _print_summary(query_summary, as_json=True)
    output = capsys.readouterr().out
    payload = json.loads(output)
    serialized = json.dumps(payload, ensure_ascii=False)

    assert "embedding_vector" not in serialized
    assert "sk-" not in serialized
    assert "본문 전문은 JSON 기본 출력에 들어가면 안 됩니다. " * 20 not in serialized
    assert payload["db_write_performed"] is False
    assert "content" not in payload["results"][0]
    assert "content_preview" in payload["results"][0]


async def async_query_for_test(result: RagRetrievalResult):
    return await query_vector_rag(
        query_text="query",
        provider=MockEmbeddingProvider(model_name="mock-model", dimension=8),
        provider_name="mock",
        model_name="mock-model",
        dimension=8,
        retriever=FakeVectorRetriever(result),
    )
