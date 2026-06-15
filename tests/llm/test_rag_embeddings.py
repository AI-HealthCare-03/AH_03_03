from types import SimpleNamespace

import pytest

from ai_runtime.llm.rag.embeddings import (
    MockEmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
)
from app.core.config import Config


def test_mock_embedding_provider_returns_same_vector_for_same_text() -> None:
    provider = MockEmbeddingProvider(dimension=16)

    first = provider.embed_text("고혈압 저염 식단")
    second = provider.embed_text("고혈압 저염 식단")

    assert first == second
    assert len(first) == 16


def test_mock_embedding_provider_returns_different_vector_for_different_text() -> None:
    provider = MockEmbeddingProvider(dimension=16)

    first = provider.embed_text("고혈압 저염 식단")
    second = provider.embed_text("당뇨 당류 식단")

    assert first != second
    assert len(second) == 16


def test_mock_embedding_provider_uses_configurable_dimension() -> None:
    provider = MockEmbeddingProvider(dimension=7)

    assert provider.dimension == 7
    assert len(provider.embed_text("짧은 문장")) == 7


def test_mock_embedding_provider_handles_empty_text() -> None:
    provider = MockEmbeddingProvider(dimension=8)

    vector = provider.embed_text("")

    assert len(vector) == 8
    assert all(isinstance(value, float) for value in vector)


def test_mock_embedding_provider_batch_preserves_order() -> None:
    provider = MockEmbeddingProvider(dimension=8)
    texts = ["첫 번째", "두 번째", "첫 번째"]

    vectors = provider.embed_texts(texts)

    assert vectors[0] == provider.embed_text("첫 번째")
    assert vectors[1] == provider.embed_text("두 번째")
    assert vectors[2] == provider.embed_text("첫 번째")
    assert vectors[0] != vectors[1]


def test_embedding_provider_factory_returns_none_when_disabled() -> None:
    settings = SimpleNamespace(
        RAG_EMBEDDING_ENABLED=False,
        RAG_EMBEDDING_PROVIDER="mock",
        RAG_EMBEDDING_MODEL="mock-model",
        RAG_EMBEDDING_DIMENSION=16,
    )

    assert get_embedding_provider(settings) is None


def test_embedding_provider_factory_returns_mock_provider_when_enabled() -> None:
    settings = SimpleNamespace(
        RAG_EMBEDDING_ENABLED=True,
        RAG_EMBEDDING_PROVIDER="mock",
        RAG_EMBEDDING_MODEL="mock-model",
        RAG_EMBEDDING_DIMENSION=12,
    )

    provider = get_embedding_provider(settings)

    assert isinstance(provider, MockEmbeddingProvider)
    assert provider.provider_name == "mock"
    assert provider.model_name == "mock-model"
    assert provider.dimension == 12


def test_embedding_provider_factory_openai_is_skeleton_without_api_call() -> None:
    settings = SimpleNamespace(
        RAG_EMBEDDING_ENABLED=True,
        RAG_EMBEDDING_PROVIDER="openai",
        RAG_EMBEDDING_MODEL="text-embedding-3-small",
        RAG_EMBEDDING_DIMENSION=1536,
    )

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_embedding_provider(settings)


def test_openai_embedding_provider_uses_fake_client_and_preserves_batch_order() -> None:
    client = FakeOpenAIEmbeddingClient(
        embeddings=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
    )
    provider = OpenAIEmbeddingProvider(model_name="text-embedding-3-small", dimension=3, client=client)

    vectors = provider.embed_texts(["첫 번째", "두 번째", ""])

    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    assert client.calls == [
        {
            "model": "text-embedding-3-small",
            "input": ["첫 번째", "두 번째", " "],
        }
    ]


def test_openai_embedding_provider_embed_text_returns_first_vector() -> None:
    client = FakeOpenAIEmbeddingClient(embeddings=[[0.1, 0.2]])
    provider = OpenAIEmbeddingProvider(model_name="text-embedding-3-small", dimension=2, client=client)

    assert provider.embed_text("단일 문장") == [0.1, 0.2]


def test_openai_embedding_provider_reorders_by_response_index() -> None:
    client = FakeOpenAIEmbeddingClient(
        entries=[
            {"index": 1, "embedding": [0.4, 0.5]},
            {"index": 0, "embedding": [0.1, 0.2]},
        ]
    )
    provider = OpenAIEmbeddingProvider(model_name="text-embedding-3-small", dimension=2, client=client)

    assert provider.embed_texts(["첫 번째", "두 번째"]) == [[0.1, 0.2], [0.4, 0.5]]


def test_openai_embedding_provider_rejects_dimension_mismatch() -> None:
    client = FakeOpenAIEmbeddingClient(embeddings=[[0.1, 0.2]])
    provider = OpenAIEmbeddingProvider(model_name="text-embedding-3-small", dimension=3, client=client)

    with pytest.raises(RuntimeError, match="dimension mismatch"):
        provider.embed_text("차원 불일치")


def test_embedding_provider_factory_returns_openai_provider_when_key_is_configured() -> None:
    settings = SimpleNamespace(
        RAG_EMBEDDING_ENABLED=True,
        RAG_EMBEDDING_PROVIDER="openai",
        RAG_EMBEDDING_MODEL="text-embedding-3-small",
        RAG_EMBEDDING_DIMENSION=1536,
        OPENAI_API_KEY="test-key",
    )

    provider = get_embedding_provider(settings)

    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert provider.provider_name == "openai"
    assert provider.model_name == "text-embedding-3-small"
    assert provider.dimension == 1536


def test_embedding_provider_factory_rejects_unknown_provider() -> None:
    settings = SimpleNamespace(
        RAG_EMBEDDING_ENABLED=True,
        RAG_EMBEDDING_PROVIDER="unknown",
        RAG_EMBEDDING_MODEL="mock-model",
        RAG_EMBEDDING_DIMENSION=16,
    )

    with pytest.raises(ValueError):
        get_embedding_provider(settings)


def test_config_embedding_defaults_are_safe() -> None:
    settings = Config(_env_file=None)

    assert settings.RAG_EMBEDDING_ENABLED is False
    assert settings.RAG_EMBEDDING_PROVIDER == "disabled"
    assert settings.RAG_EMBEDDING_MODEL == "text-embedding-3-small"
    assert settings.RAG_EMBEDDING_DIMENSION == 1536
    assert settings.RAG_EMBEDDING_BATCH_SIZE == 64
    assert get_embedding_provider(settings) is None


class FakeOpenAIEmbeddingClient:
    def __init__(self, *, embeddings: list[list[float]] | None = None, entries: list[dict] | None = None) -> None:
        self.embeddings = embeddings or []
        self.entries = entries
        self.calls: list[dict] = []
        self.embeddings_api = self

    @property
    def embeddings(self):
        return self.embeddings_api

    @embeddings.setter
    def embeddings(self, value):
        self._embeddings = value

    def create(self, *, model: str, input: list[str]):
        self.calls.append({"model": model, "input": input})
        data = self.entries
        if data is None:
            data = [{"index": index, "embedding": embedding} for index, embedding in enumerate(self._embeddings)]
        return {"data": data}
