from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from typing import Any, Protocol

DEFAULT_EMBEDDING_DIMENSION = 1536
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_TIMEOUT_SECONDS = 20.0
OPENAI_EMBEDDING_MAX_RETRIES = 2


class EmbeddingProvider(Protocol):
    provider_name: str
    model_name: str
    dimension: int

    def embed_text(self, text: str) -> list[float]: ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class MockEmbeddingProvider:
    provider_name = "mock"

    def __init__(
        self,
        *,
        model_name: str = "mock-deterministic-embedding",
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    ) -> None:
        self.model_name = model_name
        self.dimension = _validate_dimension(dimension)

    def embed_text(self, text: str) -> list[float]:
        normalized = _normalize_text(text)
        values: list[float] = []
        counter = 0
        while len(values) < self.dimension:
            digest = hashlib.sha256(f"{self.model_name}:{counter}:{normalized}".encode()).digest()
            values.extend(_bytes_to_unit_floats(digest))
            counter += 1
        return values[: self.dimension]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


class OpenAIEmbeddingProvider:
    provider_name = "openai"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        api_key: str | None = None,
        client: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.dimension = _validate_dimension(dimension)
        self._api_key = api_key
        self._client = client
        self._client_factory = client_factory
        if self._client is None and self._client_factory is None and not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is required when RAG_EMBEDDING_PROVIDER=openai.")

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self._client_instance().embeddings.create(
            model=self.model_name,
            input=[_normalize_text_for_openai(text) for text in texts],
        )
        embeddings = _extract_openai_embeddings(response)
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"OpenAI embedding response count {len(embeddings)} did not match input count {len(texts)}."
            )
        for index, embedding in enumerate(embeddings):
            if len(embedding) != self.dimension:
                raise RuntimeError(
                    f"OpenAI embedding dimension mismatch at index {index}: {len(embedding)} != {self.dimension}."
                )
        return embeddings

    def _client_instance(self) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory()
            return self._client
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is required when RAG_EMBEDDING_PROVIDER=openai.")
        self._client = _build_openai_embedding_client(api_key=self._api_key)
        return self._client


def get_embedding_provider(settings: Any) -> EmbeddingProvider | None:
    if not bool(getattr(settings, "RAG_EMBEDDING_ENABLED", False)):
        return None

    provider_name = str(getattr(settings, "RAG_EMBEDDING_PROVIDER", "disabled") or "disabled").strip().lower()
    dimension = int(getattr(settings, "RAG_EMBEDDING_DIMENSION", DEFAULT_EMBEDDING_DIMENSION))
    model_name = str(getattr(settings, "RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL) or DEFAULT_EMBEDDING_MODEL)

    if provider_name in {"", "disabled", "none", "off"}:
        return None
    if provider_name == "mock":
        return MockEmbeddingProvider(model_name=model_name, dimension=dimension)
    if provider_name == "openai":
        return OpenAIEmbeddingProvider(
            model_name=model_name,
            dimension=dimension,
            api_key=getattr(settings, "OPENAI_API_KEY", None),
        )

    raise ValueError(f"Unsupported RAG embedding provider: {provider_name}")


def _normalize_text(text: str) -> str:
    return (text or "").strip()


def _normalize_text_for_openai(text: str) -> str:
    normalized = _normalize_text(text)
    return normalized or " "


def _validate_dimension(dimension: int) -> int:
    if dimension <= 0:
        raise ValueError("Embedding dimension must be positive.")
    return dimension


def _bytes_to_unit_floats(payload: bytes) -> list[float]:
    values = []
    for index in range(0, len(payload), 4):
        chunk = payload[index : index + 4]
        if len(chunk) < 4:
            continue
        integer = int.from_bytes(chunk, byteorder="big", signed=False)
        values.append((integer / 0xFFFFFFFF) * 2.0 - 1.0)
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def _build_openai_embedding_client(*, api_key: str) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is not installed. Install it before using OpenAI embeddings.") from exc

    try:
        import httpx

        timeout = httpx.Timeout(OPENAI_EMBEDDING_TIMEOUT_SECONDS)
    except ImportError:
        timeout = OPENAI_EMBEDDING_TIMEOUT_SECONDS

    return OpenAI(api_key=api_key, timeout=timeout, max_retries=OPENAI_EMBEDDING_MAX_RETRIES)


def _extract_openai_embeddings(response: Any) -> list[list[float]]:
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")
    if not isinstance(data, list):
        raise RuntimeError("OpenAI embedding response did not include a data list.")

    ordered_entries = sorted(enumerate(data), key=lambda item: _entry_index(item[1], item[0]))
    return [_entry_embedding(entry) for _, entry in ordered_entries]


def _entry_index(entry: Any, fallback: int) -> int:
    index = getattr(entry, "index", None)
    if index is None and isinstance(entry, dict):
        index = entry.get("index")
    return int(index) if isinstance(index, int) else fallback


def _entry_embedding(entry: Any) -> list[float]:
    embedding = getattr(entry, "embedding", None)
    if embedding is None and isinstance(entry, dict):
        embedding = entry.get("embedding")
    if not isinstance(embedding, list):
        raise RuntimeError("OpenAI embedding response entry did not include an embedding list.")
    return [float(value) for value in embedding]
