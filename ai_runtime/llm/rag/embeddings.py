from __future__ import annotations

import hashlib
import math
from typing import Any, Protocol

DEFAULT_EMBEDDING_DIMENSION = 1536
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


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
    ) -> None:
        self.model_name = model_name
        self.dimension = _validate_dimension(dimension)

    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError("OpenAI embedding calls are disabled until the vector DB stage.")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("OpenAI embedding calls are disabled until the vector DB stage.")


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
        return OpenAIEmbeddingProvider(model_name=model_name, dimension=dimension)

    raise ValueError(f"Unsupported RAG embedding provider: {provider_name}")


def _normalize_text(text: str) -> str:
    return (text or "").strip()


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
