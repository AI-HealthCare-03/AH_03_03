from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DEV_COMPOSE = ROOT_DIR / "infra" / "docker" / "docker-compose.dev.yml"
PROD_COMPOSE = ROOT_DIR / "infra" / "docker" / "docker-compose.prod.yml"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_dev_compose_passes_rag_embedding_env_to_app_containers() -> None:
    compose = _read(DEV_COMPOSE)

    assert compose.count("DIET_RECOMMENDATION_RAG_STRATEGY:") >= 2
    assert compose.count("RAG_EMBEDDING_ENABLED:") >= 2
    assert compose.count("RAG_EMBEDDING_PROVIDER:") >= 2
    assert compose.count("RAG_EMBEDDING_MODEL:") >= 2
    assert compose.count("RAG_EMBEDDING_DIMENSION:") >= 2
    assert compose.count("RAG_EMBEDDING_BATCH_SIZE:") >= 2


def test_prod_compose_passes_rag_embedding_env_to_app_containers() -> None:
    compose = _read(PROD_COMPOSE)

    assert compose.count("DIET_RECOMMENDATION_RAG_STRATEGY:") >= 2
    assert compose.count("RAG_EMBEDDING_ENABLED:") >= 2
    assert compose.count("RAG_EMBEDDING_PROVIDER:") >= 2
    assert compose.count("RAG_EMBEDDING_MODEL:") >= 2
    assert compose.count("RAG_EMBEDDING_DIMENSION:") >= 2
    assert compose.count("RAG_EMBEDDING_BATCH_SIZE:") >= 2


def test_compose_rag_defaults_stay_safe() -> None:
    for compose_path in (DEV_COMPOSE, PROD_COMPOSE):
        compose = _read(compose_path)

        assert "RAG_ENABLED: ${RAG_ENABLED:-true}" in compose
        assert ("DIET_RECOMMENDATION_RAG_STRATEGY: ${DIET_RECOMMENDATION_RAG_STRATEGY:-keyword_only}") in compose
        assert "RAG_EMBEDDING_ENABLED: ${RAG_EMBEDDING_ENABLED:-false}" in compose
        assert "RAG_EMBEDDING_PROVIDER: ${RAG_EMBEDDING_PROVIDER:-disabled}" in compose
