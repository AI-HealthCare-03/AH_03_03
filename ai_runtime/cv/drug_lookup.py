"""
ai_worker/vision/drug_lookup.py

ChromaDB 벡터 검색 기반 약품명 보정 모듈 (RAG 방식).
GPT Vision이 오인식한 약품명을 정식 약품명으로 보정합니다.

사전 준비:
    python -m ai_worker.vision.scripts.build_drug_db
"""

import logging

import chromadb
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

CHROMA_DIR = "ai_runtime/cv/data/drug_chroma_db"
COLLECTION_NAME = "drug_names"
EMBED_MODEL = "text-embedding-3-small"

# ChromaDB 클라이언트 (싱글톤)
_chroma_collection = None


def get_collection():
    """ChromaDB 컬렉션 반환 (싱글톤)."""
    global _chroma_collection
    if _chroma_collection is None:
        try:
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            _chroma_collection = client.get_collection(COLLECTION_NAME)
            logger.info("ChromaDB 컬렉션 로드 완료 | 약품 수: %d", _chroma_collection.count())
        except Exception as e:
            logger.warning("ChromaDB 로드 실패 (키워드 검색으로 fallback): %s", e)
            _chroma_collection = None
    return _chroma_collection


async def _embed_text(text: str, api_key: str) -> list[float] | None:
    """텍스트를 벡터로 변환."""
    try:
        client = AsyncOpenAI(api_key=api_key)
        response = await client.embeddings.create(
            model=EMBED_MODEL,
            input=text,
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error("임베딩 실패: %s", e)
        return None


async def lookup_drug_name(
    drug_name: str,
    api_key: str,
    min_similarity: float = 0.6,
) -> str:
    """
    GPT가 추출한 약품명을 ChromaDB 벡터 검색으로 정식 약품명으로 보정합니다.

    Args:
        drug_name: GPT가 추출한 약품명
        api_key: OpenAI API 키
        min_similarity: 최소 유사도 (이하면 원본 반환)

    Returns:
        보정된 정식 약품명 (보정 실패 시 원본 반환)
    """
    if not drug_name or not api_key:
        return drug_name

    collection = get_collection()
    if collection is None:
        return drug_name

    # 텍스트 → 벡터
    logger.info("임베딩 API 키 앞부분: %s", api_key[:10] if api_key else "없음")
    embedding = await _embed_text(drug_name, api_key)
    if embedding is None:
        return drug_name

    try:
        # 벡터 유사도 검색
        results = collection.query(
            query_embeddings=[embedding],
            n_results=3,
        )

        if not results["documents"] or not results["documents"][0]:
            return drug_name

        best_name = results["documents"][0][0]
        # ChromaDB 코사인 거리 → 유사도 변환 (거리 0 = 완전일치 = 유사도 1)
        distance = results["distances"][0][0]
        similarity = 1 - distance

        if similarity >= min_similarity:
            logger.info(
                "약품명 보정 | 원본=%s → 보정=%s (유사도=%.3f)",
                drug_name,
                best_name,
                similarity,
            )
            return best_name

        logger.info(
            "약품명 보정 실패 (유사도 미달) | 원본=%s, 유사도=%.3f",
            drug_name,
            similarity,
        )
        return drug_name

    except Exception as e:
        logger.error("벡터 검색 실패: %s", e)
        return drug_name


async def correct_drug_names(
    medications: list[dict],
    api_key: str,
) -> list[dict]:
    """
    약품 목록의 약품명을 일괄 보정합니다.

    Args:
        medications: GPT 추출 약품 목록
        api_key: OpenAI API 키

    Returns:
        약품명이 보정된 약품 목록
    """
    corrected = []
    for med in medications:
        drug_name = med.get("drug_name", "")
        if drug_name:
            corrected_name = await lookup_drug_name(drug_name, api_key)
            med = {**med, "drug_name": corrected_name}
        corrected.append(med)
    return corrected
