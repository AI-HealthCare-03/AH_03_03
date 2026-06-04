"""
ai_worker/vision/scripts/build_drug_db.py

식약처 의약품 허가정보 API에서 약품명을 수집하고
OpenAI Embedding으로 벡터화하여 ChromaDB에 저장합니다.

실행 방법:
    python -m ai_worker.vision.scripts.build_drug_db

소요 시간: 약 10~20분 (43,000건 기준)
비용: OpenAI Embedding API 약 $0.02 (1회성)
"""

import asyncio
import logging
import os
import time
from pathlib import Path

import chromadb
import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DRUG_API_URL = "https://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService07/getDrugPrdtPrmsnInq07"
DRUG_API_KEY = os.getenv("DRUG_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ChromaDB 저장 경로
CHROMA_DIR = "ai_runtime/cv/data/drug_chroma_db"
COLLECTION_NAME = "drug_names"

# 배치 크기
FETCH_BATCH = 100  # API 한 번에 가져올 건수
EMBED_BATCH = 100  # 임베딩 한 번에 처리할 건수


async def fetch_all_drug_names() -> list[dict]:
    """식약처 API에서 전체 약품명 수집."""
    all_items = []
    page = 1

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 총 건수 확인
        r = await client.get(
            DRUG_API_URL,
            params={"serviceKey": DRUG_API_KEY, "type": "json", "numOfRows": 1, "pageNo": 1},
        )
        total = r.json()["body"]["totalCount"]
        logger.info("총 약품 수: %d건", total)

        while len(all_items) < total:
            r = await client.get(
                DRUG_API_URL,
                params={
                    "serviceKey": DRUG_API_KEY,
                    "type": "json",
                    "numOfRows": FETCH_BATCH,
                    "pageNo": page,
                },
            )
            items = r.json()["body"]["items"]
            if not items:
                break

            for item in items:
                name = item.get("ITEM_NAME", "").strip()
                seq = item.get("ITEM_SEQ", "").strip()
                if name:
                    all_items.append({"id": seq or f"drug_{len(all_items)}", "name": name})

            logger.info("수집 중: %d / %d", len(all_items), total)
            page += 1
            await asyncio.sleep(0.1)  # API 부하 방지

    logger.info("수집 완료: %d건", len(all_items))
    return all_items


async def embed_texts(texts: list[str], client: AsyncOpenAI) -> list[list[float]]:
    """OpenAI Embedding API로 텍스트 벡터화."""
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


async def build_db():
    """약품명 수집 → 임베딩 → ChromaDB 저장."""
    if not DRUG_API_KEY:
        logger.error("DRUG_API_KEY 환경변수가 없습니다.")
        return
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY 환경변수가 없습니다.")
        return

    # 1. 약품명 수집
    logger.info("=== 1단계: 약품명 수집 ===")
    drugs = await fetch_all_drug_names()

    # 2. ChromaDB 초기화
    logger.info("=== 2단계: ChromaDB 초기화 ===")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    # 기존 컬렉션 있으면 삭제 후 재생성
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        logger.info("기존 컬렉션 삭제")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # 코사인 유사도 사용
    )

    # 3. 임베딩 + 저장
    logger.info("=== 3단계: 임베딩 및 저장 ===")
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    total = len(drugs)
    saved = 0

    for i in range(0, total, EMBED_BATCH):
        batch = drugs[i : i + EMBED_BATCH]
        texts = [d["name"] for d in batch]
        ids = [d["id"] for d in batch]

        try:
            embeddings = await embed_texts(texts, openai_client)
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
            )
            saved += len(batch)
            logger.info("저장 중: %d / %d", saved, total)
        except Exception as e:
            logger.error("배치 처리 실패 (index=%d): %s", i, e)

        await asyncio.sleep(0.2)  # API 레이트 리밋 방지

    logger.info("=== 완료: %d건 저장 ===", saved)
    logger.info("저장 경로: %s", CHROMA_DIR)


if __name__ == "__main__":
    start = time.time()
    asyncio.run(build_db())
    elapsed = time.time() - start
    logger.info("총 소요 시간: %.1f초", elapsed)
