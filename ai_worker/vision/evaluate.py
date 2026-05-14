"""
ai_worker/vision/evaluate.py

GPT Vision 평가 스크립트 (방법 2 + 방법 4).

실행 방법:
    uv run python -m ai_worker.vision.evaluate

평가 흐름:
    1. images/ 폴더 사진을 GPT Vision으로 분석
    2. 결과를 터미널에 보여주고 사람이 맞다/틀리다 검수
    3. 유사도 기반으로 자동 채점
    4. results/ 폴더에 결과 저장
"""

import asyncio
import base64
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

from .client import AnalysisType, VisionClient
from .settings import VisionSettings

# ── 경로 설정 ─────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent
IMAGES_DIR  = BASE_DIR / "data" / "images"
RESULTS_DIR = BASE_DIR / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── 유사도 계산 ───────────────────────────────────────────────────────────────

def similarity(a: str, b: str) -> float:
    """
    두 문자열의 유사도를 0.0~1.0으로 반환합니다.
    완전 일치 → 1.0 / 한쪽이 다른 쪽을 포함 → 0.8 / 아예 다름 → 0.0
    """
    a = a.strip().lower().replace(" ", "")
    b = b.strip().lower().replace(" ", "")

    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.8

    # 공통 글자 비율
    common = sum(1 for c in a if c in b)
    return round(common / max(len(a), len(b), 1), 2)


def is_match(answer: str, gpt_result: str, threshold: float = 0.7) -> bool:
    return similarity(answer, gpt_result) >= threshold


# ── 평가 실행 ─────────────────────────────────────────────────────────────────

async def evaluate():
    settings = VisionSettings()
    client   = VisionClient(api_key=settings.openai_api_key, model=settings.openai_model)

    image_files = sorted([
        f for f in IMAGES_DIR.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ])

    if not image_files:
        print("❌ data/images/ 폴더에 이미지가 없습니다.")
        return

    print(f"\n{'='*60}")
    print(f"  GPT Vision 평가 시작 | 총 {len(image_files)}장")
    print(f"{'='*60}\n")

    results      = []
    total        = len(image_files)
    auto_correct = 0  # 유사도 기반 자동 정답
    human_correct= 0  # 사람 검수 정답
    human_wrong  = 0  # 사람 검수 오답
    skipped      = 0  # 분석 실패

    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{total}] {image_path.name} 분석 중...")

        # 이미지 읽기
        image_bytes = image_path.read_bytes()
        ext = image_path.suffix.lower()
        media_type  = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png",  ".webp": "image/webp"
        }.get(ext, "image/jpeg")

        # GPT Vision 분석
        try:
            raw = await client.analyze(
                analysis_type=AnalysisType.DIET,
                image_bytes=image_bytes,
                media_type=media_type,
            )
        except Exception as e:
            print(f"  ❌ 분석 실패: {e}\n")
            results.append({
                "image": image_path.name,
                "status": "분석실패",
                "gpt_result": None,
                "human_check": None,
                "score": 0,
            })
            skipped += 1
            continue

        # GPT 결과 출력
        foods = raw.get("foods", [])
        print(f"\n  📊 GPT Vision 결과:")
        print(f"  {'─'*40}")

        if not foods:
            print("  음식을 감지하지 못했습니다.")
        else:
            for f in foods:
                name      = f.get("name", "")
                category  = f.get("nutrient_category", "")
                cooking   = f.get("cooking_method", "")
                amount    = f.get("estimated_amount", "")
                conf      = f.get("confidence", 0)
                n         = f.get("nutrition") or {}

                print(f"  🍽️  음식명    : {name}")
                print(f"      영양카테고리: {category}")
                print(f"      조리법    : {cooking or '없음'}")
                print(f"      추정용량  : {amount or '직접입력필요'}")
                print(f"      인식신뢰도: {round(conf*100)}%")
                if n:
                    print(f"      영양성분  : 칼로리 {n.get('칼로리','?')}kcal | "
                          f"단백질 {n.get('단백질','?')}g | "
                          f"탄수화물 {n.get('탄수화물','?')}g | "
                          f"지방 {n.get('지방','?')}g | "
                          f"나트륨 {n.get('나트륨','?')}mg")
                print()

        # 사람 검수
        print(f"  ✅ 이 결과가 맞나요?")
        print(f"     1 → 정확함")
        print(f"     2 → 대략 맞음 (음식명은 맞지만 세부사항 다름)")
        print(f"     3 → 틀림")
        print(f"     s → 건너뜀")

        while True:
            choice = input("  선택: ").strip().lower()
            if choice in {"1", "2", "3", "s"}:
                break
            print("  1, 2, 3, s 중 하나를 입력해주세요.")

        # 정답 입력 (틀린 경우)
        correct_answer = None
        if choice == "3":
            correct_answer = input("  올바른 음식명을 입력해주세요: ").strip()

        # 점수 계산
        score_map = {"1": 1.0, "2": 0.5, "3": 0.0, "s": None}
        score = score_map[choice]

        if score == 1.0:
            human_correct += 1
        elif score == 0.5:
            human_correct += 0.5
            human_wrong   += 0.5
        elif score == 0.0:
            human_wrong += 1

        status_map = {"1": "정확", "2": "대략맞음", "3": "틀림", "s": "건너뜀"}

        results.append({
            "image":          image_path.name,
            "status":         status_map[choice],
            "gpt_result":     foods,
            "correct_answer": correct_answer,
            "score":          score,
        })

        print(f"  → {status_map[choice]} 기록됨\n")
        print(f"  {'─'*40}\n")

    # ── 최종 결과 출력 ────────────────────────────────────────────────────────

    evaluated = total - skipped
    accuracy  = round((human_correct / evaluated * 100), 1) if evaluated > 0 else 0

    print(f"\n{'='*60}")
    print(f"  📈 평가 완료")
    print(f"{'='*60}")
    print(f"  전체 이미지  : {total}장")
    print(f"  분석 성공    : {evaluated}장")
    print(f"  분석 실패    : {skipped}장")
    print(f"  정확         : {human_correct}장")
    print(f"  틀림         : {human_wrong}장")
    print(f"  정확도       : {accuracy}%")
    print(f"{'='*60}\n")

    # ── 결과 저장 ─────────────────────────────────────────────────────────────

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS_DIR / f"{timestamp}_result.json"

    summary = {
        "평가일시":    datetime.now().isoformat(),
        "모델":        settings.openai_model,
        "전체이미지":  total,
        "분석성공":    evaluated,
        "분석실패":    skipped,
        "정확":        human_correct,
        "틀림":        human_wrong,
        "정확도":      f"{accuracy}%",
        "상세결과":    results,
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  💾 결과 저장 완료: {result_path.name}")
    print(f"  📂 위치: data/results/\n")


if __name__ == "__main__":
    asyncio.run(evaluate())