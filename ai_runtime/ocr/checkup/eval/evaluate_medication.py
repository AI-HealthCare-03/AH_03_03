"""
ai_worker/vision/ocr/eval/evaluate_medication.py

약봉투 약품명 추출 정확도 평가 스크립트.

실행 방법:
    # 서버가 8001 포트에서 실행 중인 상태에서
    python -m ai_worker.vision.ocr.eval.evaluate_medication

폴더 구조:
    ai_worker/vision/ocr/eval/
    ├── images/medication/        # 약봉투 이미지
    ├── medication_ground_truth.json
    └── evaluate_medication.py
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "images" / "medication"
GT_PATH = BASE_DIR / "medication_ground_truth.json"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SERVER_URL = os.getenv("OCR_SERVER_URL", "http://localhost:8001")
MEDICATION_ENDPOINT = f"{SERVER_URL}/api/v1/cv/medication"
TIMEOUT = 60.0


def normalize_drug_name(name: str) -> str:
    """
    약품명 정규화.
    - 소문자 변환
    - 공백 제거
    - 용량 단위 제거 (mg, ml, g 등)
    - 괄호 내용 제거
    - 한자 제거
    """
    import re
    name = name.lower().strip()
    name = re.sub(r'\(.*?\)', '', name)          # 괄호 내용 제거
    name = re.sub(r'[0-9]+\.?[0-9]*\s*(mg|ml|g|mcg|㎍|㎎)', '', name)  # 용량 제거
    name = re.sub(r'[\u4e00-\u9fff]', '', name)  # 한자 제거
    name = re.sub(r'\s+', '', name)              # 공백 제거
    return name


def is_match(gt_name: str, ocr_name: str) -> bool:
    """약품명 유사 일치 여부 확인."""
    gt = normalize_drug_name(gt_name)
    ocr = normalize_drug_name(ocr_name)
    # 완전 일치 또는 포함 관계
    return gt == ocr or gt in ocr or ocr in gt


def calc_accuracy(gt_names: list[str], ocr_names: list[str]) -> tuple[float, dict]:
    """
    정답 약품명 목록과 OCR 추출 목록 비교.
    각 정답 약품명에 대해 OCR 결과에서 매칭되는 것이 있는지 확인.
    """
    detail = {}
    correct = 0
    total = len(gt_names)

    for gt_name in gt_names:
        matched = any(is_match(gt_name, ocr_name) for ocr_name in ocr_names)
        if matched:
            correct += 1
            detail[gt_name] = f"✅ 정답"
        else:
            detail[gt_name] = f"❌ 오답 (OCR 결과: {ocr_names})"

    accuracy = round(correct / total, 4) if total > 0 else 0.0
    return accuracy, detail


async def call_medication_api(client: httpx.AsyncClient, image_path: Path) -> list[str]:
    """약봉투 API 호출 후 약품명 목록 반환."""
    with open(image_path, "rb") as f:
        files = {"file": (image_path.name, f, "image/jpeg")}
        response = await client.post(MEDICATION_ENDPOINT, files=files, timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()
    medications = data.get("medications", [])
    return [m.get("drug_name", "") for m in medications if m.get("drug_name")]


async def evaluate():
    with open(GT_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)

    subjects = gt_data["subjects"]
    total = len(subjects)

    print(f"\n{'=' * 60}")
    print(f"  약봉투 약품명 추출 정확도 평가 | 대상 {total}건")
    print(f"{'=' * 60}\n")

    accuracies = []
    all_results = []
    failed = []

    async with httpx.AsyncClient() as client:
        for name, info in subjects.items():
            gt_names = info["ground_truth"]["drug_names"]
            image_path = IMAGES_DIR / info["files"]["image"]

            if not image_path.exists():
                print(f"  [{name}] 이미지 없음: {image_path.name}")
                failed.append({"이름": name, "이유": "파일 없음"})
                continue

            try:
                ocr_names = await call_medication_api(client, image_path)
                acc, detail = calc_accuracy(gt_names, ocr_names)
                accuracies.append(acc)
                all_results.append({
                    "이름": name,
                    "정확도": round(acc * 100, 1),
                    "정답수": sum(1 for v in detail.values() if v.startswith("✅")),
                    "전체수": len(gt_names),
                    "OCR_추출": ocr_names,
                    "상세": detail,
                })
                status = "✅" if acc == 1.0 else f"⚠️ {round(acc*100,1)}%"
                print(f"  [{name}] {status} | 정답: {sum(1 for v in detail.values() if v.startswith('✅'))}/{len(gt_names)}")
                for gt_name, result in detail.items():
                    print(f"    {gt_name}: {result}")
                print()

            except Exception as e:
                print(f"  [{name}] 실패: {e}")
                failed.append({"이름": name, "이유": str(e)})

    avg = round(sum(accuracies) / len(accuracies) * 100, 1) if accuracies else 0

    print(f"\n{'=' * 60}")
    print(f"  📊 평가 완료")
    print(f"{'=' * 60}")
    print(f"  평가 건수: {len(accuracies)}건 / 전체 {total}건")
    print(f"  실패 건수: {len(failed)}건")
    print(f"  평균 정확도: {avg}%")
    print(f"{'=' * 60}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS_DIR / f"{timestamp}_medication_eval.json"

    summary = {
        "평가일시": datetime.now().isoformat(),
        "평가건수": len(accuracies),
        "실패건수": len(failed),
        "평균_정확도": f"{avg}%",
        "실패목록": failed,
        "상세결과": all_results,
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  💾 결과 저장: results/{result_path.name}\n")


if __name__ == "__main__":
    asyncio.run(evaluate())
