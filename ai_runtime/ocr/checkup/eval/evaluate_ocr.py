"""
ai_worker/vision/ocr/eval/evaluate_ocr.py

OCR 정확도 평가 스크립트.

실행 방법:
    # 서버가 8001 포트에서 실행 중인 상태에서
    python -m ai_worker.vision.ocr.eval.evaluate_ocr

폴더 구조:
    ai_worker/vision/ocr/eval/
    ├── images/           # JPG 파일 (ground_truth의 jpg 파일명과 일치)
    ├── pdfs/             # PDF 파일 (ground_truth의 pdf 파일명과 일치)
    ├── ground_truth.json
    └── evaluate_ocr.py

결과:
    - 터미널 출력: 필드별 정확도, 평균 JPG / PDF 정확도
    - eval/results/ 폴더에 JSON 저장
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import httpx

# ── 경로 설정 ─────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "images"
PDFS_DIR = BASE_DIR / "pdfs"
GT_PATH = BASE_DIR / "ground_truth.json"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SERVER_URL = "http://localhost:8001"
IMG_ENDPOINT = f"{SERVER_URL}/api/v1/ocr/checkup"
PDF_ENDPOINT = f"{SERVER_URL}/api/v1/ocr/checkup/pdf"

TIMEOUT = 60.0  # 초


# ── 정확도 계산 ───────────────────────────────────────────────────────────────


def normalize(value) -> str | None:
    """비교를 위해 값을 문자열로 정규화. '비해당'은 None 처리."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "비해당":
        return None
    return str(value).strip().replace(" ", "")


def is_match(gt_value, ocr_value, tolerance: float = 0.05) -> bool | None:
    """
    정답과 OCR 결과를 비교.
    - '비해당' 필드는 평가 제외 (None 반환)
    - 숫자형은 tolerance 범위 내면 정답
    - 문자열은 완전 일치
    """
    gt_norm = normalize(gt_value)

    # 비해당 필드 → 평가 제외
    if gt_norm is None:
        return None

    # OCR 결과 없음 → 오답
    if ocr_value is None:
        return False

    ocr_norm = normalize(ocr_value)

    # 숫자 비교 (tolerance 허용)
    try:
        gt_num = float(gt_norm)
        ocr_num = float(ocr_norm)
        if gt_num == 0:
            return ocr_num == 0
        return abs(gt_num - ocr_num) / abs(gt_num) <= tolerance
    except (ValueError, TypeError):
        pass

    # 문자열 비교
    return gt_norm == ocr_norm


def calc_accuracy(gt: dict, ocr: dict) -> tuple[float, dict]:
    """
    정답 vs OCR 결과 비교.
    Returns:
        accuracy: 0.0~1.0
        detail: 필드별 결과 dict
    """
    detail = {}
    correct = 0
    total = 0

    for field, gt_val in gt.items():
        ocr_val = ocr.get(field)
        result = is_match(gt_val, ocr_val)

        if result is None:
            detail[field] = "비해당(제외)"
            continue

        total += 1
        if result:
            correct += 1
            detail[field] = f"✅ 정답 (GT={gt_val}, OCR={ocr_val})"
        else:
            detail[field] = f"❌ 오답 (GT={gt_val}, OCR={ocr_val})"

    accuracy = round(correct / total, 4) if total > 0 else 0.0
    return accuracy, detail


# ── API 호출 ──────────────────────────────────────────────────────────────────


async def call_image_ocr(client: httpx.AsyncClient, file_path: Path) -> dict:
    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "image/jpeg")}
        response = await client.post(IMG_ENDPOINT, files=files, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


async def call_pdf_ocr(client: httpx.AsyncClient, file_path: Path) -> dict:
    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "application/pdf")}
        response = await client.post(PDF_ENDPOINT, files=files, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


# ── 메인 평가 ─────────────────────────────────────────────────────────────────


async def evaluate():
    with open(GT_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)

    subjects = gt_data["subjects"]

    jpg_accuracies = []
    pdf_accuracies = []
    all_results = []

    print(f"\n{'=' * 60}")
    print(f"  OCR 정확도 평가 시작 | 대상 {len(subjects)}명")
    print(f"{'=' * 60}\n")

    async with httpx.AsyncClient() as client:
        for name, info in subjects.items():
            gt = info["ground_truth"]
            files = info["files"]

            print(f"▶ [{name}] 평가 중...")
            subject_result = {"이름": name, "jpg": None, "pdf": None}

            # ── JPG 평가 ──────────────────────────────────────────────────────
            jpg_path = IMAGES_DIR / files["jpg"]
            if jpg_path.exists():
                try:
                    raw = await call_image_ocr(client, jpg_path)
                    ocr_fields = raw.get("extracted_data", {})
                    acc, detail = calc_accuracy(gt, ocr_fields)
                    jpg_accuracies.append(acc)
                    subject_result["jpg"] = {"accuracy": acc, "detail": detail, "raw": raw}
                    print(f"  JPG 정확도: {round(acc * 100, 1)}%")
                    for field, result in detail.items():
                        print(f"    {field}: {result}")
                except Exception as e:
                    print(f"  JPG 실패: {e}")
            else:
                print(f"  JPG 파일 없음: {jpg_path.name}")

            print()

            # ── PDF 평가 ──────────────────────────────────────────────────────
            pdf_path = PDFS_DIR / files["pdf"]
            if pdf_path.exists():
                try:
                    raw = await call_pdf_ocr(client, pdf_path)
                    ocr_fields = raw.get("extracted_data", {})
                    acc, detail = calc_accuracy(gt, ocr_fields)
                    pdf_accuracies.append(acc)
                    subject_result["pdf"] = {"accuracy": acc, "detail": detail, "raw": raw}
                    print(f"  PDF 정확도: {round(acc * 100, 1)}%")
                    for field, result in detail.items():
                        print(f"    {field}: {result}")
                except Exception as e:
                    print(f"  PDF 실패: {e}")
            else:
                print(f"  PDF 파일 없음: {pdf_path.name}")

            print(f"\n  {'─' * 50}\n")
            all_results.append(subject_result)

    # ── 최종 요약 ─────────────────────────────────────────────────────────────

    avg_jpg = round(sum(jpg_accuracies) / len(jpg_accuracies) * 100, 1) if jpg_accuracies else 0
    avg_pdf = round(sum(pdf_accuracies) / len(pdf_accuracies) * 100, 1) if pdf_accuracies else 0

    print(f"\n{'=' * 60}")
    print("  📊 평가 완료")
    print(f"{'=' * 60}")
    print(f"  평균: JPG {avg_jpg}% / PDF {avg_pdf}%")
    print(f"{'=' * 60}\n")

    # ── 결과 저장 ─────────────────────────────────────────────────────────────

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS_DIR / f"{timestamp}_ocr_eval.json"

    summary = {
        "평가일시": datetime.now().isoformat(),
        "평균_JPG": f"{avg_jpg}%",
        "평균_PDF": f"{avg_pdf}%",
        "상세결과": all_results,
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  💾 결과 저장: results/{result_path.name}\n")


if __name__ == "__main__":
    asyncio.run(evaluate())
