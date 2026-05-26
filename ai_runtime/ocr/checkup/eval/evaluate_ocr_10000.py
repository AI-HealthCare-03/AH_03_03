"""
ai_worker/vision/ocr/eval/evaluate_ocr_10000.py

OCR 정확도 평가 스크립트 (1만건 PDF 전용, 서버 없이 직접 실행)

실행 방법:
    python -m ai_worker.vision.ocr.eval.evaluate_ocr_10000

폴더 구조:
    ai_worker/vision/ocr/eval/
    ├── pdfs/                    # PDF 파일 (checkup_00001.pdf ~ checkup_10000.pdf)
    ├── ground_truth_10000.json  # 1만건 정답지
    ├── results/                 # 결과 저장
    └── evaluate_ocr_10000.py
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from ai_worker.vision.ocr.extractor import run_ocr_on_pdf

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PDFS_DIR = BASE_DIR / "pdfs"
GT_PATH = BASE_DIR / "ground_truth_10000.json"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── 정확도 계산 ───────────────────────────────────────────────────────────────


def normalize(value) -> str | None:
    if value is None:
        return None
    return str(value).strip().replace(" ", "")


def is_match(gt_value, ocr_value) -> bool | None:
    gt_norm = normalize(gt_value)

    if gt_norm is None:
        return None

    if gt_norm == "비해당":
        ocr_norm = normalize(ocr_value)
        return ocr_norm == "비해당"

    if ocr_value is None:
        return False

    ocr_norm = normalize(ocr_value)

    try:
        return float(gt_norm) == float(ocr_norm)
    except (ValueError, TypeError):
        pass

    return gt_norm == ocr_norm


def calc_accuracy(gt: dict, ocr: dict) -> tuple[float, int, int, dict]:
    detail = {}
    correct = 0
    total = 0

    for field, gt_val in gt.items():
        ocr_val = ocr.get(field)
        result = is_match(gt_val, ocr_val)

        if result is None:
            detail[field] = f"제외 (GT={gt_val})"
            continue

        total += 1
        if result:
            correct += 1
            detail[field] = f"✅ (GT={gt_val}, OCR={ocr_val})"
        else:
            detail[field] = f"❌ (GT={gt_val}, OCR={ocr_val})"

    accuracy = round(correct / total, 4) if total > 0 else 0.0
    return accuracy, correct, total, detail


# ── 메인 평가 ─────────────────────────────────────────────────────────────────


async def evaluate():
    with open(GT_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)

    subjects = gt_data["subjects"]
    total_subjects = len(subjects)

    print(f"\n{'=' * 60}")
    print(f"  PDF OCR 정확도 평가 | 대상 {total_subjects}건 (직접 호출)")
    print(f"{'=' * 60}\n")

    pdf_accuracies = []
    all_results = []
    failed = []

    for i, (name, info) in enumerate(subjects.items(), 1):
        gt = info["ground_truth"]
        pdf_name = info["files"]["pdf"]
        pdf_path = PDFS_DIR / pdf_name

        if not pdf_path.exists():
            failed.append({"이름": name, "이유": "파일 없음"})
            if i % 500 == 0:
                print(f"  진행: {i}/{total_subjects}")
            continue

        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            data, low_conf, raw, status = await run_ocr_on_pdf(pdf_bytes)
            ocr_fields = data.model_dump()

            acc, correct, total, detail = calc_accuracy(gt, ocr_fields)
            pdf_accuracies.append(acc)
            all_results.append(
                {
                    "이름": name,
                    "파일": pdf_name,
                    "정확도": round(acc * 100, 1),
                    "정답수": correct,
                    "전체수": total,
                    "상세": detail,
                }
            )

        except Exception as e:
            failed.append({"이름": name, "이유": str(e)})

        if i % 500 == 0:
            current_avg = round(sum(pdf_accuracies) / len(pdf_accuracies) * 100, 1) if pdf_accuracies else 0
            print(f"  진행: {i}/{total_subjects} | 현재 평균: {current_avg}%")

    # ── 최종 요약 ─────────────────────────────────────────────────────────────

    avg_pdf = round(sum(pdf_accuracies) / len(pdf_accuracies) * 100, 1) if pdf_accuracies else 0

    print(f"\n{'=' * 60}")
    print("  📊 평가 완료")
    print(f"{'=' * 60}")
    print(f"  평가 건수: {len(pdf_accuracies)}건 / 전체 {total_subjects}건")
    print(f"  실패 건수: {len(failed)}건")
    print(f"  PDF 평균 정확도: {avg_pdf}%")
    print(f"{'=' * 60}\n")

    # ── 결과 저장 ─────────────────────────────────────────────────────────────

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS_DIR / f"{timestamp}_ocr_eval_10000.json"

    summary = {
        "평가일시": datetime.now().isoformat(),
        "평가방식": "PaddleOCR 직접 호출 (서버 없음)",
        "평가건수": len(pdf_accuracies),
        "실패건수": len(failed),
        "평균_PDF": f"{avg_pdf}%",
        "실패목록": failed,
        "상세결과": all_results,
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  💾 결과 저장: results/{result_path.name}\n")


if __name__ == "__main__":
    asyncio.run(evaluate())
