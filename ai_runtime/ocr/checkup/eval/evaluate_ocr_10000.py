"""
ai_runtime/ocr/checkup/eval/evaluate_ocr_10000.py

OCR 정확도 평가 스크립트 (1만건 PDF 전용, 서버 없이 직접 실행)
- checkup_*.pdf / health_check_*.pdf 두 타입 동시 평가

실행 방법:
    python -m ai_runtime.ocr.checkup.eval.evaluate_ocr_10000

폴더 구조:
    ai_runtime/ocr/checkup/eval/
    ├── pdfs/                    # checkup_00001.pdf ~ / health_check_00001.pdf ~
    ├── ground_truth_10000.json
    ├── results/
    └── evaluate_ocr_10000.py
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from ai_runtime.ocr.checkup.extractor import run_ocr_on_pdf

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PDFS_DIR = BASE_DIR / "pdfs"
GT_PATH = BASE_DIR / "ground_truth_10000.json"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 평가할 파일 타입 정의: (JSON 키, 레이블)
FILE_TYPES = [
    ("pdf",              "checkup"),
    ("health_check_pdf", "health_check"),
]


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
        return normalize(ocr_value) == "비해당"
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


# ── 단일 타입 평가 ────────────────────────────────────────────────────────────

async def evaluate_file_type(subjects: dict, file_key: str, label: str) -> dict:
    total_subjects = len(subjects)
    print(f"\n{'─' * 60}")
    print(f"  [{label}] 평가 시작 | 대상 {total_subjects}건")
    print(f"{'─' * 60}")

    accuracies = []
    all_results = []
    failed = []

    for i, (name, info) in enumerate(subjects.items(), 1):
        gt = info["ground_truth"]
        pdf_name = info["files"].get(file_key)

        if not pdf_name:
            failed.append({"이름": name, "이유": f"JSON에 {file_key} 키 없음"})
            continue

        pdf_path = PDFS_DIR / pdf_name
        if not pdf_path.exists():
            failed.append({"이름": name, "이유": "파일 없음"})
            if i % 500 == 0:
                print(f"  진행: {i}/{total_subjects}")
            continue

        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            data, low_conf, raw, status = await run_ocr_on_pdf(pdf_bytes, filename=pdf_name)
            ocr_fields = data.model_dump()

            acc, correct, total, detail = calc_accuracy(gt, ocr_fields)
            accuracies.append(acc)
            all_results.append({
                "이름": name,
                "파일": pdf_name,
                "정확도": round(acc * 100, 1),
                "정답수": correct,
                "전체수": total,
                "상세": detail,
            })

        except Exception as e:
            failed.append({"이름": name, "이유": str(e)})

        if i % 500 == 0:
            current_avg = round(sum(accuracies) / len(accuracies) * 100, 1) if accuracies else 0
            print(f"  진행: {i}/{total_subjects} | 현재 평균: {current_avg}%")

    avg = round(sum(accuracies) / len(accuracies) * 100, 1) if accuracies else 0

    print(f"\n  [{label}] 결과")
    print(f"  평가 건수: {len(accuracies)}건 / 전체 {total_subjects}건")
    print(f"  실패 건수: {len(failed)}건")
    print(f"  평균 정확도: {avg}%")

    return {
        "타입": label,
        "평가건수": len(accuracies),
        "실패건수": len(failed),
        "평균정확도": f"{avg}%",
        "실패목록": failed,
        "상세결과": all_results,
    }


# ── 메인 ──────────────────────────────────────────────────────────────────────

async def evaluate():
    with open(GT_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)

    subjects = gt_data["subjects"]

    print(f"\n{'=' * 60}")
    print(f"  PDF OCR 정확도 평가 | checkup vs health_check")
    print(f"{'=' * 60}")

    results = {}
    for file_key, label in FILE_TYPES:
        results[label] = await evaluate_file_type(subjects, file_key, label)

    # ── 최종 비교 요약 ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  📊 최종 비교")
    print(f"{'=' * 60}")
    for label, r in results.items():
        print(f"  [{label}] 평균 정확도: {r['평균정확도']} ({r['평가건수']}건)")
    print(f"{'=' * 60}\n")

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS_DIR / f"{timestamp}_ocr_eval_10000.json"

    summary = {
        "평가일시": datetime.now().isoformat(),
        "평가방식": "PaddleOCR 직접 호출 (서버 없음)",
        "비교결과": {label: r["평균정확도"] for label, r in results.items()},
        "상세": results,
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  💾 결과 저장: results/{result_path.name}\n")


if __name__ == "__main__":
    asyncio.run(evaluate())
