"""
ai_runtime/ocr/checkup/eval/debug_single.py

단일 샘플 OCR 결과 vs 정답 비교 디버그 스크립트

실행:
    python -m ai.ocr.checkup.eval.debug_single
"""

import asyncio
import json
from pathlib import Path

from ai.ocr.checkup.extractor import run_ocr_on_pdf

BASE_DIR = Path(__file__).parent
PDFS_DIR = BASE_DIR / "pdfs"
GT_PATH  = BASE_DIR / "ground_truth_10000.json"

# 확인할 샘플 수 (늘려도 됨)
DEBUG_COUNT = 3


async def debug():
    with open(GT_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)

    subjects = list(gt_data["subjects"].items())[1:305]

    for file_key, label in [("pdf", "checkup"), ("health_check_pdf", "health_check")]:
        print(f"\n{'=' * 60}")
        print(f"  [{label}] 디버그")
        print(f"{'=' * 60}")

        for name, info in subjects:
            gt = info["ground_truth"]
            pdf_name = info["files"].get(file_key)
            if not pdf_name:
                print(f"  {name}: {file_key} 키 없음")
                continue

            pdf_path = PDFS_DIR / pdf_name
            if not pdf_path.exists():
                print(f"  {name}: 파일 없음 ({pdf_name})")
                continue

            print(f"\n  ── {name} ({pdf_name}) ──")

            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            data, low_conf, raw, status = await run_ocr_on_pdf(pdf_bytes, filename=pdf_name)
            ocr = data.model_dump()

            correct = 0
            total = 0

            for field, gt_val in gt.items():
                ocr_val = ocr.get(field)
                gt_str = str(gt_val).strip().replace(" ", "") if gt_val is not None else None

                if gt_str == "비해당":
                    total += 1
                    match = str(ocr_val).strip() == "비해당" if ocr_val else False
                    icon = "✅" if match else "❌"
                    if match:
                        correct += 1
                    print(f"    {icon} {field:20s} GT={gt_val!r:12} OCR={ocr_val!r}")
                    continue

                total += 1
                try:
                    match = float(gt_str) == float(str(ocr_val).strip()) if ocr_val is not None else False
                except (ValueError, TypeError):
                    match = gt_str == str(ocr_val).strip().replace(" ", "") if ocr_val is not None else False

                icon = "✅" if match else "❌"
                if match:
                    correct += 1
                print(f"    {icon} {field:20s} GT={gt_val!r:12} OCR={ocr_val!r}")

            acc = round(correct / total * 100, 1) if total else 0
            print(f"\n    정확도: {correct}/{total} = {acc}%")
            if low_conf:
                print(f"    저신뢰도 필드: {low_conf}")


if __name__ == "__main__":
    asyncio.run(debug())
