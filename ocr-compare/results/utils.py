"""
results/utils.py

OCR 비교 테스트 공통 유틸리티.
- 정답 데이터 로드
- 추출 컬럼 정의 로드
- 정확도 측정
- 결과 저장

모든 OCR 테스트에서 이 파일을 공통으로 사용합니다.
컬럼 변경 시 results/columns.json만 수정하면 됩니다.
"""

import json
import time
from pathlib import Path

# ── 경로 설정 ─────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent
IMAGES_DIR = RESULTS_DIR / "images" / "checkup"
COLUMNS_FILE = RESULTS_DIR / "columns.json"
GROUND_TRUTH_FILE = RESULTS_DIR / "ground_truth.json"


# ── 데이터 로드 ───────────────────────────────────────────────────────────────


def load_columns():
    """추출 대상 컬럼 정의를 로드합니다."""
    with open(COLUMNS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    fields = []
    for category in data["columns"].values():
        for field in category["fields"]:
            fields.append(field)
    return fields


def load_ground_truth():
    """정답 데이터를 로드합니다."""
    with open(GROUND_TRUTH_FILE, encoding="utf-8") as f:
        return json.load(f)


def get_all_keys():
    """추출 대상 컬럼 key 목록을 반환합니다."""
    columns = load_columns()
    return [col["key"] for col in columns]


def get_image_files(ext=None):
    """
    테스트 이미지 파일 목록을 반환합니다.

    Args:
        ext: 확장자 필터 ("jpg", "pdf", None=전체)

    Returns:
        파일 경로 목록
    """
    files = list(IMAGES_DIR.iterdir())
    if ext:
        files = [f for f in files if f.suffix.lower() == f".{ext}"]
    return sorted(files)


# ── 정확도 측정 ───────────────────────────────────────────────────────────────


def evaluate(extracted: dict, subject: str) -> dict:
    """
    추출 결과와 정답 데이터를 비교해 정확도를 측정합니다.

    Args:
        extracted: OCR이 추출한 결과 dict
        subject:   대상자 이름 (ground_truth.json의 key)

    Returns:
        {
            "correct":   정답 일치 수,
            "total":     전체 컬럼 수,
            "accuracy":  정확도 (%),
            "details":   컬럼별 상세 결과
        }
    """
    gt_data = load_ground_truth()
    gt = gt_data["subjects"][subject]["ground_truth"]
    keys = get_all_keys()

    correct = 0
    details = {}

    for key in keys:
        gt_val = gt.get(key)
        ext_val = extracted.get(key)

        if gt_val is None:
            # 정답 데이터 없음 → 스킵
            details[key] = {"gt": None, "extracted": ext_val, "match": None}
            continue

        # 숫자 비교 (±5% 오차 허용)
        if ext_val is not None:
            try:
                tolerance = abs(float(gt_val)) * 0.05
                match = abs(float(ext_val) - float(gt_val)) <= tolerance
            except (ValueError, TypeError):
                match = str(ext_val).strip() == str(gt_val).strip()
        else:
            match = False

        if match:
            correct += 1

        details[key] = {
            "gt": gt_val,
            "extracted": ext_val,
            "match": match,
        }

    total = len([k for k in keys if gt.get(k) is not None])
    accuracy = round(correct / total * 100, 1) if total > 0 else 0.0

    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "details": details,
    }


# ── 속도 측정 ─────────────────────────────────────────────────────────────────


class Timer:
    """OCR 처리 시간 측정용 타이머."""

    def __init__(self):
        self._start = None

    def start(self):
        self._start = time.time()

    def stop(self) -> float:
        """경과 시간을 ms 단위로 반환합니다."""
        if self._start is None:
            return 0.0
        elapsed = (time.time() - self._start) * 1000
        self._start = None
        return round(elapsed, 1)


# ── 결과 출력 ─────────────────────────────────────────────────────────────────


def print_result(ocr_name: str, subject: str, eval_result: dict, elapsed_ms: float):
    """테스트 결과를 콘솔에 출력합니다."""
    print(f"\n{'=' * 50}")
    print(f"  {ocr_name} | {subject}")
    print(f"{'=' * 50}")
    print(f"  정확도: {eval_result['correct']}/{eval_result['total']} ({eval_result['accuracy']}%)")
    print(f"  속도:   {elapsed_ms}ms")
    print(f"{'-' * 50}")

    for key, detail in eval_result["details"].items():
        if detail["match"] is None:
            status = "⬜ 정답없음"
        elif detail["match"]:
            status = "✅ 일치"
        else:
            status = "❌ 불일치"
        print(f"  {status} {key}: 정답={detail['gt']} / 추출={detail['extracted']}")

    print(f"{'=' * 50}\n")


def save_result(ocr_name: str, subject: str, file_type: str, eval_result: dict, elapsed_ms: float):
    """
    테스트 결과를 JSON 파일로 저장합니다.
    파일명: results/{ocr_name}_{subject}_{file_type}.json
    """
    output = {
        "ocr_engine": ocr_name,
        "subject": subject,
        "file_type": file_type,
        "accuracy": eval_result["accuracy"],
        "correct": eval_result["correct"],
        "total": eval_result["total"],
        "elapsed_ms": elapsed_ms,
        "details": eval_result["details"],
    }

    filename = RESULTS_DIR / f"{ocr_name}_{subject}_{file_type}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"결과 저장: {filename}")
