"""
ai_worker/vision/ocr/eval/reeval_medication.py

기존 평가 결과 JSON을 사용해 정확도 기준을 바꿔 재평가합니다.
API 재호출 없음 → 비용 0원

실행 방법:
    python -m ai_worker.vision.ocr.eval.reeval_medication
"""

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"

# 가장 최근 medication_eval.json 자동 선택
result_files = sorted(RESULTS_DIR.glob("*_medication_eval.json"))
if not result_files:
    print("medication_eval.json 파일이 없습니다.")
    exit()
RESULT_PATH = result_files[-1]
print(f"사용 파일: {RESULT_PATH.name}\n")


def normalize(name: str) -> str:
    """약품명 정규화."""
    name = name.lower().strip()
    name = re.sub(r'\[.*?\]', '', name)          # 대괄호 내용 제거
    name = re.sub(r'\(.*?\)', '', name)          # 괄호 내용 제거
    name = re.sub(r'[0-9]+\.?[0-9]*\s*(mg|ml|g|mcg|㎍|㎎|mL)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[\u4e00-\u9fff]', '', name)  # 한자 제거
    name = re.sub(r'\s+', '', name)              # 공백 제거
    return name.strip()


def levenshtein(s1: str, s2: str) -> int:
    """편집 거리 계산."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for c1 in s1:
        curr = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(0 if c1==c2 else 1)))
        prev = curr
    return prev[-1]


def is_match(gt_name: str, ocr_name: str, max_edit_dist: int = 2) -> bool:
    """
    약품명 유사 일치 여부.
    1. 정규화 후 완전 일치
    2. 정규화 후 포함 관계
    3. 편집 거리 max_edit_dist 이하
    """
    gt = normalize(gt_name)
    ocr = normalize(ocr_name)

    if not gt or not ocr:
        return False

    # 완전 일치
    if gt == ocr:
        return True

    # 포함 관계
    if gt in ocr or ocr in gt:
        return True

    # 편집 거리
    dist = levenshtein(gt, ocr)
    if dist <= max_edit_dist:
        return True

    return False


def reeval(max_edit_dist: int = 2):
    with open(RESULT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    results = data["상세결과"]
    accuracies = []

    print(f"{'=' * 60}")
    print(f"  재평가 | 편집거리 허용={max_edit_dist}, 정규화 강화")
    print(f"{'=' * 60}\n")

    for r in results:
        name = r["이름"]
        ocr_names = r["OCR_추출"]
        detail = r["상세"]

        gt_names = [k for k in detail.keys()]
        correct = 0
        new_detail = {}

        for gt_name in gt_names:
            matched = any(is_match(gt_name, ocr_name, max_edit_dist) for ocr_name in ocr_names)
            if matched:
                correct += 1
                new_detail[gt_name] = "✅ 정답"
            else:
                new_detail[gt_name] = f"❌ 오답 (OCR: {ocr_names})"

        acc = round(correct / len(gt_names), 4) if gt_names else 0.0
        accuracies.append(acc)
        status = "✅" if acc == 1.0 else f"⚠️ {round(acc*100,1)}%"
        print(f"  [{name}] {status} | {correct}/{len(gt_names)}")
        for gt, res in new_detail.items():
            if res.startswith("❌"):
                print(f"    {gt}: {res}")

    avg = round(sum(accuracies) / len(accuracies) * 100, 1) if accuracies else 0

    print(f"\n{'=' * 60}")
    print(f"  📊 재평가 완료")
    print(f"{'=' * 60}")
    print(f"  기존 정확도: {data['평균_정확도']}")
    print(f"  재평가 정확도: {avg}%")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    reeval(max_edit_dist=2)
