"""
GPT Vision 약봉투 인식 결과 검증 스크립트

ground_truth.json(검증1~43 약품명 정답)을 기준으로, 각 약봉투 이미지를 GPT Vision
(AnalysisType.MEDICATION_BAG)에 입력하고 추출된 약품명을 정답과 비교한다.

사용법:
    OPENAI_API_KEY=sk-... python evaluate_medication_vision.py \
        --ground-truth "ai_runtime/ocr/medication/eval/ground_truth.json" \
        --images-dir "ai_runtime/ocr/medication/eval/images" \
        --output "ai_runtime/ocr/medication/eval/medication_eval_result.csv"

연결 방식 (ai_runtime/cv/food/eval/evaluate_diet_vision.py와 동일한 골격):
- ai_runtime/cv/providers/gpt_vision.py 의 VisionClient.analyze(analysis_type, image_bytes, media_type)를
  AnalysisType.MEDICATION_BAG로 호출한다. 반환값의 medications[*].drug_name 목록을 추출 결과로 본다.
- VisionClient는 비동기이므로 본 스크립트도 asyncio로 동작한다.

정답 판정 방식 — 왜 "부분 일치"까지 정답으로 인정하는가:
- 약 봉투에 인쇄된 약품명에는 보통 용량(mg, g, mL)·제형(정/캡슐/시럽/산/액 등)·성분명 괄호
  표기가 섞여 있다. GPT가 핵심 약품명만 추출하거나 표기를 일부 다르게 정규화해도 "같은 약"을
  가리키는 경우가 많으므로, 이런 표기 차이까지 오답 처리하면 평가가 실제 품질을 왜곡한다.
- 단, 느슨한 매칭은 "다른 약인데 이름 일부가 겹쳐서 정답 처리"되는 위험과 정확히 트레이드오프
  관계다. 그래서 판정을 강한 기준 → 약한 기준 순서로 단계화하고, 각 단계를 결과에 그대로
  남겨서(match_type) "관대하게 정답 처리된 항목"을 사람이 다시 검토할 수 있게 한다.
"""

import argparse
import asyncio
import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

# 정규화 후에도 포함 관계로 보지 않을 최소 길이.
# "정"/"캡슐"처럼 제형 접미사만 겹쳐도 포함 매칭되는 것을 막기 위함이다.
MIN_CONTAINMENT_LENGTH = 3

# 포함 관계가 아니어도, 문자열 유사도가 이 값 이상이면 같은 약으로 인정한다.
# (예: "소론도정" vs "소론드정"처럼 OCR 오인식/받침 차이를 흡수)
# 너무 낮으면 "페니라민" vs "페니실린"처럼 실제로 다른 약까지 묶일 위험이 있고,
# 너무 높으면 포함 매칭과 차이가 없어진다 — 0.72는 1차 기준값이며 운영 결과를 보고 조정한다.
SIMILARITY_THRESHOLD = 0.72

from ai_runtime.cv.providers.gpt_vision import AnalysisType, VisionClient
from ai_runtime.cv.settings import VisionSettings

_settings = VisionSettings()
_client = VisionClient(api_key=_settings.openai_api_key, model=_settings.openai_model)

# 정규화 단계에서 제거할 제형/단위 접미사. 정답·예측 모두에 동일하게 적용해
# "암브로콜시럽" vs "암브로콜시럽[500mL]" 같은 표기 차이를 흡수한다.
_DOSAGE_FORM_SUFFIXES = (
    "정", "캡슐", "시럽", "산", "액", "연고", "크림", "과립", "환", "주", "패치",
)
_UNIT_PATTERN = re.compile(r"[\d.]+\s?(mg|g|ml|mL|밀리그램|밀리그람|밀리리터|%)", re.IGNORECASE)
_BRACKET_PATTERN = re.compile(r"[\(\)\[\]（）［］].*?[\)\]）］]")
_NON_KOREAN_ALNUM = re.compile(r"[^\w가-힣]")


def normalize_drug_name(name: str) -> str:
    """비교용 정규화.

    1) 괄호/대괄호 안 내용 제거 (성분명·용량 부연 설명 — "암브로콜시럽_(500mL)" → "암브로콜시럽")
    2) 숫자+단위 제거 ("250밀리그램", "0.25g", "20mg" 등 — 표기 위치가 달라도 흡수)
    3) 공백·특수문자 제거
    4) 제형 접미사 제거 ("정"/"캡슐"/"시럽" 등 — "암브로콜시럽" vs "암브로콜" 비교 가능하게)

    주의: 이 정규화는 '표기 차이'만 보정하기 위한 것이다. "페니라민정"을 "페니실린정"으로
    잘못 추출한 경우처럼 실제로 다른 약을 가리키면, 정규화해도 핵심 이름이 달라 여전히
    불일치로 남는다 — 의도적으로 그렇게 설계했다 (그래야 정답률이 의미를 가진다).
    """
    if not name:
        return ""
    s = _BRACKET_PATTERN.sub("", name)
    s = _UNIT_PATTERN.sub("", s)
    s = _NON_KOREAN_ALNUM.sub("", s)
    s = s.strip()
    for suffix in _DOSAGE_FORM_SUFFIXES:
        if s.endswith(suffix) and len(s) > len(suffix) + 1:
            s = s[: -len(suffix)]
            break
    return s


def best_similarity(target: str, predicted_normalized: list[str]) -> tuple[float, str]:
    """target과 가장 유사한 정규화된 predicted 문자열, 그 유사도를 반환한다."""
    best_ratio = 0.0
    best_name = ""
    for predicted in predicted_normalized:
        ratio = SequenceMatcher(None, target, predicted).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_name = predicted
    return best_ratio, best_name


def is_match(answer_name: str, predicted_names: list[str]) -> tuple[bool, str, float, str]:
    """
    정답 약품명과 GPT가 추출한 약품명 목록을 비교해 정답 여부를 판정한다.
    (정답 여부, 매칭 종류, 유사도, 가장 유사했던 추출명) 튜플을 반환한다.

    판정은 3단계로 이루어진다 (강한 기준 → 약한 기준):
    1) 정확 일치: 정규화 후 완전히 같으면 정답.
    2) 포함 매칭: 정규화된 정답/예측 중 한쪽이 다른 쪽을 포함하면 정답
       (단, MIN_CONTAINMENT_LENGTH 미만이면 "정"/"캡슐" 같은 공통 접미사만
       겹쳐도 정답 처리되므로 제외한다).
    3) 유사도 매칭: 1·2에서 못 잡은 경우, 편집거리 기반 유사도가
       SIMILARITY_THRESHOLD 이상이면 정답으로 인정한다 (OCR 오인식/받침 차이 흡수).

    세 기준 모두 결과의 match_type에 그대로 남기므로, "느슨한 기준으로 정답
    처리된 항목"만 따로 모아 사람이 재검토할 수 있다 — 평가의 관대함과
    투명성을 동시에 확보하기 위한 설계다.
    """
    target = normalize_drug_name(answer_name)
    if not target:
        return False, "정답없음", 0.0, ""

    predicted_normalized = [normalize_drug_name(p) for p in predicted_names]

    if target in predicted_normalized:
        return True, "정확일치", 1.0, ""

    if len(target) >= MIN_CONTAINMENT_LENGTH:
        for original, normalized in zip(predicted_names, predicted_normalized):
            if not normalized or len(normalized) < MIN_CONTAINMENT_LENGTH:
                continue
            if target in normalized or normalized in target:
                return True, "포함", 1.0, original

    ratio, best_normalized = best_similarity(target, predicted_normalized)
    if ratio >= SIMILARITY_THRESHOLD:
        idx = predicted_normalized.index(best_normalized)
        return True, "유사도", ratio, predicted_names[idx]

    return False, "불일치", ratio, (
        predicted_names[predicted_normalized.index(best_normalized)] if best_normalized else ""
    )


async def call_gpt_vision(path: Path) -> list[str]:
    """GPT Vision(MEDICATION_BAG 분석)으로 이미지에서 약품명 목록을 추출한다."""
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    image_bytes = path.read_bytes()

    result = await _client.analyze(
        analysis_type=AnalysisType.MEDICATION_BAG,
        image_bytes=image_bytes,
        media_type=media_type,
    )
    medications = result.get("medications") or []
    return [m.get("drug_name", "").strip() for m in medications if m.get("drug_name")]


async def evaluate(ground_truth_json: Path, images_dir: Path, output_csv: Path,
                    limit: int | None = None):
    with open(ground_truth_json, encoding="utf-8") as f:
        ground_truth = json.load(f)

    rows_out = []
    correct = 0
    total = 0
    image_total = 0

    image_keys = [k for k in ground_truth if not k.startswith("_")]
    if limit is not None:
        image_keys = image_keys[:limit]

    for image_key in image_keys:
        entry = ground_truth[image_key]
        answer_names = [m["drug_name"] for m in entry.get("medications", [])]
        if not answer_names:
            continue  # 정답이 비어있는 이미지(예: 식별 불가 사진)는 평가에서 제외

        image_path = images_dir / f"{image_key}.jpg"
        if not image_path.exists():
            print(f"[경고] 이미지 없음: {image_path}")
            continue

        image_total += 1
        try:
            predicted_names = await call_gpt_vision(image_path)
        except Exception as e:
            predicted_names = []
            print(f"[경고] GPT Vision 호출 실패: {image_path} ({e})")

        for answer in answer_names:
            total += 1
            is_correct, match_type, similarity, matched_by = is_match(answer, predicted_names)
            if is_correct:
                correct += 1

            rows_out.append({
                "image": image_key,
                "answer_name": answer,
                "predicted_names": " | ".join(predicted_names),
                "is_correct": "O" if is_correct else "X",
                "match_type": match_type,
                "similarity": f"{similarity:.2f}",
                "closest_predicted": matched_by,
            })

    fieldnames = ["image", "answer_name", "predicted_names",
                  "is_correct", "match_type", "similarity", "closest_predicted"]
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    accuracy = (correct / total * 100) if total else 0.0
    match_type_counts = {}
    for r in rows_out:
        if r["is_correct"] == "O":
            match_type_counts[r["match_type"]] = match_type_counts.get(r["match_type"], 0) + 1

    wrong_rows = [r for r in rows_out if r["is_correct"] == "X"]

    # 오답을 유사도 구간별로 집계한다.
    # 주의: 이 유사도는 SequenceMatcher 기반 "문자 중첩 비율"이지 "의미상 같은 약인가"가
    # 아니다. 같은 구간 안에도 성격이 다른 오답(근접 오인식 vs 우연한 문자 중첩)이
    # 섞일 수 있으므로, 구간별 건수는 "근접 오답이 대략 몇 건인가"를 보는 참고 지표로만
    # 쓰고, 각 구간의 실제 의미는 examples로 뽑은 사례를 사람이 직접 확인해 판단해야 한다.
    SIMILARITY_BUCKETS = [
        ("0.6 이상 (근접 오인식 가능성 높음)", 0.6, 1.01),
        ("0.4~0.6 (부분적으로 겹침)", 0.4, 0.6),
        ("0.2~0.4 (약하게 겹침)", 0.2, 0.4),
        ("0.2 미만 (사실상 무관)", 0.0, 0.2),
    ]
    EXAMPLES_PER_BUCKET = 3

    similarity_breakdown = []
    for label, low, high in SIMILARITY_BUCKETS:
        bucket_rows = [r for r in wrong_rows if low <= float(r["similarity"]) < high]
        similarity_breakdown.append({
            "range": label,
            "count": len(bucket_rows),
            "examples": [
                {
                    "image": r["image"],
                    "answer_name": r["answer_name"],
                    "closest_predicted": r["closest_predicted"],
                    "similarity": r["similarity"],
                }
                for r in bucket_rows[:EXAMPLES_PER_BUCKET]
            ],
        })

    summary = {
        "image_total": image_total,
        "drug_name_total": total,
        "correct": correct,
        "wrong": len(wrong_rows),
        "accuracy_percent": round(accuracy, 1),
        "correct_by_match_type": match_type_counts,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "wrong_similarity_breakdown": similarity_breakdown,
        "_note": (
            "correct_by_match_type 중 '포함'/'유사도'로 정답 처리된 항목은 "
            "느슨한 기준이 적용된 것이므로, wrong_rows와 함께 사람이 재검토하는 것을 권장합니다. "
            "wrong_similarity_breakdown의 구간 라벨은 대략적인 가이드일 뿐이며, "
            "examples를 직접 확인해 각 구간의 실제 오답 성격을 판단해야 합니다."
        ),
    }

    output_json = output_csv.with_suffix(".json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": rows_out}, f, ensure_ascii=False, indent=2)

    print(f"\n이미지 {image_total}건 / 약품명 {total}건 중 {correct}건 정답 (정답률 {accuracy:.1f}%)")
    print(f"매칭 종류별 정답: {match_type_counts}")
    print(f"상세 결과(CSV): {output_csv}")
    print(f"상세 결과(JSON): {output_json}")

    print(f"\n오답 {len(wrong_rows)}건의 유사도 구간별 분포 "
          f"(※ 구간 라벨은 참고용 가이드일 뿐, 실제 오답 성격은 examples를 직접 확인할 것):")
    for bucket in similarity_breakdown:
        print(f"  - {bucket['range']}: {bucket['count']}건")
        for ex in bucket["examples"]:
            print(f"      예) [{ex['image']}] 정답='{ex['answer_name']}' "
                  f"/ 가장 유사='{ex['closest_predicted']}' (유사도={ex['similarity']})")

    if wrong_rows:
        print(f"\n오답 {len(wrong_rows)}건:")
        for r in wrong_rows:
            print(f"  - [{r['image']}] 정답='{r['answer_name']}' / 추출=[{r['predicted_names']}]"
                  f" (가장 유사: '{r['closest_predicted']}', 유사도={r['similarity']})")


def main():
    parser = argparse.ArgumentParser(description="GPT Vision 약봉투 약품명 인식 결과 검증")
    parser.add_argument("--ground-truth", required=True, help="ground_truth.json 경로")
    parser.add_argument("--images-dir", required=True, help="검증 이미지(검증N.jpg) 폴더 경로")
    parser.add_argument("--output", required=True, help="검증 결과를 저장할 CSV 경로")
    parser.add_argument("--limit", type=int, default=None,
                        help="앞에서부터 N개 이미지만 처리 (샘플 테스트용, 미지정 시 전체 처리)")
    args = parser.parse_args()

    asyncio.run(evaluate(Path(args.ground_truth), Path(args.images_dir),
                         Path(args.output), limit=args.limit))


if __name__ == "__main__":
    main()
