"""
GPT Vision 식단 인식 결과 검증 스크립트

build_validation_set.py 가 생성한 diet_ground_truth.csv 를 정답으로 사용해서,
각 샘플 이미지를 GPT Vision에 입력하고 추출된 음식명을 정답과 비교한다.

사용법:
    OPENAI_API_KEY=sk-... python evaluate_diet_vision.py \
        --ground-truth "ai_runtime/cv/food/eval/diet_ground_truth.csv" \
        --output "ai_runtime/cv/food/eval/diet_eval_result.csv"

연결 방식:
- ai_runtime/cv/providers/gpt_vision.py 의 VisionClient.analyze(analysis_type, image_bytes, media_type)를
  그대로 사용합니다. (analysis_type=AnalysisType.DIET 로 호출하면 router.py와 동일한 프롬프트/모델 경로를 탑니다.)
  반환값은 {"foods": [{"name": "...", ...}, ...], "analysis_status": ..., ...} 형태이므로
  foods[*].name 목록을 추출 결과로 사용합니다.
- VisionClient는 비동기(async)이므로 본 스크립트도 asyncio로 동작합니다.
- API 키는 ai_runtime/cv/settings.py 의 VisionSettings(.env의 OPENAI_API_KEY)를 그대로 사용합니다.

정답 판정 방식:
- ground truth json의 정답(Name)은 로마자 표기(예: galibi)이지만, build_validation_set.py가 만드는
  CSV에는 [라벨] 폴더명에서 가져온 한글 음식명(food_name_kor, 예: 가리비)도 함께 들어 있습니다.
  즉 폴더 구조 자체가 이미 "로마자<->한글" 매핑표 역할을 하므로, 별도 매핑표 없이
  food_name_kor를 기준으로 GPT 추출 결과(한글)에 포함되는지로 O/X를 판정합니다 (is_match 참고).
"""

import argparse
import asyncio
import csv
import json
from difflib import SequenceMatcher
from pathlib import Path

# 문자열이 정확히 포함되지는 않아도, 유사도가 이 값 이상이면 같은 음식으로 인정한다.
# (예: "갈비찜" vs "간장 갈비찜"처럼 표기가 더 다른 경우까지 흡수하기 위함)
# 값이 너무 낮으면 실제로 다른 음식("갈비탕"/"설렁탕")까지 정답으로 묶일 위험이 있고,
# 너무 높으면 사실상 포함 매칭과 차이가 없어진다 — 0.6은 1차 기준값이며 운영 결과를 보고 조정한다.
SIMILARITY_THRESHOLD = 0.6

from ai_runtime.cv.providers.gpt_vision import AnalysisType, VisionClient
from ai_runtime.cv.settings import VisionSettings

_settings = VisionSettings()
_client = VisionClient(api_key=_settings.openai_api_key, model=_settings.openai_model)


def resolve_image_path(base_dir: Path, raw_path: str) -> Path:
    """CSV의 sample_image_path(Windows 상대경로, 백슬래시 포함)를
    실행 환경(Linux 컨테이너) 기준 절대경로로 변환한다.

    diet_ground_truth.csv는 Windows에서 생성되어 "sample_images\\가리비\\..."처럼
    백슬래시 구분자 + ground-truth CSV 폴더 기준 상대경로로 저장돼 있다.
    컨테이너의 작업 디렉터리(/app)는 CSV 위치(ai_runtime/cv/food/eval/)와 다르므로
    base_dir(=ground-truth CSV의 부모 폴더)를 기준으로 다시 합쳐야 한다.
    """
    normalized = raw_path.replace("\\", "/")
    return (base_dir / normalized).resolve()


async def call_gpt_vision(path: Path) -> list[str]:
    """GPT Vision(DIET 분석)으로 이미지에서 음식명 목록을 추출한다."""
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    image_bytes = path.read_bytes()

    result = await _client.analyze(
        analysis_type=AnalysisType.DIET,
        image_bytes=image_bytes,
        media_type=media_type,
    )
    foods = result.get("foods") or []
    return [f.get("name", "").strip() for f in foods if f.get("name")]


def normalize_food_name(name: str) -> str:
    """비교용 정규화: 앞뒤 공백 제거 + 내부 공백 제거.

    "골드키위"(정답) vs "골드 키위"(GPT 추출)처럼 표기 방식(띄어쓰기) 차이만 있는
    경우를 같은 음식으로 인정하기 위함이다. 다만 이건 '표기 차이'만 보정하는 것이고,
    "갈비탕" 정답에 "설렁탕"으로 답한 것처럼 GPT가 실제로 다른 음식으로 인식한 경우는
    정규화해도 여전히 불일치로 남는다 — 이런 경우까지 정답 처리되면 정답률 신뢰성이
    깨지므로 의도적으로 그대로 둔다.
    """
    return name.strip().replace(" ", "")


def best_similarity(target: str, predicted_names: list[str]) -> tuple[float, str]:
    """target과 가장 유사한 predicted 문자열, 그리고 그 유사도(0~1)를 반환한다."""
    best_ratio = 0.0
    best_name = ""
    for predicted in predicted_names:
        ratio = SequenceMatcher(None, target, normalize_food_name(predicted)).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_name = predicted
    return best_ratio, best_name


def is_match(food_name_kor: str, predicted_names: list[str]) -> tuple[bool, float, str]:
    """
    정답(food_name_kor, 예: '가리비')과 GPT 추출 음식명 목록을 비교해 정답 여부를 판정한다.
    (정답 여부, 최고 유사도, 가장 유사했던 추출명) 튜플을 반환한다 — 결과 CSV에서
    "유사도가 왜 그렇게 판정됐는지" 추적할 수 있도록 하기 위함이다.

    food_name_kor는 build_validation_set.py가 [라벨] 폴더명에서 그대로 가져온 한글 음식명이며,
    json의 로마자 정답(answer_name, 예: 'galibi')과 동일 대상을 가리킨다 — 즉 폴더 구조 자체가
    이미 '로마자<->한글' 매핑표 역할을 하므로 별도 매핑표 없이 이 필드로 비교하면 된다.

    판정은 2단계로 이루어진다.
    1) '포함' 매칭: GPT는 "가리비찜", "간장 가리비구이"처럼 조리법/양념을 붙여 응답하므로
       완전 일치 대신 포함 여부로 먼저 본다 (공백 차이는 normalize_food_name으로 흡수).
    2) 포함되지 않으면, 문자열 유사도(SequenceMatcher)가 SIMILARITY_THRESHOLD 이상인
       경우도 정답으로 인정한다 — "골드키위" vs "키위(생것)"처럼 표기/수식어 차이로
       포함 관계가 깨지는 near-miss를 흡수하기 위함이다.

    주의: 이 유사도 비교는 GPT의 응답을 받은 '이후'에 채점 단계에서만 적용된다.
    GPT에게 정답 후보를 미리 알려주는 것이 아니므로, 'GPT가 후보 없이 얼마나 정확히
    인식하는가'라는 검증의 본래 목적은 훼손되지 않는다.
    """
    target = normalize_food_name(food_name_kor)
    if not target:
        return False, 0.0, ""

    normalized_predicted = [normalize_food_name(p) for p in predicted_names]
    if any(target in p for p in normalized_predicted):
        return True, 1.0, ""

    ratio, best_name = best_similarity(target, predicted_names)
    return ratio >= SIMILARITY_THRESHOLD, ratio, best_name


async def evaluate(ground_truth_csv: Path, output_csv: Path, limit: int | None = None):
    rows_out = []
    correct = 0
    total = 0
    base_dir = ground_truth_csv.resolve().parent

    with open(ground_truth_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if limit is not None and total >= limit:
                break
            total += 1
            image_path = row["sample_image_path"]
            resolved_path = resolve_image_path(base_dir, image_path)
            answer = row["answer_name"]
            food_name_kor = row["food_name_kor"]

            try:
                predicted_names = await call_gpt_vision(resolved_path)
            except Exception as e:
                predicted_names = []
                print(f"[경고] GPT Vision 호출 실패: {resolved_path} ({e})")

            is_correct, similarity, matched_by_similarity = is_match(food_name_kor, predicted_names)

            if is_correct:
                correct += 1

            rows_out.append({
                "category": row["category"],
                "food_name_kor": food_name_kor,
                "answer_name": answer,
                "predicted_names": " | ".join(predicted_names),
                "is_correct": "O" if is_correct else "X",
                "match_type": (
                    "포함" if is_correct and similarity == 1.0
                    else "유사도" if is_correct
                    else "불일치"
                ),
                "similarity": f"{similarity:.2f}",
                "closest_predicted": matched_by_similarity,
                "image_path": image_path,
            })

    fieldnames = ["category", "food_name_kor", "answer_name",
                  "predicted_names", "is_correct", "match_type",
                  "similarity", "closest_predicted", "image_path"]
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    accuracy = (correct / total * 100) if total else 0.0

    wrong_rows = [r for r in rows_out if r["is_correct"] == "X"]
    summary = {
        "total": total,
        "correct": correct,
        "wrong": len(wrong_rows),
        "accuracy_percent": round(accuracy, 1),
        "similarity_threshold": SIMILARITY_THRESHOLD,
    }

    # JSON 출력: CSV와 같은 폴더/이름으로 저장하되 확장자만 .json으로 바꾼다.
    # summary(정답률 등 요약)와 results(행별 상세, 특히 오답 분석에 필요한
    # predicted_names/closest_predicted/similarity/match_type 포함)를 함께 담는다.
    output_json = output_csv.with_suffix(".json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": rows_out}, f, ensure_ascii=False, indent=2)

    print(f"\n총 {total}건 중 {correct}건 정답 (정답률 {accuracy:.1f}%)")
    print(f"상세 결과(CSV): {output_csv}")
    print(f"상세 결과(JSON): {output_json}")

    if wrong_rows:
        print(f"\n오답 {len(wrong_rows)}건:")
        for r in wrong_rows:
            print(f"  - [{r['category']}/{r['food_name_kor']}] 정답={r['answer_name']} / 추출={r['predicted_names']}"
                  f" (가장 유사: '{r['closest_predicted']}', 유사도={r['similarity']})")


def main():
    parser = argparse.ArgumentParser(description="GPT Vision 식단 인식 결과 검증")
    parser.add_argument("--ground-truth", required=True, help="diet_ground_truth.csv 경로")
    parser.add_argument("--output", required=True, help="검증 결과를 저장할 CSV 경로")
    parser.add_argument("--limit", type=int, default=None,
                        help="앞에서부터 N개 행만 처리 (샘플 테스트용, 미지정 시 전체 처리)")
    args = parser.parse_args()

    asyncio.run(evaluate(Path(args.ground_truth), Path(args.output), limit=args.limit))


if __name__ == "__main__":
    main()
