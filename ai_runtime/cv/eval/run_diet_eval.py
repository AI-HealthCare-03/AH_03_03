"""
ai_runtime/cv/eval/run_diet_eval.py

GPT Vision 식단 분류 + 영양성분 매칭 평가 스크립트

흐름:
  1. 카테고리별 5장 이미지를 GPT Vision으로 분석
  2. 추출된 음식명을 Excel 영양 DB에서 조회 (퍼지 매칭)
  3. DB 조회 성공 시 공식 영양값 사용 / 실패 시 GPT 추정값 사용
  4. 결과를 CSV로 저장 + Langfuse에 자동 로깅

실행 방법 (프로젝트 루트에서):
    python ai_runtime/cv/eval/run_diet_eval.py
"""

import asyncio
import csv
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from ai_runtime.cv.providers.gpt_vision import VisionClient, AnalysisType
from ai_runtime.cv.settings import VisionSettings

# ── 경로 설정 ────────────────────────────────────────────
IMAGES_DIR  = Path(__file__).parent / "images" / "diet"
LABELS_DIR  = Path(__file__).parent / "labels" / "diet"
RESULTS_DIR = Path(__file__).parent / "results"
# 새 DB (42,822개) 사용
NUTRITION_DB_PATH = ROOT / "ai_runtime" / "cv" / "food" / "nutrition" / "data" / "food_disease_scores.csv"
RESULTS_DIR.mkdir(exist_ok=True)

IMAGES_PER_CATEGORY = 5
RANDOM_SEED = 42


# ── 영양 DB 로딩 ─────────────────────────────────────────
def load_nutrition_db() -> pd.DataFrame:
    df = pd.read_csv(NUTRITION_DB_PATH)
    # 컬럼명이 이미 영문(food_name)으로 되어있음
    df["음식명"] = df["food_name"].astype(str).str.strip()
    return df

NUTRITION_DB = load_nutrition_db()


def lookup_nutrition(food_name: str, search_keyword: str = "") -> dict | None:
    """
    음식명 또는 search_keyword로 영양 DB를 조회합니다.
    완전 일치 → 부분 일치 순으로 시도합니다.
    반환값은 DB의 중량(g) 기준 영양값입니다.
    """
    candidates = [food_name, search_keyword]
    candidates = [c.strip() for c in candidates if c.strip()]

    for candidate in candidates:
        # 1순위: 완전 일치
        row = NUTRITION_DB[NUTRITION_DB["음식명"] == candidate]
        if not row.empty:
            return row.iloc[0].to_dict()

        # 2순위: DB 항목이 candidate를 포함
        row = NUTRITION_DB[NUTRITION_DB["음식명"].str.contains(candidate, na=False, regex=False)]
        if not row.empty:
            return row.iloc[0].to_dict()

        # 3순위: candidate가 DB 항목을 포함 (짧은 키워드로 검색)
        if len(candidate) >= 2:
            for db_name in NUTRITION_DB["음식명"]:
                if db_name in candidate:
                    row = NUTRITION_DB[NUTRITION_DB["음식명"] == db_name]
                    if not row.empty:
                        return row.iloc[0].to_dict()

    return None


def scale_nutrition(db_row: dict, grams: float) -> dict:
    """
    DB는 특정 중량(g) 기준이므로, 사용자 용량(grams)에 맞게 스케일합니다.
    예) DB가 100g 기준, 사용자가 150g → 모든 영양값 × 1.5
    """
    base_weight = float(db_row.get("중량(g)", 100) or 100)
    ratio = grams / base_weight if base_weight > 0 else 1.0
    return {
        "칼로리(kcal)": round(float(db_row.get("에너지(kcal)", 0) or 0) * ratio, 1),
        "탄수화물(g)":  round(float(db_row.get("탄수화물(g)", 0) or 0) * ratio, 1),
        "당류(g)":      round(float(db_row.get("당류(g)", 0) or 0) * ratio, 1),
        "단백질(g)":    round(float(db_row.get("단백질(g)", 0) or 0) * ratio, 1),
        "지방(g)":      round(float(db_row.get("지방(g)", 0) or 0) * ratio, 1),
        "나트륨(mg)":   round(float(db_row.get("나트륨(mg)", 0) or 0) * ratio, 1),
        "콜레스테롤(mg)": round(float(db_row.get("콜레스테롤(mg)", 0) or 0) * ratio, 1),
        "출처": "DB",
        "DB음식명": db_row.get("음식명", ""),
        "DB기준중량(g)": base_weight,
    }


def parse_grams(estimated_amount: str) -> float:
    """
    GPT가 추정한 용량 문자열에서 g 수치를 추출합니다.
    예) '1인분(300g)' → 300.0 / '200g' → 200.0 / '1공기(210g)' → 210.0
    """
    import re
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*g", estimated_amount or "")
    if matches:
        return float(matches[-1])  # 괄호 안 g 수치 우선
    return 100.0  # 기본값


# ── 라벨 파일 로딩 ────────────────────────────────────────
def find_label_folder(category: str) -> Path | None:
    for candidate in [category, category + " json"]:
        p = LABELS_DIR / candidate
        if p.exists():
            return p
    for d in LABELS_DIR.iterdir():
        if d.is_dir() and d.name.startswith(category):
            return d
    return None


def load_ground_truth(image_path: Path) -> dict | None:
    label_folder = find_label_folder(image_path.parent.name)
    if not label_folder:
        return None
    label_path = label_folder / (image_path.stem + ".json")
    if not label_path.exists():
        return None
    with open(label_path, encoding="utf-8") as f:
        data = json.load(f)
    return data[0] if isinstance(data, list) and data else data


# ── 조리법 접미사 — 단독으로는 매칭 기준으로 쓰지 않음 ───
COOKING_SUFFIXES = {"구이", "볶음", "찜", "탕", "국", "찌개", "튀김", "조림", "무침", "전골", "볶음밥", "샐러드"}

# ── 동의어 테이블 ─────────────────────────────────────────
# 규칙: 단독 "떡", "구이", "김" 같은 1~2글자 상위 개념어 금지
# 반드시 구체적인 표현만 등록
SYNONYMS: dict[str, list[str]] = {
    # 떡류 — 각 떡은 고유명으로만 매칭
    "가래떡":    ["가래떡", "흰떡", "백떡"],
    "경단":      ["경단", "찹쌀경단"],
    "바람떡":    ["바람떡"],
    "시루떡":    ["시루떡"],
    "송편":      ["송편"],
    "떡볶이":    ["떡볶이"],
    # 구이류 — 재료+구이 조합 또는 재료명(3자 이상)만 허용
    "가지구이":  ["가지구이", "가지"],
    "감자구이":  ["감자구이", "감자"],
    "고등어구이":["고등어구이", "고등어"],
    "새우구이":  ["새우구이", "새우"],
    "갈비구이":  ["갈비구이", "갈비"],
    "닭갈비구이":["닭갈비구이", "닭갈비"],
    "생선구이":  ["생선구이", "생선"],
    "소시지구이":["소시지구이", "소시지"],
    "아스파라거스구이": ["아스파라거스구이", "아스파라거스"],
    # 찌개/국 — 재료+찌개 조합 또는 재료명(3자 이상)만 허용
    "된장찌개":  ["된장찌개", "된장"],
    "순두부찌개":["순두부찌개", "순두부"],
    "생선찌개":  ["생선찌개", "생선"],
    "새우매운탕":["새우매운탕", "매운탕"],
    # 볶음류
    "낚지볶음":  ["낙지볶음", "낚지볶음", "낙지", "낚지"],
    "닭살채소두반장볶음": ["두반장", "닭살"],
    "닭파프리카볶음": ["파프리카볶음", "파프리카"],
    "쇠고기볶음":["소고기볶음", "쇠고기볶음", "소고기", "쇠고기"],
    "아스파라거스볶음밥": ["아스파라거스볶음밥", "아스파라거스"],
    # 외래어/영어명
    "blt샌드위치":  ["샌드위치", "BLT", "blt"],
    "라따뚜이":     ["라타투이", "라따뚜이"],
    "메쉬드포테이토": ["으깬감자", "매시드포테이토", "매쉬드포테이토"],
    "가츠동":       ["가츠동", "가츠"],
    "고르곤졸라":   ["고르곤졸라", "블루치즈"],
    "고르곤졸라피자": ["고르곤졸라피자", "고르곤졸라"],
    "도리야끼":     ["데리야키", "도리야끼"],
    "나가사끼짬뽕": ["나가사끼짬뽕", "짬뽕"],
    "미소라멘":     ["미소라멘", "라멘"],
    "미소장국":     ["미소장국", "미소국"],
    # 기타 표기 차이
    "게맛살":    ["게맛살", "크래미"],
    "무우":      ["무우", "무"],
    "떡만두국,고기만두": ["떡만두국", "만두국"],
}


# ── 분류 정확도 판정 ──────────────────────────────────────
def is_correct(gpt_foods: list[dict], category: str) -> bool:
    """
    GPT 출력이 정답 카테고리와 일치하는지 다단계로 판정합니다.

    1순위: 동의어 테이블 (구체적 표현만 등록)
    2순위: 카테고리명 전체 또는 3글자 이상 부분 문자열
           단, 조리법 접미사(구이/볶음 등)는 단독 매칭 제외
    """
    if not gpt_foods or not category:
        return False

    # GPT 출력 통합 문자열 (공백 제거)
    gpt_texts = []
    for food in gpt_foods:
        name = food.get("name", "").replace(" ", "").lower()
        kw   = food.get("search_keyword", "").replace(" ", "").lower()
        gpt_texts.append(name + kw)
    combined_all = " ".join(gpt_texts)

    # 1순위: 동의어 테이블
    for syn in SYNONYMS.get(category, []):
        if syn.lower().replace(" ", "") in combined_all:
            return True

    # 2순위: 카테고리명 부분 문자열 (3글자 이상만, 조리법 접미사 단독 제외)
    cat_clean = category.replace(" ", "").lower()
    substrings = set()
    substrings.add(cat_clean)  # 전체 항상 포함
    for length in range(3, len(cat_clean) + 1):
        for start in range(len(cat_clean) - length + 1):
            sub = cat_clean[start:start + length]
            if sub not in COOKING_SUFFIXES:  # 조리법 단독 제외
                substrings.add(sub)

    for sub in substrings:
        if sub in combined_all:
            return True

    return False


# ── 이미지 1장 평가 ───────────────────────────────────────
async def eval_one(client: VisionClient, image_path: Path, category: str) -> dict:
    gt = load_ground_truth(image_path)
    gt_name = gt.get("Name", "") if gt else ""

    try:
        result = await client.analyze(
            analysis_type=AnalysisType.DIET,
            image_bytes=image_path.read_bytes(),
            media_type="image/jpeg",
        )
        gpt_foods = result.get("foods", [])
        status    = result.get("analysis_status", "failed")
        correct   = is_correct(gpt_foods, category)
        gpt_names = " | ".join(f.get("name", "") for f in gpt_foods)

        # 첫 번째 음식으로 영양 조회
        rows = []
        for food in gpt_foods:
            food_name      = food.get("name", "")
            search_keyword = food.get("search_keyword", "")
            estimated_amt  = food.get("estimated_amount", "")
            grams          = parse_grams(estimated_amt)
            gpt_nutrition  = food.get("nutrition", {})

            db_row = lookup_nutrition(food_name, search_keyword)
            if db_row:
                nutrition = scale_nutrition(db_row, grams)
            else:
                # DB 미매칭 → GPT 추정값 사용
                nutrition = {
                    "칼로리(kcal)":   gpt_nutrition.get("칼로리"),
                    "탄수화물(g)":    gpt_nutrition.get("탄수화물"),
                    "당류(g)":        gpt_nutrition.get("당류"),
                    "단백질(g)":      gpt_nutrition.get("단백질"),
                    "지방(g)":        gpt_nutrition.get("지방"),
                    "나트륨(mg)":     gpt_nutrition.get("나트륨"),
                    "콜레스테롤(mg)": None,
                    "출처":           "GPT추정",
                    "DB음식명":       "",
                    "DB기준중량(g)":  None,
                }
            rows.append({
                "category":        category,
                "image":           image_path.name,
                "gt_name":         gt_name,
                "gpt_food":        food_name,
                "search_keyword":  search_keyword,
                "estimated_amount": estimated_amt,
                "grams":           grams,
                "correct":         correct,
                "status":          status,
                "분류신뢰도":       food.get("confidence"),
                "영양신뢰도":       gpt_nutrition.get("영양성분_신뢰도"),
                **nutrition,
                "error":           "",
            })

        return rows if rows else [_empty_row(category, image_path, gt_name, status, correct)]

    except Exception as e:
        return [_empty_row(category, image_path, gt_name, "error", False, str(e))]


def _empty_row(category, image_path, gt_name, status, correct, error=""):
    return {
        "category": category, "image": image_path.name, "gt_name": gt_name,
        "gpt_food": "", "search_keyword": "", "estimated_amount": "",
        "grams": None, "correct": correct, "status": status,
        "분류신뢰도": None, "영양신뢰도": None,
        "칼로리(kcal)": None, "탄수화물(g)": None, "당류(g)": None,
        "단백질(g)": None, "지방(g)": None, "나트륨(mg)": None,
        "콜레스테롤(mg)": None, "출처": "", "DB음식명": "", "DB기준중량(g)": None,
        "error": error,
    }


# ── 메인 실행 ─────────────────────────────────────────────
async def run_eval():
    settings = VisionSettings()
    client   = VisionClient(api_key=settings.openai_api_key, model=settings.openai_model)
    categories = sorted([d for d in IMAGES_DIR.iterdir() if d.is_dir()])

    if not categories:
        print(f"❌ 이미지 폴더가 비어있습니다: {IMAGES_DIR}")
        return

    print(f"카테고리: {len(categories)}개  |  카테고리당: {IMAGES_PER_CATEGORY}장")
    print(f"총 예상 호출: {len(categories) * IMAGES_PER_CATEGORY}회  |  모델: {settings.openai_model}")
    print(f"영양 DB: {len(NUTRITION_DB)}개 항목")
    print("-" * 60)

    random.seed(RANDOM_SEED)
    all_rows = []
    correct_count = total_count = db_match_count = 0

    for i, cat_dir in enumerate(categories, 1):
        category = cat_dir.name
        images   = list(cat_dir.glob("*.jpg"))
        sampled  = random.sample(images, min(IMAGES_PER_CATEGORY, len(images)))
        print(f"[{i}/{len(categories)}] {category}")

        for img_path in sampled:
            rows = await eval_one(client, img_path, category)
            for row in rows:
                all_rows.append(row)
                if row.get("gpt_food"):   # 음식이 추출된 행만 카운트
                    total_count += 1
                    if row["correct"]:
                        correct_count += 1
                    if row.get("출처") == "DB":
                        db_match_count += 1

            # 첫 번째 음식만 콘솔에 출력
            r = rows[0]
            icon = "✅" if r["correct"] else "❌"
            db_icon = "📋DB" if r.get("출처") == "DB" else "🤖GPT"
            kcal = r.get("칼로리(kcal)")
            kcal_str = f"{kcal}kcal" if kcal else "-"
            print(f"  {icon}{db_icon} {img_path.name[:20]} | {r['gpt_food'][:20]} | {kcal_str}")

    # ── 결과 저장 ─────────────────────────────────────────
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS_DIR / f"diet_eval_{timestamp}.csv"

    with open(result_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    accuracy  = correct_count / total_count * 100 if total_count else 0
    db_rate   = db_match_count / total_count * 100 if total_count else 0

    print("\n" + "=" * 60)
    print(f"✅ 평가 완료")
    print(f"   분류 정확도: {correct_count}/{total_count} = {accuracy:.1f}%")
    print(f"   DB 매칭률:   {db_match_count}/{total_count} = {db_rate:.1f}%")
    print(f"   결과 저장:   {result_path.name}")
    print(f"   Langfuse:    http://localhost:3000")


if __name__ == "__main__":
    asyncio.run(run_eval())
