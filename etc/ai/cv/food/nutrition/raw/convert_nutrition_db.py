"""
convert_nutrition_db.py

식품의약품안전처 원본 DB(food_nutrition.xlsx, 79MB)를
서비스용 경량 CSV(food_disease_scores.csv)로 변환합니다.

- 질환 관련 핵심 영양소만 추출
- 식이섬유·포화지방 컬럼 추가
- 고혈압·당뇨·이상지질혈증·비만·빈혈 점수 자동 계산

한 번만 실행하면 됩니다.

실행 (프로젝트 루트에서):
    python etc/ai/cv/food/nutrition/raw/convert_nutrition_db.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]  # raw → nutrition → food → cv → ai → etc → AH_03_03
sys.path.insert(0, str(ROOT))

import pandas as pd

from ai_runtime.cv.food.nutrition.scoring.disease_food_scorer import (
    DiseaseFoodScorer,
    write_food_disease_score_csv,
    _clean_number,
    _normalize_dataframe,
)
from ai_runtime.cv.food.nutrition.scoring.schemas import (
    DiseaseFoodScoreRecord,
    FoodNutritionRow,
)

HERE    = Path(__file__).parent
SRC     = HERE / "food_nutrition.xlsx"
DEST    = ROOT / "ai_runtime" / "cv" / "food" / "nutrition" / "data" / "food_disease_scores.csv"
RULES   = ROOT / "ai_runtime" / "cv" / "food" / "nutrition" / "rules" / "disease_score_rules.json"

# 새 Excel의 컬럼명 → 내부 필드명 매핑
NEW_COLUMN_MAP = {
    "식품명":            "food_name",
    "식품대분류":        "category",
    "식품상세분류":      "sub_category",
    "1회제공량":         "serving_weight_g",   # 단위는 내용량_단위 컬럼 확인
    "에너지(㎉)":        "energy_kcal",
    "단백질(g)":         "protein_g",
    "지방(g)":           "fat_g",
    "탄수화물(g)":       "carbohydrate_g",
    "총당류(g)":         "sugar_g",
    "총 식이섬유(g)":    "fiber_g",            # 추가
    "나트륨(㎎)":        "sodium_mg",
    "칼슘(㎎)":          "calcium_mg",
    "철(㎎)":            "iron_mg",
    "마그네슘(㎎)":      "magnesium_mg",
    "아연(㎎)":          "zinc_mg",
    "칼륨(㎎)":          "potassium_mg",
    "콜레스테롤(g)":     "_cholesterol_g",     # g 단위 → mg 변환 필요
    "트랜스 지방산(g)":  "trans_fat_g",
    "총 포화 지방산(g)": "saturated_fat_g",    # 추가
}

RUNTIME_COLUMNS = [
    "food_name",
    "serving_weight_g",
    "energy_kcal",
    "carbohydrate_g",
    "sugar_g",
    "fat_g",
    "protein_g",
    "calcium_mg",
    "sodium_mg",
    "potassium_mg",
    "magnesium_mg",
    "iron_mg",
    "zinc_mg",
    "cholesterol_mg",
    "trans_fat_g",
    "fiber_g",
    "saturated_fat_g",
    "dm_score",
    "htn_score",
    "dl_score",
    "obe_score",
    "anem_score",
]


def load_new_excel(path: Path) -> pd.DataFrame:
    print("📂 Excel 읽는 중... (시간이 걸릴 수 있어요)")

    # 필요한 컬럼만 읽어서 속도 개선
    target_cols = list(NEW_COLUMN_MAP.keys()) + ["내용량_단위"]
    df_head = pd.read_excel(path, header=3, nrows=0)
    usecols = [c for c in target_cols if c in df_head.columns]

    df = pd.read_excel(path, header=3, usecols=usecols)
    print(f"✅ 읽기 완료: {len(df)}행, {len(df.columns)}컬럼")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # 컬럼 이름 매핑
    rename = {k: v for k, v in NEW_COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    # 식품명 정리
    df["food_name"] = df["food_name"].astype(str).str.strip()
    df = df[df["food_name"].notna() & (df["food_name"] != "") & (df["food_name"] != "nan")]

    # g 단위인 행만 serving_weight_g 사용 (mL 등 제외)
    if "내용량_단위" in df.columns:
        df = df[df["내용량_단위"].astype(str).str.strip().isin(["g", "G", ""])]

    # 콜레스테롤 g → mg 변환
    if "_cholesterol_g" in df.columns:
        df["cholesterol_mg"] = pd.to_numeric(df["_cholesterol_g"], errors="coerce") * 1000
        df.drop(columns=["_cholesterol_g"], inplace=True)

    return df


def build_records(df: pd.DataFrame, scorer: DiseaseFoodScorer) -> list[DiseaseFoodScoreRecord]:
    records = []
    for _, row in df.iterrows():
        def get(col):
            return _clean_number(row.get(col))

        nutrition = FoodNutritionRow(
            food_name       = str(row["food_name"]),
            serving_weight_g= get("serving_weight_g"),
            energy_kcal     = get("energy_kcal"),
            carbohydrate_g  = get("carbohydrate_g"),
            sugar_g         = get("sugar_g"),
            fat_g           = get("fat_g"),
            protein_g       = get("protein_g"),
            calcium_mg      = get("calcium_mg"),
            sodium_mg       = get("sodium_mg"),
            potassium_mg    = get("potassium_mg"),
            magnesium_mg    = get("magnesium_mg"),
            iron_mg         = get("iron_mg"),
            zinc_mg         = get("zinc_mg"),
            cholesterol_mg  = get("cholesterol_mg"),
            trans_fat_g     = get("trans_fat_g"),
            fiber_g         = get("fiber_g"),
            saturated_fat_g = get("saturated_fat_g"),
        )
        scores = scorer.score_food(nutrition)
        records.append(DiseaseFoodScoreRecord(
            food_name       = nutrition.food_name,
            serving_weight_g= nutrition.serving_weight_g,
            energy_kcal     = nutrition.energy_kcal,
            carbohydrate_g  = nutrition.carbohydrate_g,
            sugar_g         = nutrition.sugar_g,
            fat_g           = nutrition.fat_g,
            protein_g       = nutrition.protein_g,
            calcium_mg      = nutrition.calcium_mg,
            sodium_mg       = nutrition.sodium_mg,
            potassium_mg    = nutrition.potassium_mg,
            magnesium_mg    = nutrition.magnesium_mg,
            iron_mg         = nutrition.iron_mg,
            zinc_mg         = nutrition.zinc_mg,
            cholesterol_mg  = nutrition.cholesterol_mg,
            trans_fat_g     = nutrition.trans_fat_g,
            fiber_g         = nutrition.fiber_g,
            saturated_fat_g = nutrition.saturated_fat_g,
            **scores.to_dict(),
        ))
    return records


if __name__ == "__main__":
    df = load_new_excel(SRC)
    df = preprocess(df)
    print(f"전처리 후: {len(df)}개 음식")

    scorer = DiseaseFoodScorer(rule_path=RULES)
    print("점수 계산 중...")
    records = build_records(df, scorer)

    write_food_disease_score_csv(records, output_path=DEST, columns=RUNTIME_COLUMNS)
    print(f"\n✅ 완료: {len(records)}개 음식 → {DEST.name}")
    print(f"   추가된 컬럼: fiber_g, saturated_fat_g")
