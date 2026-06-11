"""
KNHANES 외부 검증 스크립트
────────────────────────
공단 X1 모델 (F 실험 기준) → KNHANES 2024 데이터로 외부 검증

목적:
    공단 데이터로 학습한 모델이 다른 모집단(KNHANES)에서도
    일반화되는지 확인

주의사항:
    - 타겟 정의 차이 있음 (공단=검진값 룰 / KNHANES=유병여부/검진수치)
    - 간기능이상: GGT 없어서 AST/ALT만으로 판정
    - 이상지질혈증: HE_HCHOL(고콜레스테롤) OR HE_HTG(고중성지방) 조합

실행환경: Python 3.10+
패키지  : pandas, numpy, scikit-learn, lightgbm, catboost, pyreadstat, joblib
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score, confusion_matrix

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────
# 0. 경로 설정
# ────────────────────────────────────────────
BASE_DIR = "/Users/admin/PycharmProjects/AH_03_03/etc/ml/ai_worker"
KNHANES_PATH = os.path.join(BASE_DIR, "data", "hn24_all.sas7bdat")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "NHIS", "outputs", "Modeling_X1", "saved_models")
OUT_DIR = os.path.join(BASE_DIR, "ml", "KNHANES", "outputs", "Validation_KNHANES")
os.makedirs(OUT_DIR, exist_ok=True)

RECALL_TARGET = 0.8
EXPERIMENT_TAG = "J"  # 검증할 실험 태그 변경 (F, J 등)

# ────────────────────────────────────────────
# 1. KNHANES 데이터 로드
# ────────────────────────────────────────────
print("KNHANES 2024 로드 중...")
import pyreadstat

df, meta = pyreadstat.read_sas7bdat(KNHANES_PATH)
print(f"  완료: {df.shape[0]:,}행 × {df.shape[1]}열")
print(f"  컬럼 수: {len(df.columns)}")

# ────────────────────────────────────────────
# 2. CLINICAL_BOUNDS 이상치 처리 (X1 피처 해당 컬럼)
# ────────────────────────────────────────────
CLINICAL_BOUNDS_KN = {
    "HE_ht": (100, 250),
    "HE_wt": (20, 350),
    "HE_wc": (40, 200),
    "HE_BMI": (10, 80),
}

print("\n[이상치 → NaN 처리]")
for col, (lo, hi) in CLINICAL_BOUNDS_KN.items():
    if col not in df.columns:
        print(f"  [{col}] 컬럼 없음 — 스킵")
        continue
    before = df[col].isna().sum()
    df[col] = df[col].where((df[col] >= lo) & (df[col] <= hi), other=np.nan)
    replaced = df[col].isna().sum() - before
    if replaced > 0:
        print(f"  [{col}] → NaN: {replaced:,}건")

# ────────────────────────────────────────────
# 3. X1 피처 생성 (공단과 동일한 방식)
# ────────────────────────────────────────────
print("\n[X1 피처 생성]")

# 성별 (KNHANES: sex / 1=남, 2=여)
df["성별코드"] = df["sex"]

# 나이 중간값
df["age_mid"] = df["age"].clip(20, 95)  # 연속값이라 그대로 사용

# 신장/체중
df["신장(5cm단위)"] = df["HE_ht"]
df["체중(5kg단위)"] = df["HE_wt"]

# 허리둘레
df["허리둘레"] = df["HE_wc"]

# BMI
df["bmi"] = df["HE_BMI"].where(df["HE_BMI"].notna(), df["HE_wt"] / ((df["HE_ht"] / 100) ** 2))

# WHtR
df["waist_height_ratio"] = (df["HE_wc"] / df["HE_ht"]).round(3)

# gender_age_enc
le = LabelEncoder()
df["gender_age_str"] = df["성별코드"].astype(str) + "_" + df["age"].fillna(0).astype(int).astype(str)
df["gender_age_enc"] = le.fit_transform(df["gender_age_str"])

# obesity_combined
df["obesity_combined"] = ((df["bmi"] >= 25) & (df["waist_height_ratio"] >= 0.5)).astype(float)

# 흡연 (KNHANES: sm_now 또는 연초흡연현재 관련 변수)
# BS3_1: 현재 흡연 여부 (1=매일, 2=가끔, 3=과거, 4=비흡연)
if "BS3_1" in df.columns:
    df["smoking_current"] = (df["BS3_1"].isin([1, 2])).astype(float)
    df["smoking_ever"] = (df["BS3_1"].isin([1, 2, 3])).astype(float)
    print("  흡연: BS3_1 사용")
elif "sm_now" in df.columns:
    df["smoking_current"] = (df["sm_now"] == 1).astype(float)
    df["smoking_ever"] = (df["sm_now"].isin([1, 2])).astype(float)
    print("  흡연: sm_now 사용")
else:
    df["smoking_current"] = np.nan
    df["smoking_ever"] = np.nan
    print("  흡연: 변수 없음 → NaN")

# 음주 (KNHANES: BD1_11 또는 dr_month)
if "BD1_11" in df.columns:
    # BD1_11: 최근 1년간 음주 여부 (1=예, 2=아니오)
    df["음주여부"] = (df["BD1_11"] == 1).astype(float)
    print("  음주: BD1_11 사용")
elif "dr_month" in df.columns:
    df["음주여부"] = (df["dr_month"] >= 1).astype(float)
    print("  음주: dr_month 사용")
else:
    df["음주여부"] = np.nan
    print("  음주: 변수 없음 → NaN")

# ────────────────────────────────────────────
# 4. 타겟 변수 생성
# ────────────────────────────────────────────
print("\n[타겟 변수 생성]")

# [1] 당뇨위험 — HE_DM_HbA1c (당뇨병 유병여부: 1=유병)
#     공단: 공복혈당≥100 / KNHANES: 당뇨병 유병여부
#     ⚠️ 정의 차이: 공단은 전단계 포함, KNHANES는 확진만
if "HE_DM_HbA1c" in df.columns:
    df["target_diabetes"] = np.where(df["HE_DM_HbA1c"] == 1, 1, 0)
    df.loc[df["HE_DM_HbA1c"].isna(), "target_diabetes"] = np.nan
elif "HE_glu" in df.columns:
    df["target_diabetes"] = np.where(df["HE_glu"] >= 100, 1, 0)
    df.loc[df["HE_glu"].isna(), "target_diabetes"] = np.nan
    print("  당뇨: HE_glu (공복혈당) 사용")

# [2] 고혈압 — HE_HP (고혈압 유병여부: 1=유병)
if "HE_HP" in df.columns:
    df["target_hypertension"] = np.where(df["HE_HP"] == 1, 1, 0)
    df.loc[df["HE_HP"].isna(), "target_hypertension"] = np.nan
elif "HE_sbp" in df.columns and "HE_dbp" in df.columns:
    df["target_hypertension"] = np.where((df["HE_sbp"] >= 140) | (df["HE_dbp"] >= 90), 1, 0)
    df.loc[df["HE_sbp"].isna() & df["HE_dbp"].isna(), "target_hypertension"] = np.nan

# [3] 이상지질혈증 — HE_HCHOL OR HE_HTG
if "HE_HCHOL" in df.columns and "HE_HTG" in df.columns:
    dyslip = (df["HE_HCHOL"] == 1) | (df["HE_HTG"] == 1)
    df["target_dyslipidemia"] = np.where(dyslip, 1, 0)
    na_mask = df["HE_HCHOL"].isna() & df["HE_HTG"].isna()
    df.loc[na_mask, "target_dyslipidemia"] = np.nan
elif "HE_chol" in df.columns:
    # 수치값으로 직접 계산
    hdl_low = (
        (((df["성별코드"] == 1) & (df["HE_HDL_st2"] < 40)) | ((df["성별코드"] == 2) & (df["HE_HDL_st2"] < 50)))
        if "HE_HDL_st2" in df.columns
        else pd.Series(False, index=df.index)
    )
    dyslip = (df["HE_chol"] >= 200) | hdl_low
    df["target_dyslipidemia"] = np.where(dyslip, 1, 0)
    df.loc[df["HE_chol"].isna(), "target_dyslipidemia"] = np.nan

# [4] 대사증후군 — 구성항목으로 직접 계산
#     허리둘레, TG, HDL, 혈압, 혈당 5항목 중 3개 이상
if all(c in df.columns for c in ["HE_wc", "HE_TG_st2", "HE_HDL_st2", "HE_sbp", "HE_dbp", "HE_glu"]):
    abdom = ((df["성별코드"] == 1) & (df["HE_wc"] >= 90)) | ((df["성별코드"] == 2) & (df["HE_wc"] >= 85))
    tg_hi = df["HE_TG_st2"] >= 150
    hdl_low_ms = ((df["성별코드"] == 1) & (df["HE_HDL_st2"] < 40)) | ((df["성별코드"] == 2) & (df["HE_HDL_st2"] < 50))
    bp_ms = (df["HE_sbp"] >= 130) | (df["HE_dbp"] >= 85)
    gluc_ms = df["HE_glu"] >= 100
    ms_score = (
        abdom.astype(float)
        + tg_hi.astype(float)
        + hdl_low_ms.astype(float)
        + bp_ms.astype(float)
        + gluc_ms.astype(float)
    )
    df["target_metabolic"] = np.where(ms_score >= 3, 1, 0)
    ms_na = df[["HE_wc", "HE_TG_st2", "HE_HDL_st2", "HE_sbp", "HE_glu"]].isna().sum(axis=1)
    df.loc[ms_na >= 3, "target_metabolic"] = np.nan
    print("  대사증후군: 5항목 직접 계산")
else:
    df["target_metabolic"] = np.nan
    print("  대사증후군: 필요 컬럼 부족 → NaN")

# [5] 간기능이상 — HE_ast, HE_alt (GGT 없음)
if "HE_ast" in df.columns and "HE_alt" in df.columns:
    liver = (df["HE_ast"] > 40) | (df["HE_alt"] > 40)
    df["target_liver"] = np.where(liver, 1, 0)
    liver_na = df[["HE_ast", "HE_alt"]].isna().all(axis=1)
    df.loc[liver_na, "target_liver"] = np.nan
    print("  간기능이상: AST/ALT만 사용 (GGT 없음)")
else:
    df["target_liver"] = np.nan
    print("  간기능이상: 컬럼 없음 → NaN")

# [6] 신장단백뇨 — HE_Upro (요단백: 공단과 동일 기준 코드≥3)
if "HE_Upro" in df.columns:
    df["target_proteinuria"] = np.where(df["HE_Upro"] >= 3, 1, 0)
    df.loc[df["HE_Upro"].isna(), "target_proteinuria"] = np.nan
    print("  신장단백뇨: HE_Upro 사용")
else:
    df["target_proteinuria"] = np.nan
    print("  신장단백뇨: 컬럼 없음 → NaN")

TARGETS = {
    "당뇨위험": "target_diabetes",
    "고혈압": "target_hypertension",
    "이상지질혈증": "target_dyslipidemia",
    "대사증후군": "target_metabolic",
    "간기능이상": "target_liver",
    "신장단백뇨": "target_proteinuria",
}

# ────────────────────────────────────────────
# 5. 피처 컬럼 정의 (F 실험과 동일 — bmi_category 제외)
# ────────────────────────────────────────────
FEATURE_COLS = [
    "성별코드",
    "신장(5cm단위)",
    "체중(5kg단위)",
    "허리둘레",
    "음주여부",
    "bmi",
    "waist_height_ratio",
    "age_mid",
    "gender_age_enc",
    "obesity_combined",
    "smoking_current",
    "smoking_ever",
]

# ────────────────────────────────────────────
# 6. 타겟별 유효 샘플 현황
# ────────────────────────────────────────────
print("\n[KNHANES 타겟별 양성률]")
for t_name, t_col in TARGETS.items():
    if t_col not in df.columns:
        print(f"  {t_name:8s}: 컬럼 없음")
        continue
    valid = df[t_col].dropna()
    if len(valid) == 0:
        print(f"  {t_name:8s}: 유효 샘플 없음")
        continue
    pos_rate = valid.mean() * 100
    print(f"  {t_name:8s}: {len(valid):>8,}행 | 양성률 {pos_rate:.1f}%")

# ────────────────────────────────────────────
# 7. 저장된 모델 로드 및 외부 검증
# ────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("KNHANES 외부 검증")
print(f"{'=' * 60}")

import joblib

all_results = []

for target_name, target_col in TARGETS.items():
    if target_col not in df.columns:
        continue

    df_use = df[FEATURE_COLS + [target_col]].dropna(subset=[target_col])
    X_ext = df_use[FEATURE_COLS]
    y_ext = df_use[target_col].astype(int)

    if len(y_ext) == 0 or y_ext.sum() == 0:
        print(f"\n  [{target_name}] 유효 샘플 없음 — 스킵")
        continue

    pos_rate = y_ext.mean() * 100
    print(f"\n  [{target_name}] 유효: {len(y_ext):,}행 | 양성률: {pos_rate:.1f}%")

    # 저장된 모델 로드
    # 태그별 모델 우선순위: 스태킹 → 단일 모델
    model_path = os.path.join(MODEL_DIR, f"{EXPERIMENT_TAG}_{target_name}_stacking.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, f"{EXPERIMENT_TAG}_{target_name}_best.pkl")
    if not os.path.exists(model_path):
        print(f"    모델 파일 없음: {model_path}")
        print(f"    → saved_models 디렉토리에 모델 저장 필요 (아래 모델 저장 섹션 참고)")
        continue

    model_data = joblib.load(model_path)

    # 스태킹 모델 vs 단일 모델 분기
    if isinstance(model_data, dict) and "meta_model" in model_data:
        # 스태킹 모델
        imputer = model_data["imputer"]
        base_models = model_data["base_models"]
        meta_model = model_data["meta_model"]

        X_imp = imputer.transform(X_ext)
        base_probs = np.column_stack([bm.predict_proba(X_imp)[:, 1] for bm in base_models.values()])
        y_prob = meta_model.predict_proba(base_probs)[:, 1]
        print(f"    [스태킹 모델] base: {list(base_models.keys())} → LR 메타")
    else:
        # 단일 모델
        model = model_data
        imputer = KNNImputer(n_neighbors=5)
        X_imp = imputer.fit_transform(X_ext)
        y_prob = model.predict_proba(X_imp)[:, 1]
    auc = roc_auc_score(y_ext, y_prob)

    # threshold 탐색
    best_thr = 0.5
    best_f1 = 0.0
    for thr in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= thr).astype(int)
        recall = recall_score(y_ext, y_pred, zero_division=0)
        f1 = f1_score(y_ext, y_pred, zero_division=0)
        if recall >= RECALL_TARGET and f1 > best_f1:
            best_f1, best_thr = f1, thr
    if best_f1 == 0.0:
        best_thr = 0.5

    y_pred = (y_prob >= best_thr).astype(int)
    recall = recall_score(y_ext, y_pred, zero_division=0)
    f1 = f1_score(y_ext, y_pred, zero_division=0)
    prec = precision_score(y_ext, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_ext, y_pred).ravel()

    recall_ok = "✅" if recall >= RECALL_TARGET else "❌"
    f1_ok = "✅" if f1 >= 0.6 else "❌"
    print(
        f"    AUC: {auc:.4f} | Recall: {recall:.4f} {recall_ok} | "
        f"F1: {f1:.4f} {f1_ok} | Precision: {prec:.4f} | Threshold: {best_thr:.2f}"
    )

    all_results.append(
        {
            "타겟": target_name,
            "데이터": "KNHANES_2024",
            "AUC": round(auc, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "Precision": round(prec, 4),
            "Threshold": best_thr,
            "샘플수": len(y_ext),
            "양성률": round(pos_rate, 1),
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
        }
    )

# ────────────────────────────────────────────
# 8. 결과 저장
# ────────────────────────────────────────────
if all_results:
    results_df = pd.DataFrame(all_results)
    results_df["실험"] = EXPERIMENT_TAG

    # 태그별 저장
    tag_path = os.path.join(OUT_DIR, f"validation_knhanes_{EXPERIMENT_TAG}.csv")
    results_df.to_csv(tag_path, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {tag_path}")

    # 누적 저장
    all_path = os.path.join(OUT_DIR, "validation_knhanes_all.csv")
    if os.path.exists(all_path):
        prev_df = pd.read_csv(all_path, encoding="utf-8-sig")
        # 같은 태그 결과는 덮어쓰기
        prev_df = prev_df[prev_df["실험"] != EXPERIMENT_TAG]
        all_df = pd.concat([prev_df, results_df], ignore_index=True)
    else:
        all_df = results_df
    all_df.to_csv(all_path, index=False, encoding="utf-8-sig")
    print(f"누적 저장: {all_path}")

    print(f"\n{'=' * 60}")
    print(f"외부 검증 결과 요약 [{EXPERIMENT_TAG}]")
    print(f"{'=' * 60}")
    print(results_df[["타겟", "AUC", "Recall", "F1", "Precision", "샘플수", "양성률"]].to_string(index=False))
else:
    print("\n⚠️  검증 결과 없음 — 모델 파일을 saved_models 디렉토리에 저장해주세요")
    print(f"   경로: {MODEL_DIR}")
    print("""
   모델 저장 방법 (modeling_x1.py에 추가):
   ----------------------------------------
   import joblib
   os.makedirs(MODEL_DIR, exist_ok=True)

   # 각 타겟 최고 모델 저장
   joblib.dump(best_model, f"{MODEL_DIR}/F_{target_name}_best.pkl")
   """)

print("\n완료!")
