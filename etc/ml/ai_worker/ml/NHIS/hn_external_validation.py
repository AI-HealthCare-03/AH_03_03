"""
HN CatBoost 모델 → 공단 데이터 외부검증
실험 1-A: 연령대코드 중앙값 사용 (재현성 기준)
실험 1-B: 신장/체중/나이 ±jitter 적용 (해상도 개선, 고정 시드)

실행환경: Python 3.10+
패키지  : pandas, numpy, scikit-learn, catboost
설치    : pip install pandas numpy scikit-learn catboost

※ 방법론적 주의사항
   - 가족력(12개) / 직업(7개) / 걷기일수 / 근력운동일수 / 음주빈도·량 / 걷기활동량
     → 공단 데이터에 없으므로 0 패딩 처리
   - 음주빈도·음주량: 공단의 음주여부(0/1)로 근사 대체
   - 나이: 연령대코드 중앙값 기반 (1-A) / ±2.5 jitter (1-B)
   - 키/체중: 5단위 원본 (1-A) / ±2.5 jitter (1-B)
   - 결과 해석: "가족력 없음 가정 하의 보수적 성능 하한선"으로 해석할 것
   - jitter는 고정 시드(seed=42) 적용 → 재현 가능
"""

import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score, recall_score, f1_score,
    precision_score, confusion_matrix
)

# ────────────────────────────────────────────
# 0. 경로 설정 — 본인 환경에 맞게 수정
# ────────────────────────────────────────────
BASE_DIR    = "/Users/admin/PycharmProjects/AH_03_03/etc/ml/ai_worker"
DATA_PATH   = os.path.join(BASE_DIR, "data", "국민건강보험공단_건강검진정보_2024.CSV")
MODEL_DIR   = "/Users/admin/PycharmProjects/AH_03_03/ai_worker/ml/CAT15~24/outputs/optuna_HTN_FE"
OUT_DIR     = os.path.join(BASE_DIR, "ml", "HN", "outputs", "external_validation")
os.makedirs(OUT_DIR, exist_ok=True)

N_FOLDS     = 5       # 모델 fold 수
THRESHOLD   = 0.46    # best_threshold.npy 값

# ────────────────────────────────────────────
# 1. HN 모델 feature 순서 (고정 — 절대 변경 금지)
# ────────────────────────────────────────────
HN_FEATURE_COLS = [
    '성별', '나이', '음주빈도', '음주량', '현재흡연', '걷기일수', '근력운동일수',
    '고혈압가족력_부', '고혈압가족력_모', '고혈압가족력_형제',
    '고지혈증가족력_부', '고지혈증가족력_모', '고지혈증가족력_형제',
    '당뇨가족력_부', '당뇨가족력_모', '당뇨가족력_형제',
    '키', '체중', 'BMI',
    '직업_관리전문', '직업_사무', '직업_서비스판매', '직업_농림어업',
    '직업_기능노무', '직업_주부학생', '직업_무직', '직업_작업미상',
    '나이_19_39', '나이_40대', '나이_50대', '나이_60대', '나이_70대', '나이_80이상',
    '음주위험군', '걷기활동량',
    '고혈압가족력_합산', '당뇨가족력_합산', '고지혈증가족력_합산',
    'BMI_X_나이',
]

# 공단 데이터에서 0 패딩할 컬럼 (공단에 없는 변수 전체)
ZERO_PAD_COLS = [
    '음주빈도', '음주량',                          # 음주여부로 근사 대체
    '걷기일수', '근력운동일수', '걷기활동량',        # 운동 정보 없음
    '고혈압가족력_부', '고혈압가족력_모', '고혈압가족력_형제',
    '고지혈증가족력_부', '고지혈증가족력_모', '고지혈증가족력_형제',
    '당뇨가족력_부', '당뇨가족력_모', '당뇨가족력_형제',
    '고혈압가족력_합산', '당뇨가족력_합산', '고지혈증가족력_합산',
    '직업_관리전문', '직업_사무', '직업_서비스판매', '직업_농림어업',
    '직업_기능노무', '직업_주부학생', '직업_무직', '직업_작업미상',
]

# ────────────────────────────────────────────
# 2. CLINICAL_BOUNDS 이상치 처리 (공단 코드와 동일)
# ────────────────────────────────────────────
CLINICAL_BOUNDS = {
    "신장(5cm단위)":      (100, 250),
    "체중(5kg단위)":      (20,  350),
    "허리둘레":           (40,  200),
    "수축기혈압":         (60,  280),
    "이완기혈압":         (40,  150),
    "식전혈당(공복혈당)": (40,  600),
    "총콜레스테롤":       (50,  700),
    "LDL콜레스테롤":      (10,  500),
    "HDL콜레스테롤":      (10,  150),
    "트리글리세라이드":   (20,  5000),
    "혈청지오티(AST)":    (5,   5000),
    "혈청지피티(ALT)":    (5,   5000),
    "감마지티피":         (5,   3000),
}

def preprocess(df):
    df = df.copy()
    for col, (lo, hi) in CLINICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        df[col] = df[col].where((df[col] >= lo) & (df[col] <= hi), other=np.nan)
    return df

# ────────────────────────────────────────────
# 3. 타겟 생성 (공단 코드와 동일)
# ────────────────────────────────────────────
def make_targets(df):
    # 고혈압
    df["target_hypertension"] = np.where(
        (df["수축기혈압"] >= 140) | (df["이완기혈압"] >= 90), 1, 0
    )
    df.loc[df["수축기혈압"].isna() & df["이완기혈압"].isna(), "target_hypertension"] = np.nan
    return df

# ────────────────────────────────────────────
# 4. 공단 → HN 변수 매핑 + 0 패딩
# ────────────────────────────────────────────
# 연령대코드 중앙값 매핑 (공단 코드와 동일)
AGE_MID = {5:27, 6:32, 7:37, 8:42, 9:47, 10:52,
           11:57, 12:62, 13:67, 14:72, 15:77, 16:82, 17:87, 18:92}

# 연령대코드 범위 매핑 (jitter용 — 각 코드의 실제 구간)
AGE_RANGE = {5:(19,29), 6:(30,34), 7:(35,39), 8:(40,44), 9:(45,49), 10:(50,54),
             11:(55,59), 12:(60,64), 13:(65,69), 14:(70,74), 15:(75,79),
             16:(80,84), 17:(85,89), 18:(90,99)}

def _apply_common(df):
    """중앙값/jitter 공통 처리 (성별, 흡연, 음주, 0패딩)"""
    df["성별"]     = df["성별코드"]
    df["현재흡연"] = (df["흡연상태"] == 3).astype(float)
    df["음주빈도"] = np.where(df["음주여부"] == 1, 2.0, 0.0)
    df["음주량"]   = np.where(df["음주여부"] == 1, 2.0, 0.0)
    df["음주위험군"] = 0.0
    for col in ZERO_PAD_COLS:
        if col not in df.columns:
            df[col] = 0.0
    return df

def _set_age_bins(df):
    """나이 구간 파생변수 생성 (나이 컬럼 존재 가정)"""
    df["나이_19_39"] = ((df["나이"] >= 19) & (df["나이"] <= 39)).astype(float)
    df["나이_40대"]  = ((df["나이"] >= 40) & (df["나이"] <= 49)).astype(float)
    df["나이_50대"]  = ((df["나이"] >= 50) & (df["나이"] <= 59)).astype(float)
    df["나이_60대"]  = ((df["나이"] >= 60) & (df["나이"] <= 69)).astype(float)
    df["나이_70대"]  = ((df["나이"] >= 70) & (df["나이"] <= 79)).astype(float)
    df["나이_80이상"] = (df["나이"] >= 80).astype(float)
    return df

def make_hn_features_midpoint(df):
    """1-A: 연령대코드 중앙값 / 신장·체중 5단위 원본"""
    df = df.copy()
    df = _apply_common(df)

    df["나이"]  = df["연령대코드(5세단위)"].map(AGE_MID)
    df["키"]    = df["신장(5cm단위)"]
    df["체중"]  = df["체중(5kg단위)"]
    df["BMI"]   = df["체중"] / ((df["키"] / 100) ** 2)
    df["BMI_X_나이"] = df["BMI"] * df["나이"]
    df = _set_age_bins(df)
    return df

def make_hn_features_jitter(df, seed=42):
    """1-B: 나이/신장/체중에 균등분포 jitter (고정 시드 → 재현 가능)
       - 나이   : 연령대 구간 내 균등분포 (실제 나이 근사)
       - 신장   : ±2.5cm 균등분포
       - 체중   : ±2.5kg 균등분포
       - BMI/BMI_X_나이 : jitter 적용값으로 재계산
    """
    df = df.copy()
    df = _apply_common(df)
    rng = np.random.default_rng(seed)

    # 나이 jitter — 연령대 구간 내 균등분포 샘플링
    age_lo = df["연령대코드(5세단위)"].map({k: v[0] for k, v in AGE_RANGE.items()})
    age_hi = df["연령대코드(5세단위)"].map({k: v[1] for k, v in AGE_RANGE.items()})
    df["나이"] = age_lo + rng.uniform(0, 1, len(df)) * (age_hi - age_lo)
    df["나이"] = df["나이"].round(1)

    # 신장/체중 jitter ±2.5 (공단 학습 코드와 동일 범위)
    df["키"]   = df["신장(5cm단위)"] + rng.uniform(-2.5, 2.5, len(df))
    df["체중"] = df["체중(5kg단위)"] + rng.uniform(-2.5, 2.5, len(df))

    # BMI / BMI_X_나이 재계산
    df["BMI"]        = df["체중"] / ((df["키"] / 100) ** 2)
    df["BMI_X_나이"] = df["BMI"] * df["나이"]
    df = _set_age_bins(df)
    return df

# ────────────────────────────────────────────
# 5. 데이터 로드 및 전처리
# ────────────────────────────────────────────
print("=" * 60)
print("공단 데이터 로드 중...")
df_raw = pd.read_csv(DATA_PATH, encoding="cp949")
print(f"  원본: {len(df_raw):,}행")

df_raw = preprocess(df_raw)
df_raw = make_targets(df_raw)
df_base = df_raw.dropna(subset=["target_hypertension"]).copy()
y_true  = df_base["target_hypertension"].astype(int)
print(f"  유효(타겟 non-NaN): {len(df_base):,}행")
print(f"  양성률: {y_true.mean()*100:.1f}%")
print(f"\n  0 패딩 컬럼: {len(ZERO_PAD_COLS)}개")
print(f"  음주 근사: 음주빈도/음주량 → 음주여부 기반 (음주=2.0, 비음주=0.0)")
print(f"  사용 threshold: {THRESHOLD}")

# ────────────────────────────────────────────
# 6. 모델 로드
# ────────────────────────────────────────────
print(f"\n[HN CatBoost 모델 로드 — {N_FOLDS} folds]")
print(f"  탐색 경로: {MODEL_DIR}")
models = []
for fold in range(1, N_FOLDS + 1):
    model_path = os.path.join(MODEL_DIR, f"model_fold{fold}.cbm")
    if not os.path.exists(model_path):
        print(f"  ⚠️  model_fold{fold}.cbm 없음: {model_path}")
        continue
    m = CatBoostClassifier()
    m.load_model(model_path)
    models.append(m)
    print(f"  fold{fold} 로드 완료")

if len(models) == 0:
    print("\n❌ 로드된 모델이 없습니다. MODEL_DIR 경로를 확인하세요.")
    print(f"   현재 설정: {MODEL_DIR}")
    print("   cbm 파일 위치 확인 명령어:")
    print("   find /Users/admin/PycharmProjects/AH_03_03 -name \'model_fold1.cbm\'")
    raise SystemExit(1)

def predict_ensemble(models, X):
    """fold 앙상블 평균 확률"""
    probs = [m.predict_proba(X)[:, 1] for m in models]
    return np.mean(probs, axis=0)

def evaluate(y_true, y_prob, threshold, label):
    """성능 지표 계산 + 출력"""
    y_pred = (y_prob >= threshold).astype(int)
    auc    = roc_auc_score(y_true, y_prob)
    rec    = recall_score(y_true, y_pred, zero_division=0)
    prec   = precision_score(y_true, y_pred, zero_division=0)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"{'='*60}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  Recall    : {rec:.4f}  {'✅' if rec >= 0.8 else '❌'} (목표 ≥ 0.80)")
    print(f"  Precision : {prec:.4f}")
    print(f"  F1        : {f1:.4f}  {'✅' if f1 >= 0.6 else '❌'} (목표 ≥ 0.60)")
    print(f"  Threshold : {threshold}")
    print(f"  Confusion Matrix")
    print(f"    TP: {tp:>8,}  FP: {fp:>8,}")
    print(f"    FN: {fn:>8,}  TN: {tn:>8,}")

    return {"label": label, "auc": auc, "rec": rec, "prec": prec, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

# ────────────────────────────────────────────
# 7. 실험 1-A: 중앙값
# ────────────────────────────────────────────
print(f"\n[실험 1-A] 나이/신장/체중 중앙값 사용")
df_a  = make_hn_features_midpoint(df_base)
X_a   = df_a[HN_FEATURE_COLS]
prob_a = predict_ensemble(models, X_a)
res_a  = evaluate(y_true, prob_a, THRESHOLD, "실험 1-A: 중앙값")

# ────────────────────────────────────────────
# 8. 실험 1-B: jitter
# ────────────────────────────────────────────
print(f"\n[실험 1-B] 나이(구간 균등분포) / 신장·체중 ±2.5 jitter (seed=42)")
df_b   = make_hn_features_jitter(df_base, seed=42)
X_b    = df_b[HN_FEATURE_COLS]
prob_b = predict_ensemble(models, X_b)
res_b  = evaluate(y_true, prob_b, THRESHOLD, "실험 1-B: jitter (seed=42)")

# ────────────────────────────────────────────
# 9. 공단 기준 최적 Threshold 재탐색 (1-A 기준)
# ────────────────────────────────────────────
print(f"\n[Threshold 재탐색 — 공단 데이터 기준 / 1-A 확률 사용]")
print(f"  Recall ≥ 0.80 조건에서 F1 최대 threshold 탐색 중...")

best_thr_new = 0.5
best_f1_new  = 0.0
best_rec_new = 0.0
thr_candidates = []

for thr in np.arange(0.05, 0.90, 0.01):
    thr = round(thr, 2)
    y_pred_tmp = (prob_a >= thr).astype(int)
    rec_tmp = recall_score(y_true, y_pred_tmp, zero_division=0)
    f1_tmp  = f1_score(y_true, y_pred_tmp, zero_division=0)
    prec_tmp = precision_score(y_true, y_pred_tmp, zero_division=0)
    thr_candidates.append((thr, rec_tmp, f1_tmp, prec_tmp))
    if rec_tmp >= 0.8 and f1_tmp > best_f1_new:
        best_f1_new  = f1_tmp
        best_thr_new = thr
        best_rec_new = rec_tmp

print(f"\n  {'Threshold':>10} {'Recall':>8} {'F1':>8} {'Precision':>10}")
print(f"  {'-'*42}")
# Recall 0.75~0.90 구간만 출력 (전체 출력 방지)
for thr, rec, f1, prec in thr_candidates:
    if 0.75 <= rec <= 0.92:
        marker = " ◀ 최적" if thr == best_thr_new else ""
        print(f"  {thr:>10.2f} {rec:>8.4f} {f1:>8.4f} {prec:>10.4f}{marker}")

if best_f1_new > 0:
    y_pred_new = (prob_a >= best_thr_new).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(y_true, y_pred_new).ravel()
    print(f"\n  ✅ 최적 Threshold: {best_thr_new}")
    print(f"     Recall={best_rec_new:.4f} | F1={best_f1_new:.4f}")
    print(f"     TP: {tp2:>8,}  FP: {fp2:>8,}")
    print(f"     FN: {fn2:>8,}  TN: {tn2:>8,}")
else:
    print(f"\n  ❌ Recall ≥ 0.80 달성 가능한 threshold 없음")
    # Recall 최대 지점 출력
    max_rec_row = max(thr_candidates, key=lambda x: x[1])
    print(f"     Recall 최대: thr={max_rec_row[0]:.2f} → Recall={max_rec_row[1]:.4f}, F1={max_rec_row[2]:.4f}")

# ────────────────────────────────────────────
# 10. 비교 요약
# ────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"[최종 비교 요약]")
print(f"{'='*60}")
print(f"  {'구분':<35} {'AUC':>7} {'Recall':>8} {'F1':>7}")
print(f"  {'-'*60}")
print(f"  {'HN 모델 (HN 내부검증)':<35} {'0.8152':>7} {'0.8528':>8} {'0.6009':>7}")
print(f"  {'공단 모델 (공단 자체검증)':<35} {'0.6963':>7} {'0.8127':>8} {'0.5336':>7}")
print(f"  {'공단 모델 → HN 외부검증':<35} {'0.7481':>7} {'0.8027':>8} {'0.6193':>7}")
print(f"  {'HN → 공단 1-A (중앙값)':<35} {res_a['auc']:>7.4f} {res_a['rec']:>8.4f} {res_a['f1']:>7.4f}")
print(f"  {'HN → 공단 1-B (jitter)':<35} {res_b['auc']:>7.4f} {res_b['rec']:>8.4f} {res_b['f1']:>7.4f}")

diff_auc = res_b['auc'] - res_a['auc']
diff_rec = res_b['rec'] - res_a['rec']
diff_f1  = res_b['f1']  - res_a['f1']
print(f"\n  [jitter - 중앙값 차이]")
print(f"    ΔAUC={diff_auc:+.4f}  ΔRecall={diff_rec:+.4f}  ΔF1={diff_f1:+.4f}")

# ────────────────────────────────────────────
# 10. 결과 저장
# ────────────────────────────────────────────
NOTE = "가족력12개+직업7개+운동3개=0패딩 / 음주빈도·량=음주여부근사"

rows = []
for res, tag, note in [
    (res_a, "실험1A_중앙값",        NOTE + " / 나이·신장·체중 중앙값"),
    (res_b, "실험1B_jitter_seed42", NOTE + " / 나이 구간균등분포·신장체중±2.5jitter"),
]:
    rows.append({
        "실험": tag, "방향": "HN모델→공단외부검증",
        "샘플수": len(y_true), "양성률": round(float(y_true.mean()), 4),
        "AUC": round(res["auc"], 4), "Recall": round(res["rec"], 4),
        "Precision": round(res["prec"], 4), "F1": round(res["f1"], 4),
        "Threshold": THRESHOLD,
        "TP": res["tp"], "FP": res["fp"], "TN": res["tn"], "FN": res["fn"],
        "비고": note,
    })

result_df = pd.DataFrame(rows)
save_path = os.path.join(OUT_DIR, "exp1_hn_to_gongdan_validation.csv")
result_df.to_csv(save_path, index=False, encoding="utf-8-sig")
print(f"\n  결과 저장: {save_path}")
print("\n완료!")
