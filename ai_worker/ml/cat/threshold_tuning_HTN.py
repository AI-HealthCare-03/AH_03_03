"""
고혈압유병 예측 — Threshold 튜닝
Python 3.9 | scikit-learn>=1.4 | pandas>=2.2
OOF 확률값 기준으로 최적 threshold 탐색
목적: 스크리닝 서비스 → Recall 최대화하면서 F1 균형 유지
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    classification_report, confusion_matrix,
)
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'AppleGothic'  # Mac 한글 폰트
plt.rcParams['axes.unicode_minus'] = False

# ── 경로 설정 ─────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OOF_DIR    = os.path.join(BASE_DIR, 'outputs', 'tuned_catboost_HTN_50')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'threshold_tuning_HTN')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── OOF 확률값 로드 ───────────────────────────────────────────
y_true     = np.load(os.path.join(OOF_DIR, 'oof_y_true.npy'))
oof_proba  = np.load(os.path.join(OOF_DIR, 'oof_proba.npy'))

print(f"[0] OOF 로드 완료 | 샘플 수: {len(y_true)}")
print(f"    양성(1): {y_true.sum():.0f} / 음성(0): {(1-y_true).sum():.0f}")

# ── Threshold 탐색 (0.1 ~ 0.9) ───────────────────────────────
thresholds = np.arange(0.1, 0.91, 0.01)
results = []

for thr in thresholds:
    pred = (oof_proba >= thr).astype(int)
    results.append({
        'threshold':  round(thr, 2),
        'recall':     recall_score(y_true, pred),
        'precision':  precision_score(y_true, pred, zero_division=0),
        'f1':         f1_score(y_true, pred, zero_division=0),
        'auc':        roc_auc_score(y_true, oof_proba),  # threshold 무관
        'tn':         confusion_matrix(y_true, pred)[0, 0],
        'fp':         confusion_matrix(y_true, pred)[0, 1],
        'fn':         confusion_matrix(y_true, pred)[1, 0],
        'tp':         confusion_matrix(y_true, pred)[1, 1],
    })

results_df = pd.DataFrame(results)

print(f"\n[1] Threshold별 성능 요약")
print(results_df[['threshold', 'recall', 'precision', 'f1', 'fn', 'fp']].to_string(index=False))

# ── 최적 Threshold 선정 ───────────────────────────────────────
# 스크리닝 목적: Recall >= 0.85 조건에서 F1 최대
recall_condition = results_df[results_df['recall'] >= 0.85]

if len(recall_condition) > 0:
    best_row = recall_condition.loc[recall_condition['f1'].idxmax()]
    print(f"\n[2] 최적 Threshold (Recall >= 0.85 조건에서 F1 최대)")
else:
    # 조건 완화: Recall 최대
    best_row = results_df.loc[results_df['recall'].idxmax()]
    print(f"\n[2] 최적 Threshold (Recall 최대 기준)")

print(f"    Threshold : {best_row['threshold']}")
print(f"    Recall    : {best_row['recall']:.4f}")
print(f"    Precision : {best_row['precision']:.4f}")
print(f"    F1        : {best_row['f1']:.4f}")
print(f"    AUC-ROC   : {best_row['auc']:.4f}")
print(f"    TN={best_row['tn']:.0f} FP={best_row['fp']:.0f} FN={best_row['fn']:.0f} TP={best_row['tp']:.0f}")

# ── 베이스라인(0.5) vs 튜닝 후(0.5) vs 최적 threshold 비교 ──
print(f"\n[3] 전체 비교")
print(f"    {'구분':<20} {'Threshold':>10} {'Recall':>8} {'Precision':>10} {'F1':>8} {'FN':>6} {'FP':>6}")
print("    " + "-" * 68)

# 베이스라인 0.5
base_pred = (oof_proba >= 0.5).astype(int)
print(f"    {'튜닝후 (threshold=0.5)':<20} {0.5:>10.2f} "
      f"{recall_score(y_true, base_pred):>8.4f} "
      f"{precision_score(y_true, base_pred):>10.4f} "
      f"{f1_score(y_true, base_pred):>8.4f} "
      f"{confusion_matrix(y_true, base_pred)[1,0]:>6.0f} "
      f"{confusion_matrix(y_true, base_pred)[0,1]:>6.0f}")

# 최적 threshold
best_pred = (oof_proba >= best_row['threshold']).astype(int)
print(f"    {'최적 threshold':<20} {best_row['threshold']:>10.2f} "
      f"{recall_score(y_true, best_pred):>8.4f} "
      f"{precision_score(y_true, best_pred):>10.4f} "
      f"{f1_score(y_true, best_pred):>8.4f} "
      f"{confusion_matrix(y_true, best_pred)[1,0]:>6.0f} "
      f"{confusion_matrix(y_true, best_pred)[0,1]:>6.0f}")

print(f"\n[4] Classification Report (최적 threshold={best_row['threshold']})")
print(classification_report(y_true, best_pred, target_names=['정상(0)', '고혈압(1)']))

# ── 시각화 ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Recall / Precision / F1 vs Threshold
axes[0].plot(results_df['threshold'], results_df['recall'],    label='Recall',    color='#e74c3c', linewidth=2)
axes[0].plot(results_df['threshold'], results_df['precision'], label='Precision', color='#3498db', linewidth=2)
axes[0].plot(results_df['threshold'], results_df['f1'],        label='F1',        color='#2ecc71', linewidth=2)
axes[0].axvline(x=best_row['threshold'], color='gray', linestyle='--', alpha=0.7, label=f"Best={best_row['threshold']}")
axes[0].axvline(x=0.5, color='black', linestyle=':', alpha=0.5, label='Default=0.5')
axes[0].set_xlabel('Threshold')
axes[0].set_ylabel('Score')
axes[0].set_title('Recall / Precision / F1 vs Threshold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# FN / FP vs Threshold
axes[1].plot(results_df['threshold'], results_df['fn'], label='FN (놓친 환자)', color='#e74c3c', linewidth=2)
axes[1].plot(results_df['threshold'], results_df['fp'], label='FP (과탐지)',    color='#3498db', linewidth=2)
axes[1].axvline(x=best_row['threshold'], color='gray', linestyle='--', alpha=0.7, label=f"Best={best_row['threshold']}")
axes[1].axvline(x=0.5, color='black', linestyle=':', alpha=0.5, label='Default=0.5')
axes[1].set_xlabel('Threshold')
axes[1].set_ylabel('Count')
axes[1].set_title('FN / FP vs Threshold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'threshold_tuning.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[5] 그래프 저장 완료")

# ── 결과 저장 ─────────────────────────────────────────────────
results_df.to_csv(os.path.join(OUTPUT_DIR, 'threshold_results.csv'), index=False)

best_result = {
    'threshold':       best_row['threshold'],
    'recall':          round(best_row['recall'], 4),
    'precision':       round(best_row['precision'], 4),
    'f1':              round(best_row['f1'], 4),
    'auc':             round(best_row['auc'], 4),
    'fn':              int(best_row['fn']),
    'fp':              int(best_row['fp']),
}
pd.DataFrame([best_result]).to_csv(os.path.join(OUTPUT_DIR, 'best_threshold.csv'), index=False)

print(f"[6] 저장 완료 → {OUTPUT_DIR}")
print(f"\n★ 최종 채택 Threshold: {best_row['threshold']}")
