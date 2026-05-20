"""
베이스라인 FE + Optuna v3 OOF proba 앙상블
HTN / DM / DL 각각 단순 평균 앙상블 후 Test 성능 비교

Python 3.13 | numpy | scikit-learn
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ── 경로 설정 ─────────────────────────────────────────────────
BASE_DIR: str = "/Users/admin/PycharmProjects/AH_03_03/ai_worker/ml/LGB18~24/outputs"
DATA_PATH: str = "/Users/admin/PycharmProjects/AH_03_03/ai_worker/data/hn_all_preprocessed.csv"
SEED: int = 42

# ── 타겟별 설정 ───────────────────────────────────────────────
TARGETS: dict = {
    "HTN": {
        "col": "고혈압유병",
        "baseline_dir": f"{BASE_DIR}/baseline_lgbm_HTN_FE",
        "optuna_dir": f"{BASE_DIR}/optuna_HTN_FE_v3",
    },
    "DM": {
        "col": "당뇨유병",
        "baseline_dir": f"{BASE_DIR}/baseline_lgbm_DM_FE",
        "optuna_dir": f"{BASE_DIR}/optuna_DM_FE_v3",
    },
    "DL": {
        "col": "이상지질혈증유병",
        "baseline_dir": f"{BASE_DIR}/baseline_lgbm_DL_FE",
        "optuna_dir": f"{BASE_DIR}/optuna_DL_FE_v3",
    },
}

THRESHOLD_RANGE: NDArray[np.float64] = np.arange(0.30, 0.71, 0.01)
RECALL_MIN: float = 0.87


def tune_threshold_f1(
    proba: NDArray[np.float64],
    y_true: NDArray[np.int64],
    recall_min: float = RECALL_MIN,
) -> tuple[float, float, float, float]:
    """Recall >= recall_min 만족 시 F1 최대 threshold 선택."""
    best_f1: float = 0.0
    best_thr: float = 0.30

    for thr in THRESHOLD_RANGE:
        pred = (proba >= thr).astype(int)
        r = recall_score(y_true, pred)
        f = f1_score(y_true, pred, zero_division=0)
        if r >= recall_min and f > best_f1:
            best_f1 = f
            best_thr = round(float(thr), 2)

    final_pred = (proba >= best_thr).astype(int)
    final_recall = recall_score(y_true, final_pred)
    final_prec = precision_score(y_true, final_pred, zero_division=0)

    return best_thr, best_f1, final_recall, final_prec


def evaluate(
    proba: NDArray[np.float64],
    y_true: NDArray[np.int64],
    thr: float,
    label: str,
) -> dict:
    pred = (proba >= thr).astype(int)
    return {
        "model": label,
        "auc": round(float(roc_auc_score(y_true, proba)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "threshold": thr,
    }


def run_ensemble(target_name: str, cfg: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"[{target_name}] 앙상블 시작")
    print(f"{'=' * 60}")

    # ── OOF proba 로드 ────────────────────────────────────────
    base_oof = np.load(f"{cfg['baseline_dir']}/oof_proba.npy")
    opt_oof = np.load(f"{cfg['optuna_dir']}/oof_proba.npy")
    y_train = np.load(f"{cfg['baseline_dir']}/oof_y_true.npy")

    # ── OOF 앙상블 proba (단순 평균) ─────────────────────────
    ens_oof = (base_oof + opt_oof) / 2.0

    # ── OOF 기반 threshold 탐색 ──────────────────────────────
    base_thr = float(np.load(f"{cfg['baseline_dir']}/best_threshold.npy")[0])
    opt_thr = float(np.load(f"{cfg['optuna_dir']}/best_threshold.npy")[0])
    ens_thr, ens_f1, ens_recall, ens_prec = tune_threshold_f1(ens_oof, y_train)

    print("\n[OOF Threshold]")
    print(f"  베이스라인 FE : {base_thr:.2f}")
    print(f"  Optuna v3    : {opt_thr:.2f}")
    print(f"  앙상블        : {ens_thr:.2f}  (OOF Recall: {ens_recall:.4f} | F1: {ens_f1:.4f})")

    # ── Test 데이터 재구성 ────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[cfg["col"]]).reset_index(drop=True)
    y = df[cfg["col"]].astype(int)
    X = df.drop(columns=[c for c in ["고혈압유병", "당뇨유병", "이상지질혈증유병", "비만단계"] if c in df.columns])

    _, _, _, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    # ── Test proba 로드 (fold 평균 저장된 버전 없으면 OOF로 대체 안내) ──
    # 각 모델의 fold model로 예측한 test proba가 없으므로
    # OOF correlation 확인 후 앙상블 방향 판단
    oof_corr = float(np.corrcoef(base_oof, opt_oof)[0, 1])
    print(f"\n[OOF Correlation] 베이스라인 FE vs Optuna v3 : {oof_corr:.4f}")
    if oof_corr > 0.95:
        print("  ⚠️  상관관계 높음 (>0.95) → 앙상블 다양성 제한적")
    elif oof_corr > 0.90:
        print("  △  상관관계 보통 (0.90~0.95) → 앙상블 효과 소폭 기대")
    else:
        print("  ✅  상관관계 낮음 (<0.90) → 앙상블 효과 기대")

    # ── Test proba는 fold 모델로 직접 예측 필요 안내 ─────────
    print("\n⚠️  Test proba는 저장된 fold 모델로 직접 예측이 필요합니다.")
    print("   아래 OOF 기준 앙상블 성능으로 방향성만 판단하세요.\n")

    # ── OOF 기준 단독 vs 앙상블 비교 ────────────────────────
    rows = [
        evaluate(base_oof, y_train, base_thr, "베이스라인 FE (OOF)"),
        evaluate(opt_oof, y_train, opt_thr, "Optuna v3 (OOF)"),
        evaluate(ens_oof, y_train, ens_thr, "앙상블 평균 (OOF)"),
    ]
    result_df = pd.DataFrame(rows)
    print("[OOF 성능 비교]")
    print(result_df.to_string(index=False))


def main() -> None:
    print("=" * 60)
    print("베이스라인 FE + Optuna v3 앙상블 분석")
    print(f"RECALL_MIN: {RECALL_MIN} | 목표함수: F1 최대화")
    print("=" * 60)

    for target_name, cfg in TARGETS.items():
        run_ensemble(target_name, cfg)

    print(f"\n{'=' * 60}")
    print("완료. OOF correlation 확인 후 앙상블 방향 결정하세요.")
    print("상관관계 < 0.90이면 Test proba 앙상블 코드로 진행.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
