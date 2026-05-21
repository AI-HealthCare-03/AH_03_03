"""
model_logger.py
───────────────
실험 결과를 SQLite DB에 자동 저장하는 로거 모듈.

저장 항목:
  1. 실험 메타데이터  : Run_ID, Created_At, Model_Name, Target_Var, Stage, Note
  2. 하이퍼파라미터   : Learning_Rate, Max_Depth, N_Estimators, Class_Weight, 기타(JSON)
  3. 데이터 전처리 정보: Feature_Count, Train_Test_Split, Scaling_Method
  4. 핵심 결과 지표   : Accuracy, Recall, Precision, F1_Score, AUC_ROC, Confusion_Matrix
  5. Feature Importance: Top_Features (JSON)

사용법:
    from model_logger import ModelLogger

    logger = ModelLogger('/Users/Jiyeon/Desktop/final_project/ML/model_result.db')

    run_id = logger.log_run(
        target_var   = '고혈압',
        model_name   = 'XGBoost',
        stage        = 'baseline',
        hyperparams  = {
            'learning_rate': 0.05,
            'max_depth': 6,
            'n_estimators': 500,
            'class_weight': {0: 1.0, 1: 2.59},
        },
        data_info    = {
            'feature_count': 28,
            'train_test_split': '5-Fold CV',
            'scaling_method': 'None',
        },
        oof_metrics  = {
            'accuracy':  0.7509,
            'recall':    0.8155,
            'precision': 0.5345,
            'f1_score':  0.6458,
            'auc_roc':   0.8531,
            'cm': [[3160, 1193], [310, 1370]],   # [[TN,FP],[FN,TP]]
        },
        fold_scores        = [...],   # fold별 상세 결과 리스트 (선택)
        threshold_results  = [...],   # threshold 분석 결과 리스트 (선택)
        top_features       = {'나이': 37.7, 'BMI': 9.0, ...},  # 상위 피처 (선택)
        note = '베이스라인. threshold=0.5 기본값.',
    )
"""

import json
import os
import sqlite3
from datetime import datetime


class ModelLogger:
    def __init__(self, db_path: str = None):
        if db_path is None:
            os.makedirs("logs", exist_ok=True)
            db_path = os.path.join("logs", "model_results.db")
        self.db_path = db_path
        self._init_db()

    # ── DB 초기화 ────────────────────────────────────────────
    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                -- 1. 실험 메타 + 하이퍼파라미터 + 데이터 정보 + 핵심 지표
                CREATE TABLE IF NOT EXISTS runs (
                    run_id            INTEGER PRIMARY KEY AUTOINCREMENT,

                    -- 실험 메타데이터
                    created_at        TEXT    NOT NULL,
                    model_name        TEXT    NOT NULL,
                    target_var        TEXT    NOT NULL,
                    stage             TEXT    NOT NULL DEFAULT 'baseline',
                    note              TEXT,

                    -- 하이퍼파라미터
                    learning_rate     REAL,
                    max_depth         INTEGER,
                    n_estimators      INTEGER,
                    class_weight      TEXT,    -- JSON 문자열
                    extra_params      TEXT,    -- 기타 파라미터 JSON

                    -- 데이터 전처리 정보
                    feature_count     INTEGER,
                    train_test_split  TEXT,
                    scaling_method    TEXT,

                    -- 핵심 결과 지표
                    accuracy          REAL,
                    recall            REAL,
                    precision         REAL,
                    f1_score          REAL,
                    auc_roc           REAL,
                    confusion_matrix  TEXT,    -- JSON [[TN,FP],[FN,TP]]

                    -- Feature Importance Top (JSON)
                    top_features      TEXT
                );

                -- 2. Fold별 상세 결과
                CREATE TABLE IF NOT EXISTS fold_scores (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id    INTEGER NOT NULL REFERENCES runs(run_id),
                    fold      INTEGER NOT NULL,
                    auc       REAL,
                    f1        REAL,
                    recall    REAL,
                    precision REAL,
                    fp        INTEGER,
                    best_iter INTEGER
                );

                -- 3. Threshold 분석 결과
                CREATE TABLE IF NOT EXISTS threshold_results (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id    INTEGER NOT NULL REFERENCES runs(run_id),
                    threshold REAL,
                    recall    REAL,
                    precision REAL,
                    f1        REAL,
                    fp        INTEGER
                );
            """)

    def _conn(self):
        return sqlite3.connect(self.db_path)

    # ── 실험 저장 ────────────────────────────────────────────
    def log_run(
        self,
        target_var: str,
        model_name: str,
        stage: str = "baseline",
        hyperparams: dict = None,
        data_info: dict = None,
        oof_metrics: dict = None,
        fold_scores: list = None,
        threshold_results=None,
        top_features=None,
        note: str = None,
    ) -> int:
        hp = hyperparams or {}
        di = data_info or {}
        om = oof_metrics or {}

        # class_weight / extra_params 분리
        known_keys = {"learning_rate", "max_depth", "n_estimators", "class_weight"}
        extra_params = {k: v for k, v in hp.items() if k not in known_keys}

        # confusion matrix
        cm = om.get("cm", om.get("confusion_matrix", None))

        # top_features 정리 (Series / dict 모두 수용)
        if top_features is not None:
            if hasattr(top_features, "to_dict"):
                top_features = top_features.to_dict()
            top_features_json = json.dumps(
                dict(sorted(top_features.items(), key=lambda x: x[1], reverse=True)), ensure_ascii=False
            )
        else:
            top_features_json = None

        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO runs (
                    created_at, model_name, target_var, stage, note,
                    learning_rate, max_depth, n_estimators, class_weight, extra_params,
                    feature_count, train_test_split, scaling_method,
                    accuracy, recall, precision, f1_score, auc_roc,
                    confusion_matrix, top_features
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    model_name,
                    target_var,
                    stage,
                    note,
                    hp.get("learning_rate"),
                    hp.get("max_depth"),
                    hp.get("n_estimators"),
                    json.dumps(hp.get("class_weight"), ensure_ascii=False)
                    if hp.get("class_weight") is not None
                    else None,
                    json.dumps(extra_params, ensure_ascii=False) if extra_params else None,
                    di.get("feature_count"),
                    di.get("train_test_split"),
                    di.get("scaling_method"),
                    om.get("accuracy"),
                    om.get("recall"),
                    om.get("precision"),
                    om.get("f1_score"),
                    om.get("auc_roc"),
                    json.dumps(cm, ensure_ascii=False) if cm is not None else None,
                    top_features_json,
                ),
            )
            run_id = cur.lastrowid

            # fold scores
            if fold_scores:
                conn.executemany(
                    """INSERT INTO fold_scores
                       (run_id, fold, auc, f1, recall, precision, fp, best_iter)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    [
                        (
                            run_id,
                            s.get("fold"),
                            s.get("auc"),
                            s.get("f1"),
                            s.get("recall"),
                            s.get("precision"),
                            s.get("fp"),
                            s.get("best_iter"),
                        )
                        for s in fold_scores
                    ],
                )

            # threshold results
            if threshold_results is not None:
                rows = (
                    threshold_results.to_dict("records") if hasattr(threshold_results, "to_dict") else threshold_results
                )
                conn.executemany(
                    """INSERT INTO threshold_results
                       (run_id, threshold, recall, precision, f1, fp)
                       VALUES (?,?,?,?,?,?)""",
                    [(run_id, r["threshold"], r["recall"], r["precision"], r["f1"], r["fp"]) for r in rows],
                )

        print(f"[ModelLogger] 저장 완료 | run_id={run_id} | {target_var} {model_name} ({stage})")
        return run_id

    # ── 조회 헬퍼 ────────────────────────────────────────────
    def get_runs(self):
        """전체 실험 목록 DataFrame 반환"""
        import pandas as pd

        with self._conn() as conn:
            return pd.read_sql("SELECT * FROM runs ORDER BY run_id DESC", conn)

    def get_fold_scores(self, run_id: int):
        import pandas as pd

        with self._conn() as conn:
            return pd.read_sql("SELECT * FROM fold_scores WHERE run_id=? ORDER BY fold", conn, params=(run_id,))

    def get_threshold_results(self, run_id: int):
        import pandas as pd

        with self._conn() as conn:
            return pd.read_sql(
                "SELECT * FROM threshold_results WHERE run_id=? ORDER BY threshold", conn, params=(run_id,)
            )

    def compare_runs(self, run_ids: list = None):
        """여러 run 성능 비교 DataFrame 반환"""
        import pandas as pd

        with self._conn() as conn:
            if run_ids:
                placeholders = ",".join("?" * len(run_ids))
                return pd.read_sql(
                    f"""SELECT run_id, created_at, target_var, model_name, stage,
                               auc_roc, recall, precision, f1_score, accuracy
                        FROM runs WHERE run_id IN ({placeholders}) ORDER BY run_id""",
                    conn,
                    params=run_ids,
                )
            return pd.read_sql(
                """SELECT run_id, created_at, target_var, model_name, stage,
                          auc_roc, recall, precision, f1_score, accuracy
                   FROM runs ORDER BY run_id DESC""",
                conn,
            )
