"""
실험 로그 저장 — model_versions + analysis_snapshots
Python 3.9 | sqlalchemy>=2.0 | psycopg2-binary
ERD 기준: model_versions, analysis_snapshots
"""

import datetime
import json
import os

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# ── DB 연결 설정 (환경변수로 관리) ───────────────────────────
# .env 또는 시스템 환경변수에 설정
# export DB_HOST=localhost
# export DB_PORT=5432
# export DB_NAME=chronic_health
# export DB_USER=postgres
# export DB_PASSWORD=yourpassword

DB_URL = "postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}".format(
    user     = os.environ.get('DB_USER',     'myuser'),
    password = os.environ.get('DB_PASSWORD', '05040000'),
    host     = os.environ.get('DB_HOST',     'localhost'),
    port     = os.environ.get('DB_PORT',     '5433'),   # ← 5433으로
    dbname   = os.environ.get('DB_NAME',     'mydb'),
)

engine = create_engine(DB_URL, echo=False)


# ──────────────────────────────────────────────────────────────
# 1. model_versions 저장
#    학습 완료 후 모델 성능 지표 + 메타정보 기록
# ──────────────────────────────────────────────────────────────
def save_model_version(
    model_name: str,        # "hypertension_model", "diabetes_model", "dyslipidemia_model"
    model_type: str,        # "ml", "rule", "hybrid"
    version: str,           # "v1.0", "v1.1" 등
    metric_summary: dict,   # {"auc": 0.8585, "f1": 0.6472, "recall": 0.8298, ...}
    file_path: str = None,  # 모델 파일 경로 (선택)
    is_active: bool = True,
) -> int:
    """
    model_versions 테이블에 모델 버전 저장 후 id 반환
    """
    sql = text("""
        INSERT INTO model_versions
            (model_name, model_type, version, file_path, metric_summary, is_active, created_at, updated_at)
        VALUES
            (:model_name, :model_type, :version, :file_path, :metric_summary, :is_active, :now, :now)
        RETURNING id
    """)

    with engine.begin() as conn:
        result = conn.execute(sql, {
            'model_name':     model_name,
            'model_type':     model_type,
            'version':        version,
            'file_path':      file_path,
            'metric_summary': json.dumps(metric_summary, ensure_ascii=False),
            'is_active':      is_active,
            'now':            datetime.datetime.now(),
        })
        model_version_id = result.fetchone()[0]

    print(f"[model_versions] 저장 완료 | id={model_version_id} | {model_name} {version}")
    return model_version_id


# ──────────────────────────────────────────────────────────────
# 2. analysis_snapshots 저장
#    실험마다 입력 피처 + 모델 출력값 + SHAP 로깅
# ──────────────────────────────────────────────────────────────
def save_analysis_snapshot(
    user_id: int,
    health_record_id: int,
    input_features: dict,           # X 컬럼값 (한 row)
    model_outputs: dict,            # {"hypertension_proba": 0.82, "risk_level": "high"}
    # shap_outputs: dict = None,    # ❌ ERD ANALYSIS_SNAPSHOTS에 없는 컬럼 — 추후 ERD 추가 시 복구
    rule_outputs: dict = None,      # 룰 엔진 결과 (선택)
    final_outputs: dict = None,     # 후처리 최종 결과 (선택)
    model_version_info: dict = None,# {"model_name": ..., "version": ...}
    threshold_version: str = None,  # "v1.0"
    async_job_id: int = None,
    # analysis_result_id: int = None,  # ❌ ERD ANALYSIS_SNAPSHOTS에 없는 컬럼 — 추후 ERD 추가 시 복구
) -> int:
    """
    analysis_snapshots 테이블에 실험 스냅샷 저장 후 id 반환
    """
    sql = text("""
        INSERT INTO analysis_snapshots
            (async_job_id, user_id, health_record_id,
             input_features, model_outputs, rule_outputs, final_outputs,
             model_version_info, threshold_version, snapshot_at, created_at, updated_at)
        VALUES
            (:async_job_id, :user_id, :health_record_id,
             :input_features, :model_outputs, :rule_outputs, :final_outputs,
             :model_version_info, :threshold_version, :now, :now, :now)
        RETURNING id
    """)

    with engine.begin() as conn:
        result = conn.execute(sql, {
            'async_job_id':        async_job_id,
            # 'analysis_result_id':  analysis_result_id,  # ❌ ERD에 없는 컬럼
            'user_id':             user_id,
            'health_record_id':    health_record_id,
            'input_features':      json.dumps(input_features,    ensure_ascii=False, default=_json_safe),
            'model_outputs':       json.dumps(model_outputs,     ensure_ascii=False, default=_json_safe),
            # 'shap_outputs':        json.dumps(shap_outputs, ...) if shap_outputs else None,  # ❌ ERD에 없는 컬럼
            'rule_outputs':        json.dumps(rule_outputs,      ensure_ascii=False, default=_json_safe) if rule_outputs else None,
            'final_outputs':       json.dumps(final_outputs,     ensure_ascii=False, default=_json_safe) if final_outputs else None,
            'model_version_info':  json.dumps(model_version_info,ensure_ascii=False, default=_json_safe) if model_version_info else None,
            'threshold_version':   threshold_version,
            'now':                 datetime.datetime.now(),
        })
        snapshot_id = result.fetchone()[0]

    print(f"[analysis_snapshots] 저장 완료 | id={snapshot_id} | user_id={user_id}")
    return snapshot_id


# ──────────────────────────────────────────────────────────────
# 3. OOF 실험 전체 로깅 (베이스라인/튜닝 후 실험 단위)
#    fold_scores + OOF 성능 + 피처 목록을 model_versions에 한 번에 저장
# ──────────────────────────────────────────────────────────────
def log_experiment(
    model_name: str,
    version: str,
    feature_columns: list,
    oof_auc: float,
    oof_f1: float,
    oof_recall: float,
    fold_scores: list,          # [{'fold':1,'auc':...,'f1':...,'recall':...}, ...]
    best_params: dict = None,
    file_path: str = None,
    is_active: bool = True,
) -> int:
    """
    실험 단위 로그 → model_versions 저장
    리뷰 요청사항: 인풋된 것, AUC값, FE 파일을 함께 기록
    """
    metric_summary = {
        'oof_auc':    round(oof_auc, 4),
        'oof_f1':     round(oof_f1, 4),
        'oof_recall': round(oof_recall, 4),
        'fold_scores': fold_scores,
        'feature_columns': feature_columns,         # 입력 피처 목록 (FE 버전 추적용)
        'n_features': len(feature_columns),
        'best_params': best_params or {},
        'logged_at': datetime.datetime.now().isoformat(),
    }

    return save_model_version(
        model_name    = model_name,
        model_type    = 'ml',
        version       = version,
        metric_summary= metric_summary,
        file_path     = file_path,
        is_active     = is_active,
    )


def _json_safe(obj):
    """ numpy 타입 json 직렬화 처리 """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ──────────────────────────────────────────────────────────────
# 사용 예시
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # ── 예시 1: 베이스라인 실험 로그 저장 ────────────────────
    experiment_id = log_experiment(
        model_name      = 'hypertension_model',
        version         = 'v1.0-baseline',
        feature_columns = [
            '성별', '나이', '키', '체중', 'BMI', '현재흡연',
            '고혈압가족력_부', '고혈압가족력_모', '고혈압가족력_형제',
            '당뇨가족력_부', '당뇨가족력_모', '당뇨가족력_형제',
            '고지혈증가족력_부', '고지혈증가족력_모', '고지혈증가족력_형제',
            '걷기일수', '근력운동일수', '과거음주_현재금주',
            '음주빈도_enc', '음주량_enc',
            '직업_관리전문', '직업_기능노무', '직업_농림어업', '직업_무직',
            '직업_사무', '직업_서비스판매', '직업_작업미상', '직업_주부학생',
        ],
        oof_auc    = 0.8585,
        oof_f1     = 0.6472,
        oof_recall = 0.8298,
        fold_scores = [
            {'fold': 1, 'auc': 0.8502, 'f1': 0.6460, 'recall': 0.8363},
            {'fold': 2, 'auc': 0.8538, 'f1': 0.6294, 'recall': 0.8214},
            {'fold': 3, 'auc': 0.8758, 'f1': 0.6840, 'recall': 0.8631},
            {'fold': 4, 'auc': 0.8282, 'f1': 0.6080, 'recall': 0.7917},
            {'fold': 5, 'auc': 0.8855, 'f1': 0.6706, 'recall': 0.8363},
        ],
        best_params = {
            'iterations': 500, 'learning_rate': 0.05, 'depth': 6,
            'class_weights': {0: 1.0, 1: 2.5911},
        },
        file_path = '/outputs/baseline_catboost/model.cbm',
    )
    print(f"실험 저장 완료 | model_version_id: {experiment_id}")

    # ── 예시 2: 단건 예측 스냅샷 저장 ────────────────────────
    snapshot_id = save_analysis_snapshot(
        user_id          = 1,
        health_record_id = 1,
        input_features   = {
            '나이': 55, 'BMI': 27.3, '성별': 1,
            '고혈압가족력_부': 1, '고혈압가족력_모': 0,
        },
        model_outputs = {
            'hypertension_proba': 0.82,
            'risk_level': 'high',
            'threshold': 0.5,
        },
        # shap_outputs = {...},        # ❌ ERD에 없는 컬럼 — 추후 ERD 추가 시 복구
        model_version_info = {
            'model_name': 'hypertension_model',
            'version': 'v1.0-baseline',
        },
        threshold_version = 'v1.0',
        # analysis_result_id = 1,     # ❌ ERD에 없는 컬럼 — 추후 ERD 추가 시 복구
    )
    print(f"스냅샷 저장 완료 | snapshot_id: {snapshot_id}")
