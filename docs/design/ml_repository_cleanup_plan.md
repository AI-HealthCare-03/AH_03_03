# ML Repository Cleanup Plan

작성 기준: `feature/kdu-ml`

이 문서는 ML 폴더를 서비스 추론 가능한 최소 구조로 정리하기 위한 분류 및 이동 기준 문서입니다. 실험용/legacy 파일은 삭제하지 않고 `experiment/ml/` 아래로 보존 이동하며, 서비스에서 직접 import하는 최소 코드만 `ai_runtime/ml/` 아래에 둡니다.

## 전체 결론

- 서비스 코드에서 직접 import되는 경로는 `ai_runtime/ml/common`, `ai_runtime/ml/datasets`, `ai_runtime/ml/training`, `ai_runtime/ml/inference`입니다.
- 최종 CatBoost 재현용 config와 학습 스크립트는 `ai_runtime/ml/experiments/configs`와 `ai_runtime/ml/training`에 모읍니다.
- 실제 모델 artifact(`.cbm`, `.pkl`, `.joblib`, `.npy`)와 대용량 산출물은 git에 커밋하지 않고 `ai_runtime/ml/artifacts` 또는 외부 스토리지에 둡니다.
- 기존 노트북과 legacy 스크립트는 지식 자산이지만 서비스 import 영역에는 과합니다. 현재는 `experiment/ml/` 아래로 보존 이동하고, 장기적으로 별도 archive 저장소, Notion 링크, Drive/S3, 또는 `ML-archive` repo로 이전하는 것을 권장합니다.
- `OBESITY`는 현재 최종 CatBoost 모델이 없으므로 rule 기반 fallback으로 유지합니다.

## 유지 대상

| 경로 | 분류 | 유지 이유 |
| --- | --- | --- |
| `ai_runtime/ml/common/` | 유지 | 공통 feature engineering, metric, threshold, artifact 저장 유틸입니다. 학습/추론 양쪽에서 사용합니다. |
| `ai_runtime/ml/datasets/` | 유지 | dataset registry와 loader입니다. config 기반 학습 재현에 필요합니다. |
| `ai_runtime/ml/training/` | 유지 | 최종 모델 재현용 최소 학습 진입점입니다. |
| `ai_runtime/ml/inference/` | 유지 | FastAPI 분석 서비스에서 optional import하는 서비스 추론 adapter입니다. |
| `ai_runtime/ml/experiments/configs/*_catboost_final.json` | 유지 | DM/HTN/DL 최종 CatBoost 학습 재현 config입니다. |
| `ai_runtime/ml/X2/` | 유지 | ML artifact 부재 시 사용할 룰 기반 fallback 후보입니다. |
| `ai_runtime/ml/artifacts/.gitkeep` | 유지 | artifact 디렉터리 placeholder입니다. 실제 artifact는 gitignore 대상입니다. |
| `experiment/ml/ai_worker/ml/final_models/` | 보존 | 기존 Optuna 최종 학습 스크립트 reference입니다. 서비스 import 대상이 아니므로 archive 영역에 보존합니다. |
| `experiment/ml/ML/models/final/` | 보존 | 팀원이 정리한 최종 노트북 reference입니다. 서비스 import 영역 밖에서 보존합니다. |
| `docs/pipeline/ml_reports/` | 보존 | 모델별 최종 보고서 문서입니다. README/발표/검증 근거로 사용 가능합니다. |
| `experiment/ml/ML/features/*.py` | 보존 | 노트북에서 분리된 feature engineering 실험 코드입니다. `ai_runtime/ml/common/features.py`와 통합 가능성을 확인한 뒤 정리합니다. |

## 삭제 또는 외부 보관 후보

| 경로/패턴 | 권장 조치 | 이유 |
| --- | --- | --- |
| `experiment/ml/ai_worker/ml/legacy/` | 보존 이동 완료 | 오래된 CatBoost/LGB/XGB 실험 코드가 많고 서비스 import 대상이 아닙니다. |
| `experiment/ml/ai_worker/ml/Clustering/` | 보존 이동 완료 | 현재 서비스 분석/추론 경로와 연결되지 않은 군집 실험입니다. |
| `experiment/ml/ai_worker/ml/CAT15~24/` | 보존 이동 완료 | 새 config 기반 pipeline과 중복되는 과거 학습 스크립트입니다. |
| `experiment/ml/ai_worker/ml/CAT18~24/` | 보존 이동 완료 | 최종 config 근거이지만 서비스 import 영역에는 reference 이상으로 과합니다. |
| `experiment/ml/ai_worker/ml/LGB15~24/` | 보존 이동 완료 | CatBoost 최종 모델 채택 후 서비스 경로에 쓰이지 않습니다. |
| `experiment/ml/ai_worker/ml/LogisticR/` | 보존 이동 완료 | baseline 실험 코드입니다. 최종 서비스 추론에는 불필요합니다. |
| `experiment/ml/ai_worker/ml/files/` | 보존 이동 완료 | preprocessing/EDA 자료입니다. zip 중간 산출물은 장기적으로 외부 artifact storage 권장입니다. |
| `experiment/ml/ML/models/DL/*.ipynb` | 보존 이동 완료 | DL 실험 노트북입니다. |
| `experiment/ml/ML/models/DM/*.ipynb` | 보존 이동 완료 | DM 실험 노트북입니다. |
| `experiment/ml/ML/models/HTN/*.ipynb` | 보존 이동 완료 | HTN 실험 노트북입니다. |
| `experiment/ml/ML/models/baseline/*.ipynb` | 보존 이동 완료 | baseline notebook입니다. |
| `experiment/ml/ML/baseline_model/*.ipynb` | 보존 이동 완료 | baseline notebook입니다. |
| `experiment/ml/ML/final(seed)/*.ipynb` | 보존 이동 완료 | seed ensemble 최종 산출물일 가능성이 있어 서비스 import 영역 밖에서 보존합니다. |
| `test_claude.py` | 삭제 후보 | 서비스 코드/학습 재현 코드와 무관한 임시 테스트 파일로 보입니다. |
| `.DS_Store` | 삭제 후보 | OS 메타데이터입니다. |
| `__pycache__/` | 삭제 후보 | Python 실행 캐시입니다. gitignore 대상이며 repo에 남길 필요가 없습니다. |
| `ai_runtime/data/*.csv` | 외부 보관 후보 | 학습 데이터는 대용량/민감 가능성이 있어 서비스 repo에서 제거하는 것이 바람직합니다. |
| `*.cbm`, `*.pkl`, `*.joblib`, `*.npy` | gitignore 유지 | 모델 artifact와 중간 산출물은 외부 artifact storage 또는 로컬 artifacts 폴더에 둡니다. |

## 애매한 파일 목록

| 경로 | 확인할 점 | 임시 판단 |
| --- | --- | --- |
| `ML/models/final/cat_v3.3_HTN_alldata_fe.ipynb` | HTN 최종 채택 노트북인지 확인 | 유지 후보 |
| `ML/models/final/cat_v4.3_DM_threshold.ipynb` | DM 최종 threshold 노트북인지 확인 | 유지 후보 |
| `ML/models/final/xgb_v4.3_DL_alldata_fe.ipynb` | DL 최종 모델이 CatBoost인지 XGBoost인지 정책 확인 | 유지 후보 |
| `ML/final(seed)/CatBoost_SeedEnsemble_DM_FE.ipynb` | seed ensemble이 최종 모델인지 확인 | 외부 보관 후보 |
| `ML/final(seed)/CatBoost_SeedEnsemble_HTN_FE.ipynb` | seed ensemble이 최종 모델인지 확인 | 외부 보관 후보 |
| `ML/final(seed)/CatBoost_SeedEnsemble_DL_FE.ipynb` | seed ensemble이 최종 모델인지 확인 | 외부 보관 후보 |
| `ML/features/*.py` | 새 `ai_runtime/ml/common/features.py`와 중복 여부 확인 | 통합 후 보관/삭제 |
| `experiment/ml/ai_worker/ml/db_logger.py` | legacy 내부 실험 스크립트에서만 사용 | 보존 이동 완료 |
| `ai_runtime/ml/experiments/configs/*_final.json` | 구 config와 새 `*_catboost_final.json` 중 어떤 것을 공식으로 둘지 결정 | 새 config로 통일 권장 |

## 현재 서비스 import 기준

현재 FastAPI 분석 서비스는 `app/services/analysis.py`에서 `ai_runtime.ml.inference.disease_risk_service`를 optional import합니다. 따라서 아래 파일은 삭제하면 안 됩니다.

- `ai_runtime/ml/inference/disease_risk_service.py`
- `ai_runtime/ml/inference/catboost_predictor.py`
- `ai_runtime/ml/inference/feature_mapper.py`
- `ai_runtime/ml/inference/schemas.py`
- `ai_runtime/ml/common/artifacts.py`
- `ai_runtime/ml/datasets/registry.py`

학습 재현 명령에서 사용하는 파일도 유지합니다.

- `ai_runtime/ml/training/run_experiment.py`
- `ai_runtime/ml/training/train_catboost.py`
- `ai_runtime/ml/training/train_xgboost.py`
- `ai_runtime/ml/datasets/loaders.py`

## .gitignore 보강 기준

다음 항목은 git에 커밋하지 않습니다.

- `.DS_Store`
- `node_modules/`
- `dist/`
- `.vite/`
- `*.cbm`
- `*.pkl`
- `*.joblib`
- `*.npy`
- `ai_runtime/ml/artifacts/*`
- `ML/artifacts/*`
- 대용량/중간 CSV

예외:

- `docs/data/challenges/*.csv`: 챌린지 master 문서성 CSV
- `ML/models/report/*.csv`: 최종 모델 리포트에서 필요한 문서성 CSV가 생길 경우만 허용

## 이동 후 현재 구조

- 서비스 import 영역: `ai_runtime/ml/X2`, `ai_runtime/ml/common`, `ai_runtime/ml/datasets`, `ai_runtime/ml/training`, `ai_runtime/ml/inference`, `ai_runtime/ml/experiments/configs`, `ai_runtime/ml/artifacts/.gitkeep`
- 보존 archive 영역: `experiment/ml/ML`, `experiment/ml/ai_worker/ml/* legacy/experiment folders`, `experiment/ml/ai_worker/data`
- 학습 dataset registry는 `experiment/ml/ai_worker/data`를 기본 위치로 봅니다.
- `experiment/ml/ai_worker/data`는 로컬 학습 재현용 데이터 배치 경로이며 서비스 런타임 import 대상이 아닙니다.
- CSV/전처리 데이터는 민감/대용량 가능성이 있어 git 추적에서 제외하고, 필요 시 `git rm --cached experiment/ml/ai_worker/data/*` 후 별도 archive 저장소에서 배포하는 방향을 권장합니다.

## 다음 작업 순서

1. 팀 기준 최종 모델을 `CatBoost DM/HTN/DL`로 확정합니다.
2. `ML/models/final`과 `ML/final(seed)` 중 공식 reference만 남길지 결정합니다.
3. 외부 보관 위치를 정합니다.
   - 예: Google Drive, S3, Hugging Face private repo, 별도 `ML-archive` repo
4. 삭제 PR 전에 아래 목록을 다시 출력합니다.
   - 삭제 후보
   - 외부 보관 완료 여부
   - 서비스 import 영향 여부
5. 별도 cleanup PR에서 OS/cache/legacy 파일을 제거합니다.

## 정리 후 체크리스트

- `rg -n "ai_runtime.ml.legacy|ai_runtime.ml.CAT|ai_runtime.ml.Clustering|from ML|import ML" app ai_runtime scripts docs --glob '!**/*.ipynb'`
- `uv run ruff check app scripts ai_runtime`
- `uv run ruff format app scripts ai_runtime --check`
- `uv run python -c "from app.main import app; print(app.title); print(len(app.openapi().get('paths', {})))"`
- `uv run python -m ai_runtime.ml.training.run_experiment --config ai_runtime/ml/experiments/configs/dm_catboost_final.json --dry-run --sample-size 50`
