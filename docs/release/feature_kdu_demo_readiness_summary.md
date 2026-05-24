# feature/kdu Demo Readiness Summary

이 문서는 `origin/feature/kdu` 이후 현재 `feature/kdu` 브랜치에서 진행된 시연 준비 작업을 리뷰하기 쉽게 정리한 요약본이다. 커밋된 변경과 현재 워크트리에 남아 있는 추가 변경이 섞여 있으므로, 리뷰 시에는 `git status --short`로 최종 커밋 대기 파일을 함께 확인한다.

## 1. 기준

- 기준 브랜치: `origin/feature/kdu`
- 현재 브랜치: `feature/kdu`
- 현재 OpenAPI path 수: `106`
- 주요 검증 기준:
  - `uv run ruff check app scripts ai_worker tests`
  - `uv run ruff format app scripts ai_worker tests --check`
  - `uv run pytest tests`
  - `uv run python -c "from app.main import app; print(app.title); print(len(app.openapi().get('paths', {})))"`

## 2. origin 이후 커밋 요약

`git log --oneline origin/feature/kdu..HEAD` 기준 주요 커밋:

- `82caed0` Docker Compose Redis 헬스체크와 시연 데이터 검증 보강
- `4e0aff1` 시연 전 통합 검증과 보안 운영 문서 정리
- `518d137` 정밀분석 CatBoost 스모크 검증 스크립트 추가
- `c8ecc7d` CV food 분석 결과 스키마 추가
- `bc78848` ML feature mapping 검증 보강
- `9ef0c6e` 공식 분석 API 응답과 서비스 명칭 정리
- `e9a3e8c` Docker Compose FastAPI 실행 명령에 no-sync 적용
- `afab92b` FastAPI Docker 이미지에서 ML 런타임 venv 경로 보정
- `d0e603b` FastAPI 컨테이너 ML 런타임 의존성 추가
- `b4bcc64` 식단 질병군별 영양 점수화 연결
- `74ccaa9` 건강검진 OCR 결과를 정밀분석 입력값에 반영
- `f8db126` dummy 분석 경로를 공식 서비스 경로로 정리
- `0fa6edb` GPT Vision provider를 CV provider 하위로 이동
- `9d89d3f` 음식 영양 점수화 모듈을 CV food 하위로 이동

## 3. Diff 규모

`git diff --stat origin/feature/kdu..HEAD` 기준:

- 48 files changed
- 약 2,621 insertions
- 약 180 deletions

현재 워크트리까지 포함한 `git diff --stat origin/feature/kdu` 기준:

- 66 files changed
- 약 3,314 insertions
- 약 253 deletions

현재 워크트리에는 추가 문서, QA 스크립트, LLM explanation, OCR medication skeleton, CV fallback policy 등 미커밋 변경도 존재한다.

## 4. 변경 영역별 요약

### 4.1 Docker ML runtime

해결 상태: 해결됨

주요 변경:

- FastAPI Docker image에서 `ai` dependency group을 함께 설치하도록 보강했다.
- `ai_worker` 코드가 컨테이너에 포함되도록 Dockerfile을 조정했다.
- CatBoost artifact가 Docker build context에서 빠지지 않도록 `.dockerignore` 정책을 보강했다.
- Docker compose 실행 시 `uv run --no-sync`를 사용해 빌드된 `.venv`를 안정적으로 사용하게 했다.

검증 목적:

- 컨테이너 내부에서 `catboost`, `argon2`, `ai_worker.ml.inference.disease_risk_service` import가 가능해야 한다.
- `/analysis/run` PRECISION 경로에서 CatBoost artifact를 사용할 수 있어야 한다.

### 4.2 Redis health

해결 상태: 해결됨

주요 변경:

- 앱이 실제로 읽는 Redis 설정 키를 확인하고 compose 환경에 맞게 정리했다.
- Docker compose에서 FastAPI 컨테이너는 healthy이며, `/api/v1/system/health`에서 Redis health degraded가 발생하지 않도록 보강했다.

남은 확인:

- 시연 직전 `docker compose ps`와 `/api/v1/system/health`를 한 번 더 확인한다.

### 4.3 Precision CatBoost E2E

해결 상태: 해결됨

주요 변경:

- `scripts/verify_precision_analysis_api.py`를 추가했다.
- 로그인 -> readiness -> `/analysis/run` PRECISION -> 결과 상세 조회 흐름을 검증한다.
- DM/HTN/DL은 `model_name=catboost`, `model_version=*_catboost_final`, `risk_score` 존재 여부를 확인한다.
- OBESITY는 현재 ML artifact가 없으므로 `rule_based`를 정상 경로로 검증한다.

관련 범위:

- `app/services/analysis.py`
- `ai_worker/ml/inference/disease_risk_service.py`
- `ai_worker/ml/inference/catboost_predictor.py`
- `scripts/verify_precision_analysis_api.py`

### 4.4 OCR confirm -> HealthRecord X2

해결 상태: 해결됨

주요 변경:

- 건강검진 OCR 결과가 `ExamMeasurement`에 저장된 뒤 confirm 시 `HealthRecord` X2 필드에 반영되도록 보강했다.
- `ldl`, `hdl` 같은 짧은 OCR key도 `ldl_cholesterol`, `hdl_cholesterol`로 매핑한다.
- `"130 mg/dL"` 같은 문자열 값을 숫자로 변환한다.
- confirm 이후 precision readiness 판단에 필요한 검진 필드가 채워질 수 있다.

테스트:

- `tests/exams/test_exam_confirm_to_health_record.py`

### 4.5 Diet nutrition scorer

해결 상태: 해결됨

주요 변경:

- 음식 영양 점수화 모듈을 `ai_worker/cv/food/nutrition/` 하위로 정리했다.
- 런타임은 원본 Excel을 직접 읽지 않고 `food_disease_scores.csv`와 `disease_score_rules.json`을 읽는다.
- `/diets/analyze` 공식 흐름에 nutrition scorer를 연결했다.
- 식단 분석 결과에 `DM`, `HTN`, `DL`, `OBE`, `ANEM` 질병군별 점수를 포함한다.

주요 응답/저장 항목:

- `disease_scores`
- `food_score_details`
- `scoring_source=nutrition_rule_table`

테스트:

- `tests/cv/food/nutrition/test_disease_food_scorer.py`
- `tests/diets/test_diet_analysis_nutrition_scoring.py`

### 4.6 dummy/stub 공식 경로 정리

해결 상태: 대부분 해결됨

주요 변경:

- 공식 API 경로에서 `run_dummy_*`, `ask_dummy_*` 같은 명칭을 줄이고 공식 service wrapper를 사용하도록 정리했다.
- deprecated compatibility endpoint는 삭제하지 않고 유지한다.
- Swagger/OpenAPI에는 deprecated dummy endpoint가 보이지 않도록 숨김 처리했다.
- 실제 provider가 아직 없는 경로는 `rule_based`, `fallback`, `pending_review`, `needs_review` 같은 운영 가능한 표현으로 정리했다.

남은 허용 항목:

- DB 호환 필드 `is_dummy`
- 테스트/문서/legacy fallback에서의 stub 표현
- LLM 실제 호출 off 상태의 fallback 표현

### 4.7 CV/GPT Vision provider 구조

해결 상태: 구조 정리 완료, 실제 자동 fallback은 보류

주요 변경:

- GPT Vision provider를 LLM 계층이 아니라 CV provider 하위로 이동했다.
- 식단 분석 provider schema에 `provider`, `confidence`, `detected_foods`, `needs_review`, `raw_output` 같은 공통 필드를 정리했다.
- fallback policy는 기본값에서 유료 GPT Vision 호출을 하지 않도록 설계했다.

현재 정책:

- 기본 경로: `rule_based_food_detection` + nutrition scorer
- 자체 CV 모델: 아직 미구현
- GPT Vision: fallback 후보. env flag/정책이 켜진 경우에만 연결 대상

### 4.8 LLM/RAG 설명 생성

해결 상태: rule-based 설명과 RAG-ready interface 수준

주요 변경:

- 분석 결과와 식단 점수에 대해 사용자 친화적인 설명 생성 구조를 추가했다.
- 실제 유료 LLM 호출은 기본값으로 켜지지 않는다.
- RAG는 vector DB 없이 추후 연결 가능한 interface 수준으로 유지한다.

현재 source:

- `rule_based_explanation`
- `llm_fallback` 후보

테스트:

- `tests/llm/test_explanation_service.py`

### 4.9 문서/보안/시연 체크리스트

해결 상태: 진행됨

추가/보강 문서:

- `docs/design/ai_worker_service_integration_plan.md`
- `docs/design/spec_v2_sync_plan.md`
- `docs/design/disease_scope_policy.md`
- `docs/design/cv_food_fallback_policy.md`
- `docs/design/p2_operational_backlog.md`
- `docs/demo/scenario.md`
- `docs/ops/secrets_handling.md`
- `docs/policy/privacy.md`
- `docs/policy/terms.md`
- `docs/policy/sensitive_health_data_notice.md`
- `docs/qa/frontend_demo_checklist.md`
- `docs/qa/demo_ready_checklist.md`

보안 정리:

- `docker compose config` 전체 출력 금지 문서화
- `.env` 및 실제 secret 파일 ignore 강화
- example env는 placeholder만 유지
- tracked 파일에 실제 키가 들어갔는지 마스킹 기반 점검

### 4.10 테스트 현황

현재 테스트 기준:

- `uv run pytest tests`
- 현재 통과 수: 33개

추가된 테스트 축:

- CV food nutrition scorer
- CV food fallback policy
- Diet analysis nutrition scoring
- Exam OCR confirm -> HealthRecord X2
- LLM explanation service
- ML CatBoost predictor
- ML feature mapper
- Medication OCR parser

CI 기준:

- `.github/workflows/checks.yml`가 로컬 검증 기준과 맞게 루트 `tests` 전체를 돌도록 정리했다.

## 5. 남은 위험도 표

| 구분 | 항목 | 상태 | 설명 | 다음 액션 |
| --- | --- | --- | --- | --- |
| P0 | Docker ML 빌드 | 해결됨 | FastAPI 컨테이너에서 `ai` group, `ai_worker`, CatBoost artifact를 사용할 수 있게 정리 | 시연 직전 `docker compose build fastapi`와 import 확인 |
| P0 | Redis health | 해결됨 | compose 환경에서 Redis host 설정과 health check 흐름 정리 | `/api/v1/system/health` 재확인 |
| P0 | CatBoost PRECISION E2E | 해결됨 | smoke script로 DM/HTN/DL CatBoost 결과 확인 가능 | seed 후 `verify_precision_analysis_api.py` 실행 |
| P0 | OCR confirm -> X2 | 해결됨 | OCR 측정값이 HealthRecord X2 필드에 반영됨 | 실제 OCR/confirm 화면에서 수동 QA |
| P0 | Diet nutrition score | 해결됨 | DM/HTN/DL/OBE/ANEM 점수 응답/저장 구조 연결 | 프론트 표시와 저장 payload 확인 |
| P0 | 시연 seed/계정 가이드 | 부분 해결 | README/team guide/checklist에 실행 순서 정리 | 실제 새 환경에서 한 번 리허설 |
| P0 | 프론트 QA | 부분 해결 | API type/field 대응과 QA 문서 정리 | 브라우저 클릭 리허설 필요 |
| P0 | secret 노출 | 부분 해결 | tracked 파일 점검과 문서화 완료 | 과거 git history 감사는 별도 필요 |
| P1 | CV/GPT Vision fallback | 구조만 준비 | 실제 자동 GPT Vision 호출은 꺼져 있음 | provider 결과 schema 확정 후 opt-in 연결 |
| P1 | LLM 설명 생성 | rule-based 중심 | 실제 LLM/RAG 호출은 기본값 off | 운영 키/비용/관측 붙인 뒤 단계적 활성화 |
| P1 | 모니터링/alerting | 보류 | DB 로그 중심, 외부 APM 없음 | Sentry/Datadog 또는 알림 채널 검토 |
| P1 | rate limiting | 보류 | 일부 인증 제한 외 전체 API 제한 없음 | Redis 기반 limiter 정책 설계 |
| P1 | DB backup/restore | 보류 | Docker volume 중심 | `pg_dump`/restore runbook 작성 |
| P1 | ML warm-up 운영화 | 수동 스크립트 수준 | startup 강제 warm-up은 아직 보류 | 운영 환경에서 optional startup 또는 pre-demo script 선택 |
| P2 | Redis Stream / async_jobs / worker | 의도적 보류 | 시연은 동기 API 기반 | 운영용 비동기 job table/consumer 설계 |
| P2 | vector RAG / pgvector search | 의도적 보류 | RAG-ready interface만 존재 | 신뢰 문서 ingest, embedding, retrieval 구현 |
| P2 | wearable 연동 | 의도적 보류 | MVP/시연 범위 제외 | 외부 OAuth/device API 정책 수립 |
| P2 | 외부 알림 worker | 의도적 보류 | in-app/스케줄 관리 중심 | SMS/Email/Push/Kakao worker 및 retry 구현 |
| P2 | migration squash | 의도적 보류 | 현재 migration history 유지 | 운영 초기화 전 별도 스쿼시 검토 |
| P2 | load test | 의도적 보류 | 기능 시연 우선 | k6/Locust 시나리오 추가 |
| P2 | model registry 고도화 | 의도적 보류 | artifact + JSON metadata 중심 | 모델 승인/rollback/metric registry 설계 |

## 6. 리뷰 포인트

리뷰어가 우선 확인할 항목:

1. Docker/compose 변경이 실제 시연 환경에서 동일하게 동작하는지
2. `analysis_results`에 CatBoost metadata가 저장되는지
3. OCR confirm 이후 `precision_ready=true`가 되는지
4. 식단 분석 응답에 `disease_scores`가 포함되는지
5. Swagger에 deprecated dummy endpoint가 노출되지 않는지
6. 사용자 화면에 `dummy`, `stub`, `mock` 표현이 보이지 않는지
7. `.env`/secret이 tracked 파일에 들어가지 않았는지
8. 새로 추가된 untracked 문서/테스트/스크립트를 어떤 커밋 단위로 나눌지

## 7. 추천 커밋 분리

현재 변경량이 크므로 아래 단위로 나누는 것을 권장한다.

1. Docker/compose/ML runtime
2. OCR confirm -> HealthRecord X2
3. Diet nutrition scorer
4. dummy/stub 공식 경로 정리
5. CV/GPT Vision fallback policy
6. LLM explanation skeleton
7. QA/보안/시연 문서
8. CI/test 정리

## 8. 최종 시연 전 명령

```bash
uv run ruff check app scripts ai_worker tests
uv run ruff format app scripts ai_worker tests --check
uv run pytest tests
uv run python -c "from app.main import app; print(app.title); print(len(app.openapi().get('paths', {})))"
uv run python scripts/verify_demo_ready.py
```

Docker/API E2E:

```bash
docker compose down
docker compose build fastapi
docker compose up -d postgres redis fastapi
curl http://localhost:8000/api/v1/system/health
uv run python scripts/verify_precision_analysis_api.py --warmup-ml
```
