# AI Worker 서비스 연동 및 구현 계획

## 1. 현재 상태 요약

이 문서는 `feature/kdu` 기준 AI Worker 구조 정리 이후, 시연/운영 전까지 남은 서비스 연동 작업을 우선순위별로 정리한 구현 계획서다. 이번 단계에서는 코드 수정, 라우터 연결, 스키마 변경, Dockerfile 수정, `pyproject.toml` 수정, `uv.lock` 수정은 하지 않는다.

현재 `ai_runtime/`는 새 최상위 `AI_worker` 폴더를 만들지 않고 기존 하위에서 도메인별로 정리되어 있다. `docs/design/ai_worker_structure.md`에도 구조 설명 문서가 존재한다. 다만 현재 `git status` 기준으로 여러 코드/문서/의존성 변경사항이 남아 있으므로, AI Worker 구조 정리 커밋이 최종 완료되었는지는 별도 확인이 필요하다.

현재 `ai_runtime/` 하위 구조:

| 경로 | 현재 역할 |
| --- | --- |
| `ai_runtime/ml/` | CatBoost/XGBoost 학습/추론, X2 룰 기반 분류, 모델 artifact, 학습 config |
| `ai_runtime/ocr/` | 건강검진표 OCR, OCR extractor/parser, Clova OCR PoC/deferred provider 후보 영역 |
| `ai_runtime/cv/` | 음식 이미지 분석, CV provider, 품질 판정, CV schema 후보 영역 |
| `ai_runtime/llm/` | LLM 호출, GPT Vision 호출 계층, 프롬프트, RAG 준비, 상담/해설 생성 |
| `ai_runtime/common/` | AI 영역 공통 유틸, 공통 schema 후보 영역 |
| `ai_runtime/jobs/` | 향후 worker/job entrypoint 후보 영역. 현재 queue/stream 구현 없음 |
| `ai_runtime/pipelines/` | OCR -> ML -> LLM orchestration 후보 영역. 현재 신규 파이프라인 기능 없음 |

현재 기록할 검증 명령:

```bash
uv run ruff check app scripts ai_runtime tests
uv run pytest tests
uv run python -c "from app.main import app; print(app.title); print(len(app.openapi().get('paths', {})))"
```

현재 FastAPI OpenAPI path 수는 `110`개로 기록한다.

## 2. 남은 작업 전체 목록

| 번호 | 작업명 | 우선순위 | 목적 | 현재 상태 | 완료 기준 | 관련 경로 | 구현 여부 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | AI worker 정리 커밋 완료 확인 | P0 | 구조 정리 작업의 기준점을 확정한다 | 구조는 나뉘어 있으나 git status상 변경사항이 남아 있어 커밋 완료 여부 확인 필요 | AI Worker 구조 정리 커밋 hash 확인, 남은 변경사항 분리 | `ai_runtime/`, `docs/design/ai_worker_structure.md` | 미구현 |
| 2 | dummy/stub 공식 경로 정리 | P0 | 공식 API/서비스 경로에서 개발용 명칭 노출과 호출을 줄인다 | 일부 서비스에 dummy/stub/is_dummy 표현 잔존 | 공식 endpoint는 `run_analysis`, `run_diet_analysis` 등 공식 함수명 호출. 내부 fallback source만 명확히 유지 | `app/services/*`, `app/apis/v1/*`, `ai_runtime/llm/*` | 일부 구현 |
| 3 | 식단 질병군별 점수 엑셀 작성 | P0 | 음식별 질병군 점수 기준표를 팀 공통 rule source로 만든다 | 질병군별 식단 점수 기준표 필요 | `docs/rules/diet_score_matrix.xlsx` 작성 및 리뷰 완료 | `docs/rules/` | 미구현 |
| 4 | 식단 점수 계산 모듈 구현 | P0 | CV/GPT Vision 음식 결과를 질병군별 점수로 변환한다 | 공식 `/diets/analyze`에 음식명 후보 기반 nutrition scorer 연결. 자체 CV/GPT Vision 공급자는 미연결 | 음식명 입력 -> DM/HTN/DL/OBE/ANEM 점수, 상세 매칭 결과, `nutrition_rule_table` source 반환 | `ai_runtime/cv/food/nutrition/`, `app/services/diets.py` | 일부 구현 |
| 5 | 건강검진 OCR -> 정밀검사 연결 확인/보강 | P0 | OCR 결과가 PRECISION 분석 입력으로 실제 반영되게 한다 | OCR 결과와 `HealthRecord` X2 필드 연결 검증 필요 | OCR/검진값 confirm 후 precision readiness true 가능, PRECISION 분석 실행 가능 | `app/services/exams.py`, `app/services/analysis.py`, `ai_runtime/ocr/`, `ai_runtime/ml/` | 일부 구현 |
| 6 | CV 모델 fallback 정책 정의 | P0 | 자체 식단 CV와 GPT Vision fallback 경계를 정한다 | 자체 CV 모델 미완성, fallback 정책 미정 | confidence threshold, timeout, retry, provider log 정책 문서화 | `ai_runtime/cv/`, `ai_runtime/cv/providers/gpt_vision.py` | 미구현 |
| 7 | GPT Vision fallback 연결 | P0 | 자체 CV 실패/저신뢰 시 GPT Vision 보조 분석을 연결한다 | GPT Vision provider는 있으나 식단 API 흐름 연결 검증 필요 | CV 결과 schema와 GPT Vision 결과 schema 통일, provider/source 기록 | `ai_runtime/cv/providers/gpt_vision.py`, `app/services/diets.py` | 미구현 |
| 8 | LLM/RAG 설명 생성 붙이기 | P1 | 분석/식단/OCR 결과를 사용자 친화적 설명으로 변환한다 | 분석/식단 설명은 `explanation_service`로 공식 runtime 연결. keyword RAG PoC와 Langfuse trace metadata가 붙어 있으나 챗봇 라우터/추천문구 모듈은 아직 공식 runtime 미연결 | 공식 runtime과 prepared-not-wired 범위 문서화, 챗봇/추천문구 연결 여부 별도 결정 | `ai_runtime/llm/`, `ai_runtime/llm/rag/`, `app/services/analysis.py`, `app/services/diets.py` | 일부 구현 |
| 9 | Docker ML 빌드 검증 | P0 | 시연/배포 컨테이너에서 ML import와 artifact loading을 검증한다 | 로컬 검증 중심. Docker 빌드 검증 필요 | fastapi 컨테이너에서 CatBoost import 및 `predict_chronic_disease_risks` import 성공 | `Dockerfile`, `.dockerignore`, `infra/docker/`, `ai_runtime/ml/artifacts/` | 미구현 |
| 10 | Redis Stream / async_jobs / worker 구조 | P2 | 운영용 비동기 AI 작업 처리 기반을 만든다 | 현재 구현 범위 밖. queue/stream 없음 | async job 상태 추적, retry, DLQ, worker heartbeat 설계/구현 | `ai_runtime/jobs/`, `ai_runtime/pipelines/`, future DB model | 미구현 |

## 3. 우선순위 기준

### P0: 시연 전에 반드시 필요한 작업

- AI worker 정리 커밋 완료 확인
- dummy/stub 공식 경로 정리
- 식단 질병군별 점수 엑셀 작성
- 식단 점수 계산 모듈 구현
- 건강검진 OCR -> 정밀검사 연결 확인/보강
- CV 모델 fallback 정책 정의
- GPT Vision fallback 연결
- Docker ML 빌드 검증

### P1: 시연 품질을 올리는 작업

- LLM/RAG 설명 생성 붙이기
- 약봉투 OCR 구조 설계
- 처방전 OCR 구조 설계
- 모델 warm-up
- OCR/CV/ML 결과 로그 정리

### P2: 운영 진입용 인프라 작업

- Redis Stream
- async_jobs
- 별도 AI Worker consumer
- retry/dead-letter queue
- idempotency key
- worker status monitoring
- RAG vector DB 고도화

## 4. 각 작업별 상세 설명

### 1. AI worker 정리 커밋 완료 확인

- 목적: AI Worker 구조 정리 작업의 기준점을 확정한다.
- 왜 필요한가: 이후 dummy 정리, OCR/ML/CV/LLM 연결, Docker 검증 작업이 같은 파일 구조를 기준으로 진행되어야 한다.
- 현재 문제: `ai_runtime/` 구조는 도메인별로 나뉘어 있지만 현재 워크트리에 여러 변경사항이 남아 있어 정리 커밋 완료 여부를 별도로 확인해야 한다.
- 해야 할 일:
  - `git status --short` 확인
  - AI Worker 구조 정리 관련 변경과 다른 작업 변경을 분리
  - 구조 정리 커밋 hash 기록
- 건드릴 파일 후보:
  - `ai_runtime/`
  - `docs/design/ai_worker_structure.md`
- 완료 기준:
  - AI Worker 구조 정리 커밋이 명확히 존재한다.
  - 남은 변경사항과 후속 작업 범위가 분리되어 있다.
- 검증 명령:
  ```bash
  git status --short
  find ai_runtime -maxdepth 3 -type d | sort
  ```
- 주의사항:
  - 이번 문서 작업에서는 커밋하지 않는다.
  - `experiment/ml` legacy 실험 파일은 건드리지 않는다.

### 2. dummy/stub 공식 경로 정리

- 목적: 외부 API/공식 서비스 경로에서 `dummy`라는 이름을 줄이고, 실제 구현 상태는 source/fallback 명칭으로 명확히 남긴다.
- 왜 필요한가: 시연/운영 전 사용자와 OpenAPI에 개발용 명칭이 노출되면 서비스 완성도가 낮아 보이고, 실제 fallback과 테스트 데이터를 구분하기 어렵다.
- 현재 문제: 현재 검색 결과 기준으로 아래 함수/표현이 남아 있다.
  - `app/services/analysis.py`
    - `run_dummy_analysis`
    - `_calculate_dummy_scores`
    - `_dummy_factors`
    - `_dummy_snapshot_request`
    - `_create_dummy_challenge_recommendations`
  - `app/services/diets.py`
    - `run_dummy_diet_analysis`
    - `is_dummy`
    - `rule_stub`
    - `image_analysis_stub`
  - `app/services/exams.py`
    - `run_dummy_ocr`
  - `app/services/medications.py`
    - `run_dummy_medication_ocr`
  - `app/services/chatbot.py`
    - `ask_dummy_chatbot`
  - `app/services/main.py`
    - `_build_dummy_ai_comment`
  - `ai_runtime/llm/llm_generator.py`
    - `llm_stub`
    - `llm_rewrite_stub`
  - `ai_runtime/llm/rag_generator.py`
    - `build_stub_answer`
- 해야 할 일:
  - 공식 API/서비스 경로에서는 `dummy` 함수명을 줄인다.
  - 공식 함수명은 `run_analysis`, `run_diet_analysis`, `run_exam_ocr`, `run_medication_ocr`, `ask_chatbot`처럼 정리한다.
  - deprecated endpoint나 내부 compatibility wrapper는 유지할 수 있다.
  - 실제 구현이 없는 부분은 fake로 포장하지 않고 `provider_fallback`, `rule_based`, `pending_review`, `needs_review` 같은 source/status 표현을 사용한다.
  - `is_dummy`처럼 DB/응답 스키마 호환성이 필요한 필드는 즉시 삭제하지 않고 의미를 재정의하거나 후속 migration 계획으로 분리한다.
- 건드릴 파일 후보:
  - `app/apis/v1/analysis_routers.py`
  - `app/apis/v1/diet_routers.py`
  - `app/apis/v1/exam_routers.py`
  - `app/apis/v1/medication_routers.py`
  - `app/apis/v1/chatbot_routers.py`
  - `app/services/analysis.py`
  - `app/services/diets.py`
  - `app/services/exams.py`
  - `app/services/medications.py`
  - `app/services/chatbot.py`
  - `app/services/main.py`
  - `ai_runtime/llm/llm_generator.py`
  - `ai_runtime/llm/rag_generator.py`
- 완료 기준:
  - 공식 service/router 호출 경로에서 `run_dummy`, `ask_dummy`, `_build_dummy` 호출이 사라진다.
  - deprecated 경로는 wrapper로 남아도 된다.
  - 사용자 응답에 `dummy`, `더미`, `stub` 표현이 노출되지 않는다.
- 검증 명령:
  ```bash
  grep -R -n "run_dummy\|ask_dummy\|_build_dummy" app/services app/apis/v1 ai_runtime || true
  uv run ruff check app scripts ai_runtime tests
  uv run pytest tests
  ```
- 주의사항:
  - 실제 AI/OCR/LLM provider가 아직 없는데 구현된 것처럼 표현하지 않는다.
  - DB 스키마 변경은 별도 작업으로 분리한다.

### 3. 식단 질병군별 점수 엑셀 작성

- 목적: 음식별 질병군 점수 기준을 팀 공통 rule source로 만든다.
- 왜 필요한가: CV/GPT Vision이 음식명을 추출해도, 질병군별 식단 점수를 계산하려면 별도의 영양/질병군 기준표가 필요하다.
- 현재 문제: 질병군별 음식 점수 기준이 서비스 코드와 분리된 공식 산출물로 정리되어 있지 않다.
- 해야 할 일:
  - 엑셀 파일 후보 위치를 `docs/rules/diet_score_matrix.xlsx`로 제안한다.
  - 질병군은 아래 5개를 기준으로 한다.
    - 당뇨: `DM` / `DIABETES`
    - 고혈압: `HTN` / `HYPERTENSION`
    - 이상지질혈증: `DL` / `DYSLIPIDEMIA`
    - 비만: `OBE` / `OBESITY`
    - 빈혈: `ANEM` / `ANEMIA`
  - 엑셀 시트 후보:
    - `foods`
    - `disease_scores`
    - `nutrients`
    - `scoring_rules`
    - `examples`
  - `foods` 시트 필수 컬럼 후보:
    - `food_id`
    - `food_name_ko`
    - `food_category`
    - `serving_unit`
    - `default_serving_g`
    - `calories_kcal`
    - `carbohydrate_g`
    - `sugar_g`
    - `protein_g`
    - `fat_g`
    - `saturated_fat_g`
    - `sodium_mg`
    - `fiber_g`
    - `cholesterol_mg`
    - `iron_mg`
    - `notes`
  - `disease_scores` 시트 필수 컬럼 후보:
    - `food_id`
    - `disease_code`
    - `score`
    - `risk_level`
    - `reason`
    - `caution_message`
    - `recommend_message`
- 점수 방향:
  - 질병군별로 같은 음식이라도 점수가 달라질 수 있다.
  - 예: 흰쌀밥은 당뇨/비만에는 감점, 빈혈에는 직접 감점이 약할 수 있다.
  - 예: 짠 국물 음식은 고혈압에는 큰 감점, 당뇨에는 상대적으로 작은 감점.
  - 예: 붉은 살코기는 빈혈에는 가점 가능, 이상지질혈증에는 조리 방식에 따라 감점 가능.
- 건드릴 파일 후보:
  - `docs/rules/diet_score_matrix.xlsx`
  - `docs/design/full_service_scope.md`
- 완료 기준:
  - 팀이 리뷰 가능한 엑셀 기준표가 존재한다.
  - 최소 5개 질병군에 대해 score/reason/caution/recommend_message가 작성되어 있다.
- 검증 명령:
  ```bash
  ls docs/rules/diet_score_matrix.xlsx
  ```
- 주의사항:
  - 이 엑셀은 rule source이며, 서비스 런타임에서 직접 읽을지 변환 산출물을 읽을지는 별도 결정한다.
  - 의료적 진단/치료 단정 문구를 쓰지 않는다.

### 4. 식단 점수 계산 모듈 구현

- 목적: CV 또는 GPT Vision 결과로 나온 음식명을 질병군별 식단 점수로 변환한다.
- 왜 필요한가: 음식 이미지 분석 결과만으로는 사용자가 이해할 수 있는 만성질환 관리 점수가 나오지 않는다.
- 현재 문제: 식단 점수 계산 모듈은 공식 `/diets/analyze` 경로에 연결되었지만, 음식명 후보는 아직 자체 CV 모델이 아니라 rule-based 후보 생성 흐름에서 나온다.
- 해야 할 일:
  - 현재는 `app/services/diets.py`의 음식명 후보를 입력받는다.
  - `ai_runtime/cv/food/nutrition/data/food_disease_scores.csv`와 `disease_score_rules.json`을 읽는다.
  - 사용자 질병군별 점수를 계산한다.
  - `disease_scores`, `food_score_details`, `scoring_source`를 응답과 `DietPhotoResult.raw_output`에 포함한다.
  - 추후 CV 또는 GPT Vision 결과로 나온 음식명을 같은 scorer 입력으로 연결한다.
- 건드릴 파일 후보:
  - `ai_runtime/cv/food/nutrition/scoring/disease_food_scorer.py`
  - `ai_runtime/cv/food/nutrition/scoring/schemas.py`
  - `ai_runtime/cv/food/nutrition/data/food_disease_scores.csv`
  - `ai_runtime/cv/food/nutrition/rules/disease_score_rules.json`
  - `app/services/diets.py`
- 완료 기준:
  - 음식명 배열 입력 시 DM/HTN/DL/OBE/ANEM 점수와 음식별 매칭 상세를 반환한다.
  - 런타임에서는 엑셀 원본이 아니라 변환된 CSV/JSON rule table을 읽는다.
  - 자체 CV 모델과 GPT Vision fallback은 아직 미연결임을 사용자/문서에서 혼동하지 않게 유지한다.
- 검증 명령:
  ```bash
  uv run pytest tests
  uv run ruff check ai_runtime app tests
  ```
- 주의사항:
  - 영양소/음식 성분 점수화는 `ai_runtime/cv/food/nutrition/` 하위에 두고 음식 이미지 분석 흐름과 함께 관리한다.
  - 이번 문서 작업에서는 구현하지 않는다.

### 5. 건강검진 OCR -> 정밀검사 연결 확인/보강

- 목적: 사용자가 건강검진 정보가 없으면 BASIC 경로를 사용하고, 건강검진 OCR 결과가 있으면 PRECISION 경로로 전환되게 한다.
- 왜 필요한가: 현재 사용자가 검진표를 올려도 그 결과가 정밀 분석 입력값으로 실제 연결되지 않으면 UX와 분석 로직이 분리된다.
- 현재 문제:
  - OCR 결과가 `HealthRecord`의 X2 필드로 반영되는지 검증 필요.
  - `AnalysisResult` 공식 질병군은 DM/HTN/DL/OBE 중심이며, ANEM은 X2 참고 분류로 유지 가능하다.
- 해야 할 일:
  - `app/services/exams.py`에서 OCR/confirm 결과가 health record에 반영되는지 확인한다.
  - `app/apis/v1/exam_routers.py` confirm 흐름을 확인한다.
  - `app/services/analysis.py`에서 PRECISION 모드가 DM/HTN/DL CatBoost와 OBE rule fallback을 실제로 사용하는지 검증한다.
  - `ai_runtime/ml/inference/disease_risk_service.py`의 artifact loading/fallback 정책을 확인한다.
  - OCR field key와 health field key를 매핑한다.
- 건드릴 파일 후보:
  - `app/services/analysis.py`
  - `ai_runtime/ml/inference/disease_risk_service.py`
  - `ai_runtime/ocr/checkup/`
  - `ai_runtime/ocr/providers/clova_ocr/`
  - `app/apis/v1/analysis_routers.py`
  - `app/apis/v1/exam_routers.py`
- 완료 기준:
  - 건강검진 OCR 또는 수동 입력 후 `precision_ready=true`가 될 수 있다.
  - PRECISION 분석 실행 시 DM/HTN/DL은 CatBoost 가능 시 ML 결과를 사용하고, artifact가 없으면 rule fallback을 사용한다.
  - OBE는 현재 ML 모델이 없으면 rule 기반으로 처리한다.
- 검증 명령:
  ```bash
  uv run pytest tests
  uv run python -c "from app.main import app; print(app.title); print(len(app.openapi().get('paths', {})))"
  ```
- 주의사항:
  - 분석 결과는 진단/확진이 아니라 위험도/참고용 분석으로 표현한다.
  - OCR 원문/이미지 경로 같은 민감정보는 로그에 남기지 않는다.

### 6. CV 모델 fallback 정책 정의

- 목적: 식단 분석에서 자체 CV 모델과 GPT Vision fallback의 경계를 명확히 한다.
- 왜 필요한가: 비용이 발생하는 GPT Vision을 언제 호출할지, 자체 CV 결과를 언제 신뢰할지 정책이 필요하다.
- 현재 문제:
  - 아직 식단 CV 모델은 미완성이다.
  - GPT Vision API는 fallback 후보이며, 현재 정책은 비용 보호를 위해 사용자 확인 후 호출하는 방향이다.
- 해야 할 일:
  - 1차: 자체 식단 CV 모델
  - 2차: 자체 CV 모델 confidence 부족 또는 음식명 후보 부족 시 GPT Vision API
  - 3차: GPT Vision 실패 시 수동 입력 또는 `needs_review`
  - confidence threshold, timeout, retry, provider 로그 정책을 문서화한다.
  - provider 결과는 `ai_runtime/cv/food/schemas.py`의 `FoodDetectionCandidateSet` 기준으로 정규화한다.
  - fallback 호출 정책은 `GPT_VISION_FALLBACK_POLICY=user_confirmation_required`를 기본으로 둔다.
  - 최종 `DiseaseFoodScorer` 입력은 provider와 무관하게 음식명 후보 `list[str]`로 통일한다.
- 건드릴 파일 후보:
  - `ai_runtime/cv/`
  - `ai_runtime/cv/food/`
  - `ai_runtime/cv/providers/gpt_vision.py`
  - `app/services/diets.py`
- 완료 기준:
  - provider priority와 fallback 조건이 문서화되어 있다.
  - fallback source가 `cv_model`, `gpt_vision`, `rule_based_food_detection`, `manual_input`, `needs_review` 등으로 구분된다.
  - 정규화 필드 `provider`, `confidence`, `detected_foods`, `needs_review`, `fallback_reason`이 타입 또는 문서에 명시되어 있다.
- 검증 명령:
  ```bash
  rg -n "provider|confidence|gpt_vision|needs_review" ai_runtime app/services/diets.py
  ```
- 주의사항:
  - GPT Vision은 비용 발생 API이므로 무조건 자동 호출하지 않는다.
  - 실패한 provider raw output 저장 범위는 개인정보/비용/보안 기준으로 제한한다.

### 7. GPT Vision fallback 연결

- 목적: 자체 CV 모델 결과가 없거나 신뢰도가 낮을 때 GPT Vision으로 식단 분석을 보조한다.
- 왜 필요한가: 시연 전 자체 CV 정확도가 충분하지 않으면 사용자가 식단 분석 결과를 얻지 못할 수 있다.
- 현재 문제:
  - `ai_runtime/cv/providers/gpt_vision.py`는 존재하지만 공식 식단 API 흐름과 schema 통합이 필요하다.
  - `ai_runtime/cv/router.py`와 `app/services/diets.py` 연결 정책을 확정해야 한다.
- 해야 할 일:
  - CV 결과 schema와 GPT Vision 결과 schema를 통일한다.
  - `provider` 필드를 남긴다.
  - `raw_output` 저장 여부와 보관 범위를 정한다.
  - 비용 발생 API이므로 호출 조건을 명확히 한다.
  - GPT Vision 결과도 `FoodDetectionCandidateSet`으로 정규화한 뒤 nutrition scorer에 전달한다.
- 건드릴 파일 후보:
  - `ai_runtime/cv/providers/gpt_vision.py`
  - `ai_runtime/cv/router.py`
  - `app/services/diets.py`
  - `app/apis/v1/diet_routers.py`
- 완료 기준:
  - 식단 분석 서비스에서 자체 CV 실패/저신뢰 시 GPT Vision fallback을 호출할 수 있다.
  - fallback 호출 여부와 provider가 응답/로그에서 구분된다.
  - 자동 호출인지 사용자 확인 후 호출인지 정책이 화면/API에서 일관되게 표현된다.
- 검증 명령:
  ```bash
  uv run pytest tests
  uv run ruff check app scripts ai_runtime tests
  ```
- 주의사항:
  - 실제 API key가 없을 때 서버 import가 깨지면 안 된다.
  - 사용자에게 GPT Vision 내부 오류나 provider secret을 노출하지 않는다.

### 8. LLM/RAG 설명 생성 붙이기

- 목적: 식단 점수 결과, 건강검진 분석 결과, OCR 결과를 기반으로 사용자가 이해할 수 있는 설명 문장을 생성한다.
- 왜 필요한가: ML/CV/OCR 결과만으로는 사용자가 무엇을 실천해야 하는지 이해하기 어렵다.
- 현재 문제:
  - 분석/식단 설명은 rule-based explanation으로 공식 runtime에 연결되어 있다.
  - keyword RAG PoC는 markdown source 기반으로 reference source를 붙인다.
  - 메인 챗봇의 `response_router.py`, `health_chatbot.py`, `rule_engine.py`, `llm_generator.py`는 준비됐지만 `app/services/chatbot.py` 공식 경로에는 아직 직접 연결되지 않았다.
  - vector RAG, LangChain/LangGraph, 운영형 평가 파이프라인은 아직 완성된 운영 경로가 아니다.
- 해야 할 일:
  - 공식 runtime 범위는 `docs/design/llm_runtime_scope.md`를 기준으로 유지한다.
  - 챗봇 공식 API를 `ai_runtime.llm.response_router`로 연결할지 별도 작업에서 결정한다.
  - 추천/챌린지 문구 모듈을 DB challenge recommendation 흐름과 통합할지 결정한다.
  - 운영형 RAG는 질병/영양/복약 주의사항 근거 문서가 준비된 뒤 붙인다.
  - 입력에 없는 질환/수치/챌린지를 LLM이 생성하지 않도록 grounding 검사를 유지한다.
- 건드릴 파일 후보:
  - `ai_runtime/llm/`
  - `ai_runtime/llm/rag/`
  - `ai_runtime/llm/explanation_service.py`
  - `ai_runtime/llm/response_router.py`
  - `ai_runtime/llm/recommendation_message.py`
  - `app/services/chatbot.py`
- 완료 기준:
  - 분석 결과를 기반으로 안전한 설명 문구를 생성한다.
  - 진단/치료/처방 단정 표현을 하지 않는다.
  - RAG 사용 시 허용 출처 기반 context만 사용한다.
- 검증 명령:
  ```bash
  uv run pytest tests
  uv run ruff check ai_runtime tests
  ```
- 주의사항:
  - RAG vector DB 구현은 이번 계획의 P2로 둔다.
  - 의료적 판단은 LLM이 하지 않는다.

### 9. Docker ML 빌드 검증

- 목적: 시연/배포 환경의 Docker 이미지에서 ML import, CatBoost artifact loading, FastAPI 실행이 가능한지 확인한다.
- 왜 필요한가: 로컬에서는 동작해도 Docker image에 `ai_runtime`, `catboost`, artifact가 누락되면 시연/배포에서 정밀분석이 실패한다.
- 현재 문제:
  - Dockerfile과 `.dockerignore` 기준으로 ML artifact 포함 여부와 AI dependency 설치 여부 확인이 필요하다.
- 확인할 것:
  - `pyproject.toml` ai 그룹에 `catboost`, `xgboost`, `pandas`, `numpy`가 포함되는지
  - app Dockerfile에서 `--group ai`를 설치하는지
  - app Docker image에 `ai_runtime`가 COPY 되는지
  - `.dockerignore`가 `ai_runtime/ml/artifacts/*.cbm`을 제외하지 않는지
  - FastAPI 컨테이너에서 ML import가 되는지
- 건드릴 파일 후보:
  - `Dockerfile`
  - `.dockerignore`
  - `infra/docker/`
  - `pyproject.toml`
  - `uv.lock`
  - `ai_runtime/ml/artifacts/`
- 완료 기준:
  - FastAPI 컨테이너 안에서 `predict_chronic_disease_risks` import 성공
  - CatBoost artifact 존재 시 predictor load 가능
- 검증 명령 후보:
  ```bash
  docker compose build fastapi
  docker compose up -d fastapi
  docker compose exec fastapi python -c "from ai_runtime.ml.inference.disease_risk_service import predict_chronic_disease_risks; print('OK')"
  ```
- 주의사항:
  - 이번 문서 작업에서는 Dockerfile을 수정하지 않는다.
  - 모델 artifact 중 최종 추론에 필요한 파일만 이미지에 포함한다.

### 10. Redis Stream / async_jobs / worker 구조

- 목적: 운영용 비동기 AI 작업 처리 기반을 만든다.
- 왜 필요한가: OCR, CV, LLM, RAG는 비용과 지연시간이 커서 동기 API로만 처리하기 어렵다.
- 현재 문제:
  - 이번 시연 전에는 구현 우선순위가 낮다.
  - 현재 Redis Stream, async_jobs, 별도 AI Worker consumer 구현은 없다.
- 필요한 구성:
  - `async_jobs` 테이블
  - `job_id`
  - `request_id`
  - status: `pending`, `processing`, `success`, `failed`, `retrying`, `canceled`
  - Redis Stream event
  - producer
  - consumer
  - retry
  - dead-letter
  - worker heartbeat
  - idempotency key
- 건드릴 파일 후보:
  - `ai_runtime/jobs/`
  - `ai_runtime/pipelines/`
  - future DB models/migrations
  - future admin monitoring API
- 완료 기준:
  - async job 생성/조회/상태 갱신 가능
  - worker heartbeat와 retry/dead-letter 정책 존재
- 검증 명령:
  ```bash
  uv run pytest tests
  uv run ruff check app scripts ai_runtime tests
  ```
- 주의사항:
  - 이번 시연 전에는 P2로 유지한다.
  - 이 문서에서는 설계만 정리하고 구현하지 않는다.

## 5. 이번 단계에서 구현하지 않을 것

아래 항목은 명시적으로 이번 문서 작업에서 제외한다.

- 약봉투 OCR 구현
- 처방전 OCR 구현
- Redis Stream 구현
- async_jobs 모델/마이그레이션 생성
- RAG vector DB 구현
- Dockerfile 수정
- `pyproject.toml` 수정
- app router 추가 등록
- 실제 코드 리팩터링

## 6. 다음 작업 순서 제안

추천 실행 순서:

1. 이 문서 생성
2. dummy/stub 공식 경로 정리
3. 식단 질병군별 점수 엑셀 작성
4. 식단 점수 계산 모듈 구현
5. 건강검진 OCR -> 정밀검사 연결 확인
6. CV fallback 정책 문서화
7. GPT Vision fallback 연결
8. Docker ML 빌드 검증
9. LLM 설명 생성 연결
10. Redis Stream / async_jobs 설계 및 구현

## 7. 완료 기준

이번 작업의 완료 기준:

- `docs/design/ai_worker_service_integration_plan.md` 파일이 생성되어야 한다.
- 코드 구현 파일은 수정하지 않는다.
- 기존 테스트/ruff를 실행할 필요는 없지만, 문서 생성 후 `git status --short`로 변경 파일을 확인한다.
- 문서 안에 P0/P1/P2 우선순위와 완료 기준이 있어야 한다.
- 현재 서비스 상태를 과장하지 않고, 미구현/구조만 존재/검증 필요를 명확히 표기한다.
