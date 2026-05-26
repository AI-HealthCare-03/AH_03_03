# AI HealthCare MVP

AI HealthCare MVP는 사용자의 건강정보, 건강검진 OCR, 식단 이미지, 복약 정보, 챗봇 응답을 하나의 웹 서비스 흐름으로 묶은 FastAPI + React 기반 헬스케어 시연 프로젝트입니다.

이 README는 팀원이 clone 후 같은 방식으로 실행하고, 실제 모델/provider 연결 상태를 재현할 수 있도록 작성한 기준 문서입니다.

## 현재 구현 범위

### 핵심 사용자 흐름

- 이메일 인증 기반 회원가입/로그인
- 건강정보 입력 및 수정
- 건강검진 OCR 후보 추출, 사용자 확인, 최신 건강정보 반영
- 건강분석 실행
  - `DIABETES`, `HYPERTENSION`, `DYSLIPIDEMIA`: CatBoost artifact 기반 정밀분석
  - `OBESITY`: rule-based 분석
- 식단 이미지 분석
  - 기본: rule-based food detection + DiseaseFoodScorer
  - 옵션: GPT Vision 음식명 추론
- 식단 결과의 질환별 점수 표시
- 복약/OCR parser skeleton
- 챗봇
  - 기본: rule-based 응답
  - 옵션: OpenAI LLM rewrite
- Langfuse trace/prompt/eval 관측 연결
- 관리자/모니터링 계정 seed
- Redis Stream `DEMO_ECHO` async job skeleton

### 의도적으로 보류된 범위

- 소셜 로그인
- 휴대폰 SMS 인증
- 웨어러블 연동
- 운영형 vector RAG, embedding, pgvector retrieval
- LangChain/LangGraph
- 실제 OCR/CV/ML/LLM workflow의 Redis Stream 비동기 전환
- retry, DLQ, heartbeat 기반 worker 운영 구조
- Clova OCR 공식 실행 경로

## 기술 스택

- Backend: FastAPI, Tortoise ORM, Aerich
- Frontend: React, Vite, TypeScript
- DB: PostgreSQL + pgvector image
- Cache/Queue: Redis
- AI Runtime Package: `ai_runtime`
- ML: CatBoost
- OCR/CV Provider: GPT Vision optional, PaddleOCR/text extraction path, rule/fallback path
- LLM Provider: OpenAI optional
- Observability: Langfuse optional
- Package Manager: uv
- Container: Docker Compose

## 저장소 구조

```text
.
├── app/                         # FastAPI API, service, DTO, DB model
├── ai_runtime/                  # ML/CV/OCR/LLM/RAG runtime package
│   ├── ml/                      # CatBoost inference, feature mapper, artifacts
│   ├── cv/                      # food detection, nutrition scoring, GPT Vision provider
│   ├── ocr/                     # checkup/medication OCR parser/provider structure
│   ├── llm/                     # explanation, chatbot routing, RAG PoC
│   └── jobs/                    # Redis Stream DEMO_ECHO worker skeleton
├── frontend/                    # React/Vite frontend
├── infra/docker/                # frontend 포함 dev/prod Docker compose
├── infra/langfuse/              # optional self-host Langfuse compose
├── scripts/                     # seed, audit, smoke, deploy scripts
├── docs/                        # design, demo, ops, QA, RAG source registry
├── envs/                        # example env files only
├── app/Dockerfile
├── ai_runtime/Dockerfile
├── docker-compose.yml           # backend/AI 검증용 root compose
├── Makefile
└── pyproject.toml
```

주의: Docker service 이름은 `ai-worker`로 유지하지만 Python package 이름은 `ai_runtime`입니다. 서비스 코드에서는 `ai_runtime`을 import합니다.

## 사전 준비

- Docker Desktop 또는 Docker Engine
- uv
- Python 3.13 이상

기본 실행은 Full Docker dev stack 기준입니다. 프론트, Nginx, FastAPI, AI Worker, PostgreSQL, Redis를 모두 Docker로 띄우므로 웹 화면 확인만 할 때는 Node.js/npm을 직접 실행할 필요가 없습니다.

로컬 테스트, seed, smoke script 실행을 위해 Python/uv는 준비합니다.

```bash
uv sync
```

Node.js 20 이상과 npm은 프론트 단독 개발, `npm run build`, `npm run dev` 디버깅을 할 때만 필요합니다.

## 환경변수 준비

팀원 실행 기준으로는 루트 `.env`를 사용합니다.

```bash
cp envs/example.local.env .env
```

`.env`는 git에 커밋하지 않습니다. 실제 OpenAI, Langfuse, SMTP key는 각자 로컬 `.env`에만 넣습니다.

최소 로컬/시연 기본값:

```env
ENV=local
DB_HOST=postgres
DB_PORT=5432
DB_USER=ozcoding
DB_PASSWORD=PLEASE_CHANGE_ME
DB_NAME=ai_health

EMAIL_ENABLED=false
EMAIL_VERIFICATION_DEBUG=true
PASSWORD_RESET_DEBUG=false

OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
CHATBOT_USE_REAL_LLM=false
RAG_ENABLED=false

DIET_VISION_PROVIDER=rule_based
DIET_GPT_VISION_ENABLED=false

EXAM_OCR_PROVIDER=fallback
EXAM_GPT_VISION_ENABLED=false
PADDLE_OCR_ENABLED=false

MEDICATION_OCR_PROVIDER=fallback
MEDICATION_GPT_VISION_ENABLED=false

LANGFUSE_ENABLED=false
LANGFUSE_BASE_URL=http://host.docker.internal:3000
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

실제 provider 연결 재현 시에는 `.env`에서 필요한 flag만 켭니다.

```env
OPENAI_API_KEY=<각자 발급받은 키>
OPENAI_MODEL=gpt-4o-mini

CHATBOT_USE_REAL_LLM=true

DIET_VISION_PROVIDER=gpt_vision
DIET_GPT_VISION_ENABLED=true

EXAM_OCR_PROVIDER=gpt_vision
EXAM_GPT_VISION_ENABLED=true

MEDICATION_OCR_PROVIDER=gpt_vision
MEDICATION_GPT_VISION_ENABLED=true

LANGFUSE_ENABLED=true
LANGFUSE_BASE_URL=http://host.docker.internal:3000
LANGFUSE_PUBLIC_KEY=<각자 Langfuse public key>
LANGFUSE_SECRET_KEY=<각자 Langfuse secret key>
```

Secret 주의:

- `.env`, `envs/.local.env`, `frontend/.env`는 커밋하지 않습니다.
- `docker compose config` 전체 출력은 secret이 펼쳐질 수 있으므로 화면 공유에 사용하지 않습니다.
- 설정 확인은 `docker compose ps`, `docker compose config --services`, 마스킹 스크립트, SET/MISSING 출력 위주로 합니다.
- API key 원문, SMTP password, Langfuse secret key, OpenAI key를 로그/문서/스크린샷에 남기지 않습니다.

## 기본 실행: Full Docker dev stack

팀원/시연 표준 실행은 `infra/docker/docker-compose.dev.yml` 기반 Full Docker dev stack입니다. 이 스택이 프론트, Nginx, FastAPI, AI Worker, PostgreSQL, Redis를 함께 실행합니다.

`make demo-up`은 아래 compose 명령을 감싼 단축 명령입니다.

```bash
make demo-up
```

동일한 명령을 직접 실행하면 다음과 같습니다.

```bash
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml up -d --build
```

접속 주소:

- Web: `http://localhost:8080`
- API Docs: `http://localhost:8080/api/docs`
- API Health: `http://localhost:8080/api/v1/system/health`
- FastAPI 직접 접근: `http://localhost:8000`
- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`

기본 실행 순서:

```bash
# 1. 환경변수 파일 준비
cp envs/example.local.env .env

# 2. Full Docker dev stack 실행
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml up -d --build

# 3. DB migration 적용
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml exec fastapi uv run --no-sync aerich upgrade

# 4. 챌린지 마스터 seed
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml exec fastapi uv run --no-sync python scripts/seed_mvp_challenges.py

# 5. 데모 사용자/FAQ/대시보드 seed
DB_HOST=localhost uv run python scripts/setup_local_mvp_db.py

# 6. health check
curl -fsS http://localhost:8080/api/v1/system/health
```

상태 확인:

```bash
make demo-ps
make demo-health
make demo-logs
```

종료:

```bash
make demo-down
```

주의:

- `make demo-down`은 컨테이너만 내립니다.
- DB volume까지 삭제하는 `docker compose down -v`는 시연/협업 중 사용하지 않습니다.
- 시연 기준 compose는 `infra/docker/docker-compose.dev.yml`입니다.
- 루트 `docker-compose.yml`은 backend/AI 검증용입니다. 프론트까지 보려면 `make demo-up` 또는 `infra/docker/docker-compose.dev.yml`을 사용합니다.
- 루트 compose의 `http://localhost`가 404인 것은 설계상 정상입니다. 전체 웹은 `http://localhost:8080`입니다.
- `.env`나 provider flag를 바꾼 뒤에는 FastAPI/AI Worker 컨테이너를 재생성해야 반영됩니다.

```bash
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml up -d --build --force-recreate fastapi ai-worker nginx frontend
```

## DB migration 및 seed

DB도 Full Docker dev stack의 PostgreSQL 컨테이너를 기준으로 사용합니다. 새 volume이거나 schema가 비어 있으면 먼저 migration을 적용합니다.

```bash
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml exec fastapi uv run --no-sync aerich upgrade
```

챌린지 seed는 FastAPI Docker image에 포함되어 있으므로 컨테이너 내부에서 실행할 수 있습니다.

```bash
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml exec fastapi uv run --no-sync python scripts/seed_mvp_challenges.py
```

데모 사용자, FAQ, 대시보드용 통합 seed는 현재 호스트에서 실행하는 방식을 우선 사용합니다. dev compose는 PostgreSQL을 `localhost:5432`로 노출하므로 `DB_HOST=localhost`를 붙입니다.

```bash
DB_HOST=localhost uv run python scripts/setup_local_mvp_db.py
```

필요한 seed만 호스트에서 개별 실행할 수도 있습니다.

```bash
DB_HOST=localhost uv run python scripts/seed_mvp_challenges.py
DB_HOST=localhost uv run python scripts/seed_mvp_faqs.py
DB_HOST=localhost uv run python scripts/seed_demo_users.py
DB_HOST=localhost uv run python scripts/seed_current_user_dashboard_demo.py --email demo@example.com
```

정리:

- `seed_mvp_challenges.py`: Docker 내부 실행 가능
- `setup_local_mvp_db.py`, `seed_demo_users.py`, `seed_mvp_faqs.py`: 현재 호스트 실행 권장
- DB 데이터는 Docker volume에 남습니다. PR이나 Docker image에 DB row가 포함되는 것은 아닙니다.

## 데모 계정

모든 데모 계정의 기본 비밀번호:

```text
Demo1234!
```

일반 사용자:

| 용도 | 이메일 | login_id |
| --- | --- | --- |
| 일반 데모 | `demo@example.com` | `demo_user` |
| 고위험 데모 | `demo_high@example.com` | `demo_high` |

관리자/모니터링:

| 권한 | 이메일 | login_id |
| --- | --- | --- |
| SUPER_ADMIN | `admin@example.com` | `admin01` |
| MONITOR | `monitor@example.com` | `monitor01` |

관리자 화면은 로그인 후 사이드바의 “관리자 콘솔” 또는 `/admin`에서 확인합니다.

## AI 기능 연결 상태

### 건강분석 ML

정밀분석(`PRECISION`) 기준:

| 질환 | 현재 runtime | 기대 model/source |
| --- | --- | --- |
| 당뇨 | CatBoost artifact | `catboost`, `dm_catboost_final` |
| 고혈압 | CatBoost artifact | `catboost`, `htn_catboost_final` |
| 이상지질혈증 | CatBoost artifact | `catboost`, `dl_catboost_final` |
| 비만 | rule-based | `rule_based`, `web-precision-v1` |

간편분석(`BASIC`)은 rule-based 성격입니다. CatBoost 실제 호출 여부는 API 응답의 `model_name`, `model_version`, DB `analysis_results` 최신 row, audit script로 확인합니다.

확인 명령:

```bash
uv run python scripts/warmup_ml_models.py
uv run python scripts/audit_ai_runtime_capabilities.py --skip-warmup
```

### 식단분석

기본값:

```env
DIET_VISION_PROVIDER=rule_based
DIET_GPT_VISION_ENABLED=false
```

이 경우 실제 유료 Vision API 호출 없이 `rule_based_food_detection` + `DiseaseFoodScorer` + `nutrition_rule_table` 경로를 사용합니다.

GPT Vision 음식명 추론을 재현하려면:

```env
OPENAI_API_KEY=<key>
DIET_VISION_PROVIDER=gpt_vision
DIET_GPT_VISION_ENABLED=true
```

응답에서 확인할 필드:

- `detected_foods`
- `raw_output.source`
- `vision_provider`
- `fallback_used`
- `scoring_source`
- `disease_scores`
- `food_score_details`

프론트는 파일 업로드와 모바일 카메라 촬영 input을 모두 제공합니다.

### 건강검진 OCR

현재 공식 경로는 Clova OCR이 아닙니다. Clova provider 코드는 legacy/deferred/PoC로 보존합니다.

기본값:

```env
EXAM_OCR_PROVIDER=fallback
EXAM_GPT_VISION_ENABLED=false
PADDLE_OCR_ENABLED=false
```

GPT Vision 경로를 테스트하려면:

```env
OPENAI_API_KEY=<key>
EXAM_OCR_PROVIDER=gpt_vision
EXAM_GPT_VISION_ENABLED=true
```

PaddleOCR/text extraction 경로를 테스트하려면:

```env
EXAM_OCR_PROVIDER=paddleocr
PADDLE_OCR_ENABLED=true
```

주의:

- 텍스트 레이어가 있는 PDF는 pdf text extraction으로 성공할 수 있습니다.
- 실제 스캔 PDF OCR에는 PaddleOCR engine/paddlepaddle dependency와 이미지 변환 경로가 필요합니다.
- fallback 사용 시에는 `ocr_provider=fallback`, `fallback_used=true`로 표시되어야 하며, 실제 OCR 성공처럼 설명하지 않습니다.
- confirm 버튼을 누르면 OCR 후보값이 최신 `HealthRecord`에 반영될 수 있으므로 검진일 기준 수치가 맞는지 확인해야 합니다.

### 약봉투/처방전 OCR

기본값:

```env
MEDICATION_OCR_PROVIDER=fallback
MEDICATION_GPT_VISION_ENABLED=false
```

현재는 medication parser/fallback 구조를 중심으로 보존되어 있습니다. 실제 provider 연결은 GPT Vision/PaddleOCR 후보로 남아 있으며, fallback은 실제 OCR 성공으로 포장하지 않습니다.

### 챗봇 LLM

기본값은 rule-based입니다.

```env
CHATBOT_USE_REAL_LLM=false
```

실제 OpenAI rewrite 경로를 테스트하려면:

```env
OPENAI_API_KEY=<key>
CHATBOT_USE_REAL_LLM=true
```

기대 동작:

- flag false: `source=rule_engine`
- flag true + key missing: rule-based fallback
- flag true + key set: OpenAI 호출 경로 진입, 실패 시 fallback

### RAG keyword PoC

RAG source registry는 `docs/rag_sources/`에 있습니다. 원문 PDF 전체가 아니라 공식 출처 metadata와 짧은 요약만 보존합니다.

기본값:

```env
RAG_ENABLED=false
```

RAG를 켜면 keyword retrieval과 reference source 연결, Langfuse `rag.keyword_retrieval` trace가 활성화될 수 있습니다.

```env
RAG_ENABLED=true
```

운영형 vector RAG, embedding, pgvector retrieval은 P2 backlog입니다.

### Langfuse

Langfuse는 앱 실행 필수 구성요소가 아니라 선택 관측 도구입니다. RAG 엔진이 아니라 trace, prompt, evaluation metadata를 관리합니다. Full Docker dev stack은 Langfuse 없이도 실행됩니다.

self-host Langfuse 실행:

```bash
make langfuse-up
```

접속:

```text
http://localhost:3000
```

FastAPI가 Docker 컨테이너 안에서 host의 Langfuse에 접근할 때:

```env
LANGFUSE_ENABLED=true
LANGFUSE_BASE_URL=http://host.docker.internal:3000
LANGFUSE_PUBLIC_KEY=<self-host project public key>
LANGFUSE_SECRET_KEY=<self-host project secret key>
```

Cloud Langfuse 예:

```env
LANGFUSE_ENABLED=true
LANGFUSE_BASE_URL=https://jp.cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=<cloud public key>
LANGFUSE_SECRET_KEY=<cloud secret key>
```

Cloud key와 self-host key는 서로 호환되지 않습니다. 전환할 때는 key와 `LANGFUSE_BASE_URL`을 함께 바꿉니다.

## 이메일 인증

MVP/시연 회원가입은 이메일 인증만 필수입니다. 휴대폰 인증은 보류입니다.

로컬/시연 기본값:

```env
EMAIL_ENABLED=false
EMAIL_VERIFICATION_DEBUG=true
PASSWORD_RESET_DEBUG=false
```

이 상태에서는 실제 SMTP 발송 없이 인증코드가 debug 응답으로 표시될 수 있습니다.

Brevo SMTP 등 실제 이메일 발송을 쓰려면:

```env
EMAIL_ENABLED=true
EMAIL_VERIFICATION_DEBUG=false
PASSWORD_RESET_DEBUG=false
SMTP_HOST=<smtp host>
SMTP_PORT=587
SMTP_USERNAME=<smtp login>
SMTP_PASSWORD=<smtp key>
SMTP_FROM_EMAIL=<sender email>
SMTP_FROM_NAME=AI HealthCare
SMTP_USE_TLS=true
```

설정 확인:

```bash
uv run python scripts/verify_auth_delivery_config.py
```

실제 발송 테스트는 명시적으로 옵션을 붙였을 때만 수행합니다.

```bash
uv run python scripts/verify_auth_delivery_config.py --send-email you@example.com
```

운영 환경에서는 `EMAIL_VERIFICATION_DEBUG=false`, `PASSWORD_RESET_DEBUG=false`를 유지합니다.

## Async job skeleton

현재 Redis Stream 기반 async job은 `DEMO_ECHO` skeleton만 연결되어 있습니다.

- `/api/v1/jobs/demo`
- `/api/v1/jobs/{job_id}`
- `ai-worker` service
- `ai_runtime/jobs/*`

아직 `/analysis/run`, `/diets/analyze`, OCR confirm 흐름은 비동기로 넘기지 않습니다. `AnalysisResult.async_job_id`는 향후 async job 연동을 위한 reserved field입니다.

## 테스트 및 검증

기본 검증:

```bash
uv run ruff check app scripts ai_runtime tests
uv run ruff format app scripts ai_runtime tests --check
CHATBOT_USE_REAL_LLM=false OPENAI_API_KEY= LANGFUSE_ENABLED=false RAG_ENABLED=false uv run pytest tests
cd frontend
npm run build
```

프론트 단독 개발이 필요할 때만 Vite dev server를 별도로 실행합니다. 이 방식은 기본 시연 실행이 아니라 UI 디버깅용입니다.

```bash
cd frontend
npm install
npm run dev
```

API import/OpenAPI smoke:

```bash
uv run python -c "from app.main import app; print(app.title); print(len(app.openapi().get('paths', {})))"
```

시연 준비 확인:

```bash
uv run python scripts/verify_demo_ready.py
uv run python scripts/audit_ai_runtime_capabilities.py --skip-warmup
uv run python scripts/verify_auth_delivery_config.py
```

정밀분석 smoke:

```bash
uv run python scripts/verify_precision_analysis_api.py
```

주의: `verify_precision_analysis_api.py`는 기본 demo 계정 seed/env가 맞지 않으면 로그인 단계에서 실패할 수 있습니다. 이 경우 모델 실패가 아니라 demo seed/env 정렬 문제인지 먼저 구분합니다.

## 운영/배포 메모

운영 compose는 `infra/docker/docker-compose.prod.yml`을 기준으로 합니다.

운영 전 필수 확인:

- `.env`와 secret 관리 정책
- HTTPS 설정
- `REFRESH_TOKEN_COOKIE_SECURE=true`
- `CORS_ALLOW_ORIGINS` 제한
- production debug flag false
- Aerich PostgreSQL migration baseline 정책
- 관리자 계정/권한/감사 로그
- 실제 SMS provider 도입 여부
- OCR/CV/LLM 외부 provider 비용 제한과 timeout
- Langfuse Cloud/self-host key와 base URL 일치

## 자주 막히는 문제

### `http://localhost`가 404입니다

정상입니다. 프론트 포함 dev stack 주소는 `http://localhost:8080`입니다.

### 로그인 또는 회원가입 API가 500입니다

새 Docker volume이면 migration/seed가 안 들어갔을 수 있습니다.

```bash
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml exec fastapi uv run --no-sync aerich upgrade
DB_HOST=localhost uv run python scripts/setup_local_mvp_db.py
```

### demo/admin 계정 로그인이 안 됩니다

seed를 다시 확인합니다.

```bash
DB_HOST=localhost uv run python scripts/seed_demo_users.py
```

계정:

- `admin@example.com` / `Demo1234!`
- `monitor@example.com` / `Demo1234!`
- `demo@example.com` / `Demo1234!`

### OCR 결과가 fallback으로 나옵니다

`.env`의 provider flag를 확인합니다.

```env
EXAM_OCR_PROVIDER=gpt_vision
EXAM_GPT_VISION_ENABLED=true
OPENAI_API_KEY=<key>
```

또는 PaddleOCR/text extraction 경로:

```env
EXAM_OCR_PROVIDER=paddleocr
PADDLE_OCR_ENABLED=true
```

컨테이너 재생성:

```bash
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml up -d --build --force-recreate fastapi ai-worker nginx
```

### Langfuse trace가 안 찍힙니다

확인할 것:

- `LANGFUSE_ENABLED=true`
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` SET
- self-host/Cloud key와 `LANGFUSE_BASE_URL`이 같은 프로젝트를 바라보는지
- Docker FastAPI 기준 self-host URL은 보통 `http://host.docker.internal:3000`
- `RAG_ENABLED=false`이면 `rag.keyword_retrieval` trace는 생성되지 않는 것이 정상

### 프론트가 예전 화면을 보여줍니다

Docker frontend/nginx를 재빌드합니다.

```bash
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml up -d --build --force-recreate frontend nginx
```

브라우저 hard refresh도 함께 수행합니다.

## 개발 규칙

- 기능 코드와 실험/문서/seed를 구분합니다.
- 모델 artifact는 `ai_runtime/ml/artifacts/` 아래 runtime 경로에 둡니다.
- 원본 학습 데이터, 실험 산출물, OCR raw/reference 자료는 `etc/` 또는 외부 저장소로 분리합니다.
- 실제 secret은 절대 commit하지 않습니다.
- dependency 변경이 없으면 `uv.lock`을 건드리지 않습니다.
- migration은 기존 파일을 수정하지 않고 새 migration으로 관리합니다.
- 시연 중 DB volume 삭제 명령(`down -v`)은 사용하지 않습니다.
- 테스트 fixture/mock은 유지하되 사용자 화면/공식 API 응답에 dummy/mock/stub 결과가 실제 결과처럼 보이지 않게 합니다.

## 참고 문서

- 시연 시나리오: `docs/demo/scenario.md`
- 시연 체크리스트: `docs/qa/demo_ready_checklist.md`
- 프론트 QA 체크리스트: `docs/qa/frontend_demo_checklist.md`
- LLM/RAG runtime 범위: `docs/design/llm_runtime_scope.md`
- AI runtime 모델 범위: `docs/design/ai_worker_model_scope.md`
- Docker stack 설명: `docs/ops/docker_stacks.md`
- Secret 관리: `docs/ops/secrets_handling.md`
- DB migration 정책: `docs/ops/database_migration_policy.md`
- release readiness 요약: `docs/release/feature_kdu_demo_readiness_summary.md`
