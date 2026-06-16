# AI HealthCare MVP

AI HealthCare MVP는 건강정보 입력, 건강검진 OCR, 식단/복약 분석, 챗봇, 만성질환 위험 안내를 하나의 웹 서비스 흐름으로 묶은 FastAPI + React 기반 시연 프로젝트입니다.

이 README는 팀원이 clone 후 바로 실행할 수 있는 Quick Start 문서입니다. 상세 설계와 운영 설명은 하단의 참고 문서를 확인하세요.

## 빠른 실행 요약

Docker 통합 개발:

```bash
cp envs/example.local.env .env
make dev-up
make dev-migrate
make dev-seed
make dev-health
```

접속:

- Web: `http://localhost:8080`
- API Docs: `http://localhost:8080/api/docs`
- Health: `http://localhost:8080/api/v1/system/health`

프론트 로컬 개발:

```bash
# 터미널 1
make dev-stack

# 터미널 2
make dev-front
```

접속:

- Frontend Vite dev server: `http://localhost:5173`
- Docker/Nginx 통합 확인: `http://localhost:8080`

## 사전 준비

- Docker Desktop 또는 Docker Engine
- Python 3.13 이상
- `uv`

Full Docker dev stack은 프론트엔드, Nginx, FastAPI, AI Worker, PostgreSQL, Redis를 함께 실행합니다. 웹 시연만 할 때는 Node.js/npm을 직접 실행할 필요가 없습니다. 프론트엔드는 Docker build 단계에서 정적 파일로 빌드되고, frontend 컨테이너 내부 Nginx가 이를 서빙합니다.

프론트 개발 중에는 Docker frontend를 매번 rebuild하지 않고 로컬 Vite dev server를 권장합니다. 이때 Backend/FastAPI, Postgres, Redis, ai-worker, Nginx는 Docker로 실행하고, 화면은 `http://localhost:5173`에서 확인합니다.

로컬에서 테스트나 스크립트를 직접 실행할 때만 의존성을 동기화합니다.

```bash
uv sync --group app --group ai --group dev
```

## 환경변수 준비

팀원/시연 기준은 루트 `.env`입니다.

```bash
cp envs/example.local.env .env
```

주의:

- `.env`와 secret은 절대 commit하지 않습니다.
- `.prod.env`도 절대 commit하지 않습니다.
- 실제 OpenAI key, SMTP password, Langfuse secret 값은 README나 PR에 넣지 않습니다.
- secret 예시는 `<OPENAI_API_KEY>`, `<SMTP_PASSWORD>`처럼 placeholder로만 작성합니다.
- `docker compose config` 전체 출력은 secret이 펼쳐질 수 있으므로 문서/화면공유에 사용하지 마세요.
- 운영 배포용 `.prod.env`도 GitHub에 올리지 않습니다. `envs/example.prod.env`는 변수명과 placeholder를 보여주는 템플릿입니다.

최소 dev 예시:

```env
ENV=local
DB_HOST=postgres
DB_PORT=5432
DB_USER=ozcoding
DB_PASSWORD=pw1234
DB_NAME=ai_health

EMAIL_ENABLED=false
EMAIL_VERIFICATION_DEBUG=true
PASSWORD_RESET_DEBUG=false

OPENAI_API_KEY=<OPENAI_API_KEY>
CHATBOT_USE_REAL_LLM=false
RAG_ENABLED=false
RAG_RETRIEVAL_STRATEGY=keyword_only

DIET_VISION_PROVIDER=rule_based
DIET_GPT_VISION_ENABLED=false

EXAM_OCR_PROVIDER=auto
EXAM_GPT_VISION_ENABLED=false
PADDLE_OCR_ENABLED=false

LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=<LANGFUSE_PUBLIC_KEY>
LANGFUSE_SECRET_KEY=<LANGFUSE_SECRET_KEY>
```

## Full Docker Dev Stack 실행

팀원/시연 표준은 `infra/docker/docker-compose.dev.yml`입니다.

| 용도 | Compose 파일 | 실행 기준 | 비고 |
|---|---|---|---|
| 개발/시연 표준 | `infra/docker/docker-compose.dev.yml` | `make dev-up` | frontend, nginx, fastapi, ai-worker, postgres, redis 포함 |
| 운영 배포 | `infra/docker/docker-compose.prod.yml` | Docker Hub image pull | app, ai-worker, frontend 이미지를 먼저 push해야 함 |
| legacy/minimal 검증 | `docker-compose.yml` | `make app-*` | backend/AI 빠른 확인용, 표준 dev stack 아님 |

```bash
make dev-up
```

동일한 Docker 통합 stack을 직접 compose로 실행해야 할 때는 아래 명령을 사용합니다.

```bash
cp envs/example.local.env .env
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml up -d --build
make dev-health
```

접속:

- Docker 통합 Web: `http://localhost:8080`

중지:

```bash
make dev-down
```

중요:

- 루트 `docker-compose.yml`은 legacy/minimal backend/AI 검증용입니다.
- 팀원/시연 표준 실행은 반드시 `infra/docker/docker-compose.dev.yml`을 사용합니다.
- Docker Compose 직접 명령 대신 README의 `make dev-*`, `make prod-*` 명령을 사용하세요.
- `docker compose up -d`만 단독 실행하지 마세요. 의도와 다른 루트 compose가 실행될 수 있습니다.
- `docker rm -f redis postgres fastapi ai-worker nginx` 방식으로 컨테이너를 직접 지우지 마세요.
- `down -v`는 DB volume을 삭제할 수 있으므로 시연/협업 중 사용하지 마세요.

## 프론트 로컬 개발

프론트 팀원이 UI를 빠르게 개발할 때는 Docker frontend rebuild 대신 로컬 Vite dev server를 사용합니다.

터미널 1에서 Docker 백엔드 개발 스택을 실행합니다.

```bash
make dev-stack
```

터미널 2에서 로컬 Vite dev server를 실행합니다.

```bash
make dev-front
```

한 터미널에서 이어서 실행하려면 아래 명령을 사용할 수 있습니다. `npm run dev`가 foreground로 실행되므로 터미널을 점유합니다.

```bash
make dev-local
```

접속:

- 프론트 로컬 개발: `http://localhost:5173`
- API/Nginx 통합: `http://localhost:8080`
- Docker frontend 빌드 결과 확인: `http://localhost:8080`

`localhost:5173`은 Vite dev server라 프론트 수정사항이 바로 반영됩니다. `frontend/vite.config.ts`에서 `/api` 요청은 `http://localhost:8080`으로 proxy됩니다. 따라서 화면은 `localhost:5173`으로 접속하되, API는 Docker Nginx/FastAPI를 사용합니다.

프론트 개발 중에는 매번 Docker frontend를 rebuild하지 않아도 됩니다. 최종 통합 확인이 필요할 때만 Docker frontend 이미지를 다시 빌드합니다.

```bash
make dev-rebuild-frontend
make dev-health
```

## DB Migration / Seed

DB migration:

```bash
make dev-migrate
```

챌린지 seed:

```bash
make dev-seed
```

`make dev-seed`는 `docs/data/challenges/team_challenge_master.csv` 기준으로 챌린지를 반영합니다. CSV에 없는 기존 active 챌린지는 seed 정책상 inactive 처리됩니다. DB migration이 아니라 seed 데이터 반영입니다.

데모 seed:

```bash
DB_HOST=localhost uv run python scripts/setup_local_mvp_db.py
```

## 접속 주소

- Frontend local dev: `http://localhost:5173`
- Docker integrated Web: `http://localhost:8080`
- API Docs: `http://localhost:8080/api/docs`
- API Health: `http://localhost:8080/api/v1/system/health`
- FastAPI 직접 접근: `http://localhost:8000`
- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`

## Health Check

```bash
make dev-health
```

컨테이너 상태 확인:

```bash
make dev-ps
```

로그 확인 예시:

```bash
make dev-logs
```

백엔드/AI Worker 코드나 env를 수정한 뒤 빠르게 반영:

```bash
make dev-rebuild-api
```

AI Worker만 빠르게 재빌드:

```bash
make dev-rebuild-worker
```

프론트엔드 UI/정적 빌드 변경만 Docker에 반영:

```bash
make dev-rebuild-frontend
```

프론트엔드, 백엔드, worker 변경이 섞였을 때 전체 dev stack 재빌드:

```bash
make dev-rebuild-all
```

Nginx 502 또는 upstream 캐시가 꼬였을 때 복구:

```bash
make dev-restart-nginx
```

dev compose 설정 검증:

```bash
make dev-config-check
```

`make dev-config-check`는 `docker compose config --quiet`만 실행해 compose 유효성만 확인합니다. secret이 펼쳐질 수 있는 `docker compose config` 전체 출력은 문서/화면공유에 사용하지 마세요.

로컬 QA 검증:

```bash
make qa-diff
make qa-backend
make qa-frontend
```

## Prod Image / EC2 배포 준비

운영 compose는 `infra/docker/docker-compose.prod.yml` 기준입니다. EC2에서 이미지를 build하지 않고 Docker Hub에서 pull합니다. secret, password, 서버별 URL은 이미지에 넣지 않고 `.prod.env` 같은 배포 환경 파일로 주입합니다. 실제 `.prod.env`는 commit하지 않고, `envs/example.prod.env`를 템플릿으로 사용하세요.

기본 이미지 tag:

- `kdu0312/ai-health:app-v1.0.0`
- `kdu0312/ai-health:ai-v1.0.0`
- `kdu0312/ai-health:frontend-v1.0.0`

배포 전 로컬 build 검증:

```bash
make image-tags
make image-build-check
docker image inspect kdu0312/ai-health:app-v1.0.0 --format '{{.Os}}/{{.Architecture}}'
docker image inspect kdu0312/ai-health:ai-v1.0.0 --format '{{.Os}}/{{.Architecture}}'
docker image inspect kdu0312/ai-health:frontend-v1.0.0 --format '{{.Os}}/{{.Architecture}}'
```

기대값은 `linux/amd64`입니다. Apple Silicon 로컬에서도 EC2 Ubuntu 배포 이미지는 `make image-build-check`의 buildx `linux/amd64` 기준으로 확인합니다.

운영 배포 env 준비와 실행:

```bash
cp envs/example.prod.env .prod.env
# .prod.env 실제 운영값 수정
make prod-pull
make prod-up
make prod-migrate
make prod-health
make prod-ps
make prod-logs
```

DuckDNS 배포 기준 도메인은 `healthladder.duckdns.org`입니다. `.prod.env`에는 같은 도메인 기준으로 아래 public URL 값을 맞춥니다.

```env
COOKIE_DOMAIN=healthladder.duckdns.org
CORS_ALLOW_ORIGINS=https://healthladder.duckdns.org
FRONTEND_BASE_URL=https://healthladder.duckdns.org
VITE_API_BASE_URL=/api/v1
```

DuckDNS token은 secret이므로 README, PR, issue, shell history, 배포 로그에 남기지 않습니다. DNS/HTTP 확인은 실제 배포 전 운영자가 아래처럼 확인합니다.

```bash
nslookup healthladder.duckdns.org
curl -I http://healthladder.duckdns.org
curl -I https://healthladder.duckdns.org
curl -fsS https://healthladder.duckdns.org/api/v1/system/health
```

초기 환경에서 챌린지 seed가 반드시 필요하고 운영자가 명시적으로 승인한 경우에만 아래 danger target을 따로 실행합니다.

```bash
make danger-prod-seed
```

`make prod-release-db`는 하위 호환용으로 남아 있지만 이제 migration만 실행합니다. 운영/배포용 target에서는 seed와 RAG ingest를 자동 실행하지 않습니다.

QA/smoke 스크립트는 로컬 검증용입니다. `scripts/qa/smoke_notification_email.py` 같은 파일은 운영 이미지 런타임 필수 파일이 아니며, 실제 발송 smoke는 운영자가 명시적으로 허용한 별도 절차에서만 실행합니다.

상세 절차는 `docs/deployment/ec2_docker_deploy_guide.md`, `docs/deployment/ec2_prod_env.md`, `docs/ops/docker_stacks.md`를 참고하세요.

## 주요 Provider Flag 요약

| 기능 | 주요 flag | 기본 방향 |
|---|---|---|
| 챗봇 LLM | `CHATBOT_USE_REAL_LLM`, `OPENAI_API_KEY` | 기본 mock/rule, 필요 시 OpenAI |
| RAG | `RAG_ENABLED`, `RAG_RETRIEVAL_STRATEGY` | 기본 keyword-only, 필요 시 vector fallback/hybrid |
| 식단 이미지 | `DIET_VISION_PROVIDER`, `DIET_GPT_VISION_ENABLED` | 기본 rule/fallback |
| 건강검진 OCR | `EXAM_OCR_PROVIDER`, `EXAM_GPT_VISION_ENABLED`, `PADDLE_OCR_ENABLED` | 기본 auto |
| 복약 정보 | 별도 OCR provider 없음 | MVP에서는 사용자가 직접 입력 |
| Langfuse | `LANGFUSE_ENABLED`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | 기본 off |
| 이메일 | `EMAIL_ENABLED`, `SMTP_*`, `EMAIL_VERIFICATION_DEBUG` | local은 debug 또는 SMTP |

provider별 상세 정책은 `docs/design/*`, `docs/ops/*` 문서를 참고하세요.

챗봇 응답은 현재 `/api/v1/chatbot/ask`의 POST JSON 응답을 사용합니다. SSE/streaming은 아직
미구현이며, Bearer token 인증에서는 브라우저 `EventSource`에 Authorization header를 붙이기
어렵기 때문에 추후에는 `POST + fetch ReadableStream` 방식을 우선 검토합니다.

복약 정보는 사용자가 직접 입력합니다. 약물 정보 확인은 약학정보원/약찾기 서비스를 참고하고,
본 서비스는 처방, 복용량 판단, 약물 변경 안내를 제공하지 않습니다. 복약 관련 의사결정은
의사 또는 약사와 상담해야 합니다.

## 이메일 인증 / ai-worker 주의사항

이메일 발송 job은 FastAPI가 직접 처리하지 않습니다.

흐름:

```text
FastAPI 요청
→ async_jobs DB row 생성
→ Redis Stream service job enqueue
→ ai-worker가 SMTP service job 처리
```

따라서 실제 이메일 발송을 켜려면 SMTP 환경변수가 `fastapi`와 `ai-worker` 양쪽에 전달되어야 합니다. 서비스 내부 알림은 앱 DB에 저장되며, 브라우저 Push 알림과 SMS 알림은 MVP 범위에서 제외합니다. 표준 dev stack은 `make dev-up`으로 실행합니다.

```bash
make dev-up
```

로컬에서 실제 SMTP를 쓰지 않을 때는 다음처럼 debug 모드를 사용합니다.

```env
EMAIL_ENABLED=false
EMAIL_VERIFICATION_DEBUG=true
```

실제 SMTP 발송 예시:

```env
EMAIL_ENABLED=true
EMAIL_VERIFICATION_DEBUG=false
SMTP_HOST=<SMTP_HOST>
SMTP_PORT=587
SMTP_USERNAME=<SMTP_USERNAME>
SMTP_PASSWORD=<SMTP_PASSWORD>
SMTP_FROM_EMAIL=<SMTP_FROM_EMAIL>
SMTP_FROM_NAME=<SMTP_FROM_NAME>
SMTP_USE_TLS=true
```

현재 MVP에서는 별도 `email-worker`, `notification-worker`, `scheduler-worker` 없이 `ai-worker` 하나가 AI/OCR/ML job과 service job을 함께 처리합니다. service job에는 이메일 인증, 비밀번호 재설정, 가족 초대, 가족 알림 생성이 포함됩니다. `SCHEDULER_ENABLED=true`이면 notification scheduler도 `ai-worker` 안에서 함께 실행됩니다.

운영 안정화 단계에서는 AI/OCR job 적체가 인증 이메일이나 가족 알림 생성을 지연시키지 않도록 `email-worker`, `notification-worker`, `scheduler-worker` 분리를 목표 구조로 둡니다. 현재 구조와 분리 계획은 `docs/ops/docker_stacks.md`, 배포 전 확인 항목은 `docs/deployment/pre_deploy_checklist.md`를 참고하세요.

## 자주 막히는 문제

### `docker compose up -d`만 실행해서 서비스가 이상하게 뜸

루트 `docker-compose.yml`은 표준 dev stack이 아닙니다. 아래 명령을 사용하세요.

```bash
make dev-up
```

### 이메일 인증 버튼은 눌리지만 메일이 안 옴

- `ai-worker`가 떠 있는지 확인합니다.
- `--env-file .env`로 compose를 실행했는지 확인합니다.
- `EMAIL_ENABLED`, `SMTP_*`, `EMAIL_VERIFICATION_DEBUG` 설정을 확인합니다.
- `docker compose config` 전체 출력은 secret 노출 위험이 있으므로 공유하지 않습니다.

```bash
make dev-ps
make dev-logs
```

### migration 명령에서 `aerich`를 못 찾음

컨테이너 내부에서 실행하세요.

```bash
make dev-migrate
```

### DB를 초기화하고 싶음

시연/협업 중에는 `down -v`를 쓰지 마세요. DB volume이 삭제될 수 있습니다. 필요한 경우 팀원과 먼저 합의하고 `docs/ops/database_migration_policy.md`를 확인하세요.

## 참고 문서 링크

- Docker stack 정책: `docs/ops/docker_stacks.md`
- Secret 처리: `docs/ops/secrets_handling.md`
- DB migration 정책: `docs/ops/database_migration_policy.md`
- Langfuse self-host: `docs/ops/langfuse_self_host.md`
- AI Worker / Redis Stream 구조: `docs/design/ai_worker_structure.md`
- LLM / LangGraph runtime 계획: `docs/design/llm_langgraph_runtime_plan.md`
- CV food fallback 정책: `docs/design/cv_food_fallback_policy.md`
- 시연 시나리오: `docs/demo/scenario.md`
- 시연 준비 체크리스트: `docs/demo/ready_checklist.md`
