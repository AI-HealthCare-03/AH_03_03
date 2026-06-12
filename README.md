# AI HealthCare MVP

AI HealthCare MVP는 건강정보 입력, 건강검진 OCR, 식단/복약 분석, 챗봇, 만성질환 위험 안내를 하나의 웹 서비스 흐름으로 묶은 FastAPI + React 기반 시연 프로젝트입니다.

이 README는 팀원이 clone 후 바로 실행할 수 있는 Quick Start 문서입니다. 상세 설계와 운영 설명은 하단의 참고 문서를 확인하세요.

## 빠른 실행 요약

```bash
cp envs/example.local.env .env
uv sync
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml up -d --build
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml exec fastapi uv run --no-sync aerich upgrade
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml exec fastapi uv run --no-sync python scripts/seed_mvp_challenges.py
curl -fsS http://localhost:8080/api/v1/system/health
```

접속:

- Web: `http://localhost:8080`
- API Docs: `http://localhost:8080/api/docs`
- Health: `http://localhost:8080/api/v1/system/health`

## 사전 준비

- Docker Desktop 또는 Docker Engine
- Python 3.13 이상
- `uv`

Full Docker dev stack은 프론트엔드, Nginx, FastAPI, AI Worker, PostgreSQL, Redis를 함께 실행합니다. 웹 시연만 할 때는 Node.js/npm을 직접 실행할 필요가 없습니다.

로컬 스크립트 실행과 의존성 동기화를 위해 한 번 실행합니다.

```bash
uv sync
```

## 환경변수 준비

팀원/시연 기준은 루트 `.env`입니다.

```bash
cp envs/example.local.env .env
```

주의:

- `.env`와 secret은 절대 commit하지 않습니다.
- 실제 OpenAI key, SMTP password, Langfuse secret, Firebase service account 값은 README나 PR에 넣지 않습니다.
- secret 예시는 `<OPENAI_API_KEY>`, `<SMTP_PASSWORD>`처럼 placeholder로만 작성합니다.
- `docker compose config` 전체 출력은 secret이 펼쳐질 수 있으므로 문서/화면공유에 사용하지 마세요.

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

DIET_VISION_PROVIDER=rule_based
DIET_GPT_VISION_ENABLED=false

EXAM_OCR_PROVIDER=auto
EXAM_GPT_VISION_ENABLED=false
PADDLE_OCR_ENABLED=false

MEDICATION_OCR_PROVIDER=fallback
MEDICATION_GPT_VISION_ENABLED=false

LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=<LANGFUSE_PUBLIC_KEY>
LANGFUSE_SECRET_KEY=<LANGFUSE_SECRET_KEY>
```

## Full Docker Dev Stack 실행

팀원/시연 표준은 `infra/docker/docker-compose.dev.yml`입니다.

```bash
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml up -d --build
```

중지:

```bash
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml down --remove-orphans
```

중요:

- 루트 `docker-compose.yml`은 legacy/minimal backend/AI 검증용입니다.
- 팀원/시연 표준 실행은 반드시 `infra/docker/docker-compose.dev.yml`을 사용합니다.
- `docker compose up -d`만 단독 실행하지 마세요. 의도와 다른 루트 compose가 실행될 수 있습니다.
- `docker rm -f redis postgres fastapi ai-worker nginx` 방식으로 컨테이너를 직접 지우지 마세요.
- `down -v`는 DB volume을 삭제할 수 있으므로 시연/협업 중 사용하지 마세요.

## DB Migration / Seed

DB migration:

```bash
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml exec fastapi uv run --no-sync aerich upgrade
```

챌린지 seed:

```bash
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml exec fastapi uv run --no-sync python scripts/seed_mvp_challenges.py
```

데모 seed:

```bash
DB_HOST=localhost uv run python scripts/setup_local_mvp_db.py
```

## 접속 주소

- Web: `http://localhost:8080`
- API Docs: `http://localhost:8080/api/docs`
- API Health: `http://localhost:8080/api/v1/system/health`
- FastAPI 직접 접근: `http://localhost:8000`
- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`

## Health Check

```bash
curl -fsS http://localhost:8080/api/v1/system/health
```

컨테이너 상태 확인:

```bash
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml ps
```

로그 확인 예시:

```bash
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml logs fastapi --tail=100
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml logs ai-worker --tail=100
```

## 주요 Provider Flag 요약

| 기능 | 주요 flag | 기본 방향 |
|---|---|---|
| 챗봇 LLM | `CHATBOT_USE_REAL_LLM`, `OPENAI_API_KEY` | 기본 mock/rule, 필요 시 OpenAI |
| RAG | `RAG_ENABLED` | 기본 off |
| 식단 이미지 | `DIET_VISION_PROVIDER`, `DIET_GPT_VISION_ENABLED` | 기본 rule/fallback |
| 건강검진 OCR | `EXAM_OCR_PROVIDER`, `EXAM_GPT_VISION_ENABLED`, `PADDLE_OCR_ENABLED` | 기본 auto |
| 복약 OCR | `MEDICATION_OCR_PROVIDER`, `MEDICATION_GPT_VISION_ENABLED` | 기본 fallback |
| Langfuse | `LANGFUSE_ENABLED`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | 기본 off |
| 이메일 | `EMAIL_ENABLED`, `SMTP_*`, `EMAIL_VERIFICATION_DEBUG` | local은 debug 또는 SMTP |

provider별 상세 정책은 `docs/design/*`, `docs/ops/*` 문서를 참고하세요.

## 이메일 인증 / ai-worker 주의사항

이메일 발송 job은 FastAPI가 직접 처리하지 않습니다.

흐름:

```text
FastAPI 요청
→ async_jobs DB row 생성
→ Redis Stream service job enqueue
→ ai-worker가 SMTP 발송
```

따라서 실제 이메일 발송을 켜려면 SMTP 환경변수가 `fastapi`와 `ai-worker` 양쪽에 전달되어야 합니다. dev stack을 실행할 때는 반드시 아래처럼 `--env-file .env`를 붙이세요.

```bash
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml up -d --build
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

## 자주 막히는 문제

### `docker compose up -d`만 실행해서 서비스가 이상하게 뜸

루트 `docker-compose.yml`은 표준 dev stack이 아닙니다. 아래 명령을 사용하세요.

```bash
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml up -d --build
```

### 이메일 인증 버튼은 눌리지만 메일이 안 옴

- `ai-worker`가 떠 있는지 확인합니다.
- `--env-file .env`로 compose를 실행했는지 확인합니다.
- `EMAIL_ENABLED`, `SMTP_*`, `EMAIL_VERIFICATION_DEBUG` 설정을 확인합니다.
- `docker compose config` 전체 출력은 secret 노출 위험이 있으므로 공유하지 않습니다.

```bash
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml ps
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml logs ai-worker --tail=100
```

### migration 명령에서 `aerich`를 못 찾음

컨테이너 내부에서 실행하세요.

```bash
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml exec fastapi uv run --no-sync aerich upgrade
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
