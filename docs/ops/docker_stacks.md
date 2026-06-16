# Docker Stacks

이 프로젝트는 용도별 Docker Compose 구성을 분리한다. 시연/개발/관측 도구를 한 파일에 모두 넣지 않고, 필요한 스택만 선택해서 실행한다.

## 1. 스택 요약

| 스택 | Compose 파일 | 용도 | 주요 서비스 |
| --- | --- | --- | --- |
| `app` | `docker-compose.yml` | legacy/minimal backend/AI 검증용 스택 | `postgres`, `redis`, `fastapi` |
| `dev` | `infra/docker/docker-compose.dev.yml` | 표준 로컬 개발/시연 전체 스택 | `postgres`, `redis`, `fastapi`, `ai-worker`, `frontend`, `nginx` |
| `prod` | `infra/docker/docker-compose.prod.yml` | 표준 EC2/운영 이미지 기반 스택 | `postgres`, `redis`, `fastapi`, `ai-worker`, `frontend`, `nginx`, `certbot` |
| `langfuse` | `infra/langfuse/docker-compose.yml` | Langfuse self-host optional 관측 스택 | `langfuse-web`, `langfuse-worker`, `postgres`, `redis`, `clickhouse`, `minio` |

`Makefile`은 표준 dev full stack 실행을 감싼다. `scripts/docker_stack.sh`는 legacy `app` 스택과 optional `langfuse` 스택 wrapper로 보존한다. `prod`는 이미지 태그와 운영 환경변수 확인이 필요하므로 명시적으로 compose 파일을 지정해서 실행한다.

현재 표준 로컬 실행은 `make dev-up`이며 내부적으로 `infra/docker/docker-compose.dev.yml`을 사용한다. 루트 `docker-compose.yml`은 legacy/minimal backend/AI 검증용으로만 유지하며, frontend, storage, scheduler, 최신 dev full stack 검증에는 사용하지 않는다.

## 현재 Redis / async job 범위

현재 앱 스택의 Redis는 health check뿐 아니라 Redis Stream 기반 job 처리에도 사용한다.

- Docker Compose Redis 컨테이너 실행
- FastAPI의 `REDIS_HOST`, `REDIS_PORT` 기반 연결
- `/api/v1/system/health` Redis 연결 확인
- Compose `redis` service healthcheck
- AI stream: `analysis.run`, `exam_ocr.run`, `diet.analyze_image`
- Service stream: `email.verification.send`, `password_reset.email.send`, `family.invite.email.send`, `family.notification.create`
- retry/backoff, DLQ, pending recovery 기반 `ai-worker` consumer

P2 운영 확장 범위로 남은 항목은 대시보드/알림 수준의 운영 관측, queue 지표 노출, alerting, worker 수평 확장 정책이다.

시연 설명 문구:

> 현재 MVP는 긴 OCR/식단/분석/외부 발송 작업을 Redis Stream async job으로 넘기고, `/api/v1/jobs/{job_id}` polling으로 상태를 확인합니다. 긴 작업의 기존 동기 분석 API는 410 Gone으로 막고 async 전용 경로를 사용합니다.

## 현재 worker 구조와 분리 계획

현재 MVP 배포에서는 `notification-worker`, `email-worker`, `scheduler-worker` 같은 별도 Compose service를 두지 않고 `ai-worker` 통합 구조를 유지한다. dev/prod/root compose 모두 `ai-worker` 하나가 Redis Stream consumer를 실행하며, AI/OCR/ML job뿐 아니라 service job도 함께 처리한다.

현재 `ai-worker`가 처리하는 service job:

- `email.verification.send`
- `password_reset.email.send`
- `family.invite.email.send`
- `family.notification.create`

또한 `SCHEDULER_ENABLED=true`일 때 notification scheduler도 `ai-worker` 프로세스 안에서 함께 실행된다.

현재 구조의 리스크:

- `ai-worker`가 무거운 OCR/GPT Vision/ML 작업으로 밀리면 인증 이메일, 비밀번호 재설정 이메일, 가족 알림도 함께 지연될 수 있다.
- 이메일 인증은 회원가입 UX와 직접 연결되므로 AI/OCR job 장애와 분리하는 편이 운영 안정성에 유리하다.
- scheduler loop와 Redis Stream consumer가 같은 프로세스에 있으므로, 장기적으로 장애 격리와 스케일링 단위가 거칠다.

향후 목표 구조:

| worker | 담당 |
| --- | --- |
| `ai-worker` | OCR, GPT Vision, ML inference, analysis job |
| `email-worker` 또는 `service-worker` | 이메일 인증 코드, 비밀번호 재설정 이메일, 가족 초대 이메일 |
| `notification-worker` | 가족 알림 생성, `notification_logs` 기록 |
| `scheduler-worker` | 예약 알림, 리마인더 스케줄링, `SCHEDULER_ENABLED` 기반 주기 작업 |

장기 TODO:

- Redis Stream job type별 consumer group 또는 worker command 분리 정책을 정한다.
- Compose prod/dev에 worker별 service를 추가하되, 최초 전환 시에는 기존 `ai-worker` 통합 모드와 병행 가능한 migration path를 둔다.
- 이메일/알림 worker는 AI model dependency 없이 가벼운 image 또는 command로 분리할 수 있는지 검토한다.
- worker별 로그, health check, queue 지표, DLQ 확인 절차를 배포 체크리스트에 반영한다.

## Provider Availability 정리 방향

장기적으로 앱 `.env`는 secret, key, URL, password, host, port 같은 실행 환경값 중심으로 유지한다. provider 사용 가능 여부는 코드의 availability helper가 판단하고, key나 runtime dependency가 없으면 fallback 또는 no-op으로 처리한다.

현재 1단계에서는 `OPENAI_API_KEY`, `SMTP_*`, `LANGFUSE_*` 설정 완성 여부를 helper로 확인한다. 다만 `EMAIL_ENABLED`, `CHATBOT_USE_REAL_LLM`, `DIET_GPT_VISION_ENABLED`, `EXAM_GPT_VISION_ENABLED`, `PADDLE_OCR_ENABLED`, `LANGFUSE_ENABLED`, `SCHEDULER_ENABLED`, `RAG_ENABLED`, `GPT_VISION_FALLBACK_ENABLED` 같은 기존 flag는 호환성과 비용/발송/디버그 안전장치 때문에 아직 유지한다.

이메일은 회원가입 인증, 비밀번호 재설정, 가족 초대/비회원 가족 초대에서 공통으로 쓰는 핵심 delivery provider다. 따라서 SMTP provider availability는 `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM_EMAIL`이 모두 준비된 경우에만 true로 본다. local/dev에서 SMTP 설정이 없으면 실제 발송 대신 debug 응답 또는 로그 흐름으로 개발할 수 있지만, prod에서는 조용한 no-op으로 넘기지 않고 configuration error를 유지한다. `EMAIL_ENABLED`는 1단계 호환성 flag로 남기되, 장기적으로는 SMTP availability와 `ENV` 기준으로 대체한다. `EMAIL_VERIFICATION_DEBUG`, `PASSWORD_RESET_DEBUG`는 production에서 강제 off되는 보안 가드로 유지한다.

후속 정리 방향:

- GPT Vision 계열 flag는 각 provider enum과 pipeline policy로 흡수한다.
- PaddleOCR는 import/runtime dependency availability로 대체한다.
- Scheduler는 `ai-worker` consumer와 별도 `scheduler-worker` service/command로 분리한다.
- Debug 응답 flag는 production에서 강제 off되는 안전장치로 유지한다.

## 2. 빠른 실행

### app 스택

루트 `docker-compose.yml`을 사용하는 legacy/minimal backend/AI 검증용 스택이다. FastAPI, PostgreSQL, Redis만 빠르게 띄우며 `ai-worker`, `frontend`, `nginx`는 기본으로 올리지 않는다.

이 스택은 FastAPI 중심 검증용이지만 frontend, storage, scheduler까지 포함한 최신 dev full stack 검증에는 사용하지 않는다. 실제 프론트 포함 비동기 UX 확인은 `dev` 스택에서 `ai-worker`, `frontend`, `nginx`까지 함께 올려 검증한다.

```bash
./scripts/docker_stack.sh app up
./scripts/docker_stack.sh app up-full
./scripts/docker_stack.sh app worker-up
./scripts/docker_stack.sh app build
./scripts/docker_stack.sh app worker-build
./scripts/docker_stack.sh app rebuild
./scripts/docker_stack.sh app clean-image
./scripts/docker_stack.sh app ps
./scripts/docker_stack.sh app logs
./scripts/docker_stack.sh app worker-logs
./scripts/docker_stack.sh app down
```

Makefile:

```bash
make app-up
make app-up-full
make app-worker-up
make app-build
make app-worker-build
make app-rebuild
make app-clean-image
make app-ps
make app-logs
make app-worker-logs
make app-down
```

직접 compose 명령:

```bash
docker compose -f docker-compose.yml up -d postgres redis fastapi
```

`app up`은 구버전 FastAPI 이미지가 남아 최신 `ai_runtime` 파일이 반영되지 않는 상황을 피하기 위해 `--build`를 함께 실행한다. `ai_runtime` 코드, CatBoost artifact, FastAPI Dockerfile, `pyproject.toml`, `uv.lock`이 바뀐 뒤에는 아래 명령 중 하나를 실행한다.

```bash
make app-build       # fastapi 이미지만 일반 재빌드
make app-rebuild     # fastapi 이미지를 no-cache로 재빌드
make app-clean-image # 알려진 로컬 app/test 이미지를 삭제
```

`make app-clean-image`는 아래 이미지를 대상으로 한다.

- `kdu0312/ai-health:app-v1.0.0`
- `test-ai-health:latest`

이미지가 없으면 건너뛴다. 실행 중인 컨테이너가 이미지를 사용 중이면 Docker가 삭제를 거부할 수 있으므로 먼저 `make app-down`으로 컨테이너를 내린 뒤 다시 실행한다.

### dev 스택

frontend, Nginx, FastAPI, AI Worker, PostgreSQL, Redis를 모두 올린다.

`ai-worker` service는 `ai_runtime/main.py`를 통해 Redis Stream consumer와 scheduler loop를 실행한다. 처리 job은 `DEMO_ECHO`뿐 아니라 `analysis.run`, `exam_ocr.run`, `diet.analyze_image`, 이메일/비밀번호/가족초대/가족알림 service job을 포함한다.

로컬 개발/시연 표준은 루트 `.env` 하나를 사용한다. 시작할 때 `envs/example.local.env`를 복사하고, 실제 secret 값은 `.env`에만 채운다.

```bash
cp envs/example.local.env .env
```

직접 compose 명령을 사용할 때는 Docker Compose variable interpolation이 `env_file`이 아니라 `--env-file` 또는 루트 `.env`를 기준으로 동작한다는 점에 주의한다.

```bash
make dev-up
make dev-ps
make dev-down
```

Makefile:

```bash
make dev-up
make dev-ps
make dev-logs
make dev-migrate
make dev-seed
make dev-health
make dev-rebuild-worker
make dev-down
```

`make dev-up`은 `ai-health-shared` Docker network가 없으면 생성한 뒤 아래 compose를 실행한다.

```bash
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml up -d --build
```

프론트엔드는 Vite dev server가 아니라 `frontend/Dockerfile`에서 정적 파일로 빌드된 뒤 frontend 컨테이너 내부 Nginx로 서빙된다. dev Nginx는 기본 `http://localhost:8080`에서 `/api/`를 FastAPI로, 나머지 요청을 frontend로 proxy한다.

### langfuse 스택

Langfuse self-host 실험용 optional 스택만 올린다. 앱 DB/Redis와 Langfuse DB/Redis는 공유하지 않는다. `infra/langfuse/.env.example`은 앱 env가 아니라 Langfuse self-host 전용 템플릿이다.

```bash
cd infra/langfuse
cp .env.example .env
docker compose up -d
```

wrapper 명령:

```bash
./scripts/docker_stack.sh langfuse up
./scripts/docker_stack.sh langfuse ps
./scripts/docker_stack.sh langfuse logs
./scripts/docker_stack.sh langfuse down
```

Makefile:

```bash
make langfuse-up
make langfuse-ps
make langfuse-logs
make langfuse-down
```

`langfuse up`은 `ai-health-shared` Docker network가 없으면 자동 생성한다.

## 3. prod 스택

운영/EC2 표준 스택은 `infra/docker/docker-compose.prod.yml` 기준이다. 로컬 시연용 기본 명령에 포함하지 않는다. prod compose는 app, ai-worker, frontend service image를 registry에서 pull하는 구조이며, EC2에서 직접 build하지 않는다.

prod compose가 pull하는 이미지 tag는 `.prod.env`의 version 값과 Makefile image target의 tag 규칙을 함께 맞춘다.

| 서비스 | 이미지 tag 규칙 | Dockerfile |
| --- | --- | --- |
| `fastapi` | `${DOCKER_USER}/${DOCKER_REPOSITORY}:app-${APP_VERSION}` | `app/Dockerfile` |
| `ai-worker` | `${DOCKER_USER}/${DOCKER_REPOSITORY}:ai-${AI_WORKER_VERSION}` | `ai_runtime/Dockerfile` |
| `frontend` | `${DOCKER_USER}/${DOCKER_REPOSITORY}:frontend-${FRONTEND_VERSION}` | `frontend/Dockerfile` |

운영 배포마다 같은 tag를 재사용하지 말고 `v1.0.1` 또는 `20260616-1` 같은 새 tag를 사용한다. 로컬 release 브랜치에서 세 image를 같은 배포 tag로 build/push한 뒤, EC2 `.prod.env`의 `APP_VERSION`, `AI_WORKER_VERSION`, `FRONTEND_VERSION`을 같은 값으로 갱신한다. Makefile image target은 secret 노출을 피하기 위해 `.prod.env` 전체를 자동 source하지 않으므로, build/push 시 version 값만 shell에서 export한다.

배포 전 로컬 build 검증:

```bash
make image-tags
make image-build-check
make image-push
```

`make image-build-check`는 배포 검증용 alias로 `linux/amd64` buildx target을 실행한다. Apple Silicon native build는 ai-worker의 `paddlepaddle` Linux arm64 wheel 부재로 실패할 수 있으므로 배포 이미지는 amd64로 확인한다.

```bash
make image-buildx-amd64-check
```

현재 로컬 아키텍처 build를 별도로 확인할 때만 아래 target을 사용한다.

```bash
make image-build-native-check
```

`frontend/Dockerfile`의 `VITE_*` 값은 build-time에 정적 bundle로 들어간다. 브라우저 public config만 넣고 server secret은 절대 build arg로 전달하지 않는다.

```bash
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml pull
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml up -d
```

Makefile 표준 흐름:

```bash
cp envs/example.prod.env .prod.env
# 최초 HTTPS 인증서 발급 전에는 .prod.env에서 NGINX_CONF=../nginx/prod_http.conf 사용
make prod-pull
make prod-up
make prod-migrate
make prod-health
```

`make prod-up`은 external network `ai-health-shared`가 없으면 자동 생성한 뒤 prod compose를 실행한다. 직접 compose 명령을 사용하는 운영자는 먼저 `docker network inspect ai-health-shared >/dev/null 2>&1 || docker network create ai-health-shared`로 네트워크를 보장한다.

`make prod-health`는 `.prod.env`의 `NGINX_HTTP_PORT`를 읽고, 값이 없으면 운영 기본 `80`으로 `http://localhost:80/api/v1/system/health`를 확인한다. HTTPS/certbot 전환 전에는 HTTP health가 먼저 성공해야 한다.

운영 배포 target은 seed와 RAG ingest를 자동 실행하지 않는다. 초기 운영 DB에 챌린지 seed가 반드시 필요하고 운영자가 명시적으로 승인한 경우에만 `make danger-prod-seed-challenges`를 별도로 실행한다. FAQ 테이블은 migration으로 생성되지만 FAQ row는 별도 seed이므로, FAQ 목록이 비어 있고 운영자가 승인한 경우에만 `make danger-prod-seed-faqs`를 실행한다. 하위 호환용 `make danger-prod-seed`는 챌린지 seed alias이며, `make prod-release-db`는 migration만 수행한다.

운영 실행 전 확인할 것:

- app, ai-worker, frontend image registry와 tag가 준비되어 있는지
- prod env 값이 placeholder가 아닌지
- `healthladder.duckdns.org`가 EC2 public IP로 resolve되는지
- HTTPS/certbot 설정이 `healthladder.duckdns.org`와 맞는지
- DB backup/restore 계획이 있는지
- QA/smoke script는 운영 이미지 런타임 필수 파일이 아니며, `scripts/qa`는 로컬 QA 또는 별도 수동 smoke 절차에서만 사용한다.

DuckDNS token은 secret이므로 Git tracked 파일, README, issue, shell history, 배포 로그에 남기지 않는다. DNS/HTTP/HTTPS 확인은 실제 배포 전후에 아래처럼 수행한다.

```bash
nslookup healthladder.duckdns.org
curl -I http://healthladder.duckdns.org
curl -I https://healthladder.duckdns.org
curl -fsS https://healthladder.duckdns.org/api/v1/system/health
```

## 4. 주의사항

### docker compose config 출력 주의

`docker compose config` 전체 출력에는 `.env`와 compose environment 값이 평문으로 펼쳐질 수 있다. 화면 공유, 이슈, 메신저에 전체 출력을 공유하지 않는다.

안전한 확인 명령:

```bash
docker compose config --quiet
docker compose config --services
docker compose ps
docker compose logs --tail=100 fastapi
```

### up/build와 push는 다르다

`docker compose up` 또는 `docker compose build`는 원격 registry에 이미지를 올리지 않는다. 원격 저장소에 이미지를 올리는 작업은 `docker push`를 실행할 때만 발생한다.

다만 `docker compose up` 또는 `docker compose build` 과정에서 로컬에 없는 base image나 service image는 Docker Hub 등 registry에서 pull될 수 있다.

### Docker build context / .dockerignore

현재 app, ai-worker, frontend 이미지 build context는 모두 repository root다. 따라서 Docker가 실제로 적용하는 ignore 파일은 root `.dockerignore` 하나다. 하위 `app/.dockerignore`, `ai_runtime/.dockerignore`는 이 빌드 경로에서는 적용되지 않아 혼동을 줄 수 있으므로 사용하지 않는다.

root `.dockerignore`는 `.env`, private env 파일, `.venv`, `.git`, cache/output, `node_modules`, 로컬 데이터와 대용량 실험 산출물을 제외한다. 단, `app/Dockerfile`이 copy하는 challenge seed master data와 runtime CatBoost artifact는 명시적으로 예외 처리한다.

### volume 삭제 주의

시연 직전에는 DB volume을 삭제하지 않는다.

금지:

```bash
docker compose down -v
```

DB를 초기화해야 할 때는 백업/seed 재현 가능성을 먼저 확인한다.

### Langfuse 분리

Langfuse는 앱의 PostgreSQL/Redis를 공유하지 않는다. Langfuse 전용 compose는 자체 `postgres`, `redis`, `clickhouse`, `minio`를 사용한다.

Langfuse는 RAG 엔진이나 vector DB가 아니라 RAG 검색/LLM 호출의 trace, prompt, evaluation metadata를 관측하기 위한 도구다. keyword RAG PoC는 `docs/rag_sources`와 `ai_runtime/llm/rag`의 로컬 markdown source를 사용하고, Langfuse에는 어떤 query와 source가 선택됐는지만 metadata로 남긴다.

Cloud Langfuse를 사용할 때는 같은 Cloud 프로젝트의 key와 Cloud base URL을 함께 사용한다.

```env
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=<cloud-public-key>
LANGFUSE_SECRET_KEY=<cloud-secret-key>
LANGFUSE_BASE_URL=https://jp.cloud.langfuse.com
```

Docker self-host Langfuse를 사용할 때는 self-host 프로젝트에서 발급한 key와 FastAPI 실행 위치에서 접근 가능한 base URL을 함께 사용한다.

```env
# FastAPI를 호스트 터미널에서 실행하는 경우
LANGFUSE_BASE_URL=http://localhost:3000

# FastAPI 컨테이너가 Langfuse와 같은 Docker network에 붙어 있는 경우
LANGFUSE_BASE_URL=http://langfuse-web:3000

# FastAPI 컨테이너가 Langfuse network에 붙어 있지 않고 host port로 접근하는 경우
LANGFUSE_BASE_URL=http://host.docker.internal:3000
```

Cloud key와 self-host key는 서로 호환되지 않는다. Cloud와 self-host를 전환할 때는 key만 바꾸지 말고 `LANGFUSE_BASE_URL`도 함께 바꾼다.

현재 root `docker-compose.yml`의 `fastapi`는 기본적으로 앱 전용 `ws` network에 붙는다. `http://langfuse-web:3000` service name을 쓰려면 FastAPI도 Langfuse의 `ai-health-shared` network에 연결해야 한다. 네트워크를 추가하지 않을 때는 Docker Desktop 기준 `http://host.docker.internal:3000`을 사용한다.

example env 파일에는 실제 key를 넣지 않는다. 설정을 확인할 때도 `docker compose config` 전체 출력은 secret을 평문으로 펼칠 수 있으므로 공유하지 않는다.

## 5. 시연 전 추천 순서

```bash
make app-up
curl http://localhost:8000/api/v1/system/health
uv run python scripts/verify_demo_ready.py
uv run python scripts/verify_precision_analysis_api.py --warmup-ml
```

frontend까지 Docker로 확인하려면:

```bash
make app-down
make dev-up
```

Langfuse 관측 실험이 필요할 때만:

```bash
make langfuse-up
```
