# Docker Stacks

이 프로젝트는 용도별 Docker Compose 구성을 분리한다. 시연/개발/관측 도구를 한 파일에 모두 넣지 않고, 필요한 스택만 선택해서 실행한다.

## 1. 스택 요약

| 스택 | Compose 파일 | 용도 | 주요 서비스 |
| --- | --- | --- | --- |
| `app` | `docker-compose.yml` | legacy/minimal backend/AI 검증용 스택 | `postgres`, `redis`, `fastapi` |
| `dev` | `infra/docker/docker-compose.dev.yml` | 표준 로컬 개발/시연 전체 스택 | `postgres`, `redis`, `fastapi`, `ai-worker`, `frontend`, `nginx` |
| `prod` | `infra/docker/docker-compose.prod.yml` | 표준 EC2/운영 이미지 기반 스택 | `postgres`, `redis`, `fastapi`, `ai-worker`, `frontend`, `nginx`, `certbot` |
| `langfuse` | `infra/langfuse/docker-compose.yml` | Langfuse self-host optional 관측 스택 | `langfuse-web`, `langfuse-worker`, `postgres`, `redis`, `clickhouse`, `minio` |

`scripts/docker_stack.sh`와 `Makefile`은 자주 쓰는 `app`, `dev`, `langfuse` 스택 실행을 감싼다. `prod`는 이미지 태그와 운영 환경변수 확인이 필요하므로 명시적으로 compose 파일을 지정해서 실행한다.

현재 표준 로컬 실행은 `infra/docker/docker-compose.dev.yml`이다. 루트 `docker-compose.yml`은 legacy/minimal backend/AI 검증용으로만 유지하며, frontend, Firebase Web Push build args, storage, scheduler, 최신 dev full stack 검증에는 사용하지 않는다.

## 현재 Redis / async job 범위

현재 앱 스택의 Redis는 health check뿐 아니라 Redis Stream 기반 job 처리에도 사용한다.

- Docker Compose Redis 컨테이너 실행
- FastAPI의 `REDIS_HOST`, `REDIS_PORT` 기반 연결
- `/api/v1/system/health` Redis 연결 확인
- Compose `redis` service healthcheck
- AI stream: `analysis.run`, `exam_ocr.run`, `diet.analyze_image`, `medication_ocr.run`
- Service stream: `email.verification.send`, `password_reset.email.send`, `family.invite.email.send`, `fcm.push.send`, `family.notification.create`
- retry/backoff, DLQ, pending recovery 기반 `ai-worker` consumer

P2 운영 확장 범위로 남은 항목은 대시보드/알림 수준의 운영 관측, queue 지표 노출, alerting, worker 수평 확장 정책이다.

시연 설명 문구:

> 현재 MVP는 긴 OCR/식단/분석/외부 발송 작업을 Redis Stream async job으로 넘기고, `/api/v1/jobs/{job_id}` polling으로 상태를 확인합니다. 긴 작업의 기존 동기 분석 API는 410 Gone으로 막고 async 전용 경로를 사용합니다.

## 2. 빠른 실행

### app 스택

루트 `docker-compose.yml`을 사용하는 legacy/minimal backend/AI 검증용 스택이다. FastAPI, PostgreSQL, Redis만 빠르게 띄우며 `ai-worker`, `frontend`, `nginx`는 기본으로 올리지 않는다.

이 스택은 FastAPI 중심 검증용이지만 frontend, Firebase Web Push build args, storage, scheduler까지 포함한 최신 dev full stack 검증에는 사용하지 않는다. 실제 프론트 포함 비동기 UX 확인은 `dev` 스택에서 `ai-worker`, `frontend`, `nginx`까지 함께 올려 검증한다.

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

- `ozcodingschool/ai-health:app-v1.0.0`
- `test-ai-health:latest`

이미지가 없으면 건너뛴다. 실행 중인 컨테이너가 이미지를 사용 중이면 Docker가 삭제를 거부할 수 있으므로 먼저 `make app-down`으로 컨테이너를 내린 뒤 다시 실행한다.

### dev 스택

frontend, Nginx, FastAPI, AI Worker, PostgreSQL, Redis를 모두 올린다.

`ai-worker` service는 `ai_runtime/main.py`를 통해 Redis Stream consumer와 scheduler loop를 실행한다. 처리 job은 `DEMO_ECHO`뿐 아니라 `analysis.run`, `exam_ocr.run`, `diet.analyze_image`, `medication_ocr.run`, 이메일/비밀번호/가족초대/FCM/가족알림 service job을 포함한다.

로컬 개발에서는 `.env`를 직접 계속 바꾸지 않도록 provider 성격에 따라 아래 프로필을 사용한다.

```bash
make dev-up-mock
make dev-up-real
```

직접 compose 명령을 사용할 때는 Docker Compose variable interpolation이 `env_file`이 아니라 `--env-file` 또는 루트 `.env`를 기준으로 동작한다는 점에 주의한다.

```bash
./scripts/docker_stack.sh dev up
./scripts/docker_stack.sh dev ps
./scripts/docker_stack.sh dev down
```

Makefile:

```bash
make dev-up
make dev-up-mock
make dev-up-real
make dev-ps
make dev-down
```

직접 compose 명령:

```bash
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml up -d --build --force-recreate
docker compose --env-file .env.local.mock -f infra/docker/docker-compose.dev.yml up -d --build --force-recreate
docker compose --env-file .env.local.real -f infra/docker/docker-compose.dev.yml up -d --build --force-recreate
```

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

운영/EC2 표준 스택은 `infra/docker/docker-compose.prod.yml` 기준이다. 로컬 시연용 기본 명령에 포함하지 않는다.

```bash
docker compose --env-file .env -f infra/docker/docker-compose.prod.yml up -d --build --force-recreate
```

운영 실행 전 확인할 것:

- image registry와 tag가 준비되어 있는지
- prod env 값이 placeholder가 아닌지
- HTTPS/certbot 설정이 실제 도메인과 맞는지
- DB backup/restore 계획이 있는지

## 4. 주의사항

### docker compose config 출력 주의

`docker compose config` 전체 출력에는 `.env`와 compose environment 값이 평문으로 펼쳐질 수 있다. 화면 공유, 이슈, 메신저에 전체 출력을 공유하지 않는다.

안전한 확인 명령:

```bash
docker compose config --services
docker compose ps
docker compose logs --tail=100 fastapi
```

### up/build와 push는 다르다

`docker compose up` 또는 `docker compose build`는 원격 registry에 이미지를 올리지 않는다. 원격 저장소에 이미지를 올리는 작업은 `docker push`를 실행할 때만 발생한다.

다만 `docker compose up` 또는 `docker compose build` 과정에서 로컬에 없는 base image나 service image는 Docker Hub 등 registry에서 pull될 수 있다.

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
