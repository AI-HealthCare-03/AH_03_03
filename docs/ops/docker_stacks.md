# Docker Stacks

이 프로젝트는 용도별 Docker Compose 구성을 분리한다. 시연/개발/관측 도구를 한 파일에 모두 넣지 않고, 필요한 스택만 선택해서 실행한다.

## 1. 스택 요약

| 스택 | Compose 파일 | 용도 | 주요 서비스 |
| --- | --- | --- | --- |
| `app` | `docker-compose.yml` | 로컬 FastAPI/ML 시연용 최소 스택 | `postgres`, `redis`, `fastapi` |
| `dev` | `infra/docker/docker-compose.dev.yml` | frontend 포함 개발 전체 스택 | `postgres`, `redis`, `fastapi`, `ai-worker`, `frontend`, `nginx` |
| `prod` | `infra/docker/docker-compose.prod.yml` | 운영 이미지 기반 스택 | `postgres`, `redis`, `fastapi`, `ai-worker`, `frontend`, `nginx`, `certbot` |
| `langfuse` | `infra/langfuse/docker-compose.yml` | Langfuse 관측 실험 전용 스택 | `langfuse-web`, `langfuse-worker`, `postgres`, `redis`, `clickhouse`, `minio` |

`scripts/docker_stack.sh`와 `Makefile`은 자주 쓰는 `app`, `dev`, `langfuse` 스택 실행을 감싼다. `prod`는 이미지 태그와 운영 환경변수 확인이 필요하므로 명시적으로 compose 파일을 지정해서 실행한다.

## 현재 Redis 범위

현재 앱 스택의 Redis는 아래 용도로만 사용한다.

- Docker Compose Redis 컨테이너 실행
- FastAPI의 `REDIS_HOST`, `REDIS_PORT` 기반 연결
- `/api/v1/system/health` Redis 연결 확인
- Compose `redis` service healthcheck

아래 항목은 아직 구현하지 않으며 P2 운영 확장 범위로 둔다.

- Redis Stream `XADD` / `XREADGROUP`
- `async_jobs` 테이블
- AI Worker consumer
- retry / dead-letter queue
- 비동기 OCR/CV/ML/LLM 처리

시연 설명 문구:

> 현재 MVP는 FastAPI 동기 처리로 OCR/ML/CV/LLM 준비 흐름을 검증합니다. 운영 확장 단계에서는 Redis Stream 기반 비동기 worker로 전환해 장시간 작업, 재시도, dead-letter queue, 작업 상태 추적을 붙입니다.

## 2. 빠른 실행

### app 스택

로컬 시연에서 FastAPI, PostgreSQL, Redis만 띄운다. `ai-worker`, `frontend`, `nginx`는 기본으로 올리지 않는다.

```bash
./scripts/docker_stack.sh app up
./scripts/docker_stack.sh app build
./scripts/docker_stack.sh app rebuild
./scripts/docker_stack.sh app clean-image
./scripts/docker_stack.sh app ps
./scripts/docker_stack.sh app logs
./scripts/docker_stack.sh app down
```

Makefile:

```bash
make app-up
make app-build
make app-rebuild
make app-clean-image
make app-ps
make app-logs
make app-down
```

직접 compose 명령:

```bash
docker compose -f docker-compose.yml up -d postgres redis fastapi
```

`app up`은 구버전 FastAPI 이미지가 남아 최신 `ai_worker` 파일이 반영되지 않는 상황을 피하기 위해 `--build`를 함께 실행한다. `ai_worker` 코드, CatBoost artifact, FastAPI Dockerfile, `pyproject.toml`, `uv.lock`이 바뀐 뒤에는 아래 명령 중 하나를 실행한다.

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

frontend, Nginx, FastAPI, AI Worker 자리, PostgreSQL, Redis를 모두 올린다.

```bash
./scripts/docker_stack.sh dev up
./scripts/docker_stack.sh dev ps
./scripts/docker_stack.sh dev down
```

Makefile:

```bash
make dev-up
make dev-ps
make dev-down
```

직접 compose 명령:

```bash
docker compose -f infra/docker/docker-compose.dev.yml up -d
```

### langfuse 스택

Langfuse self-host 실험용 스택만 올린다. 앱 DB/Redis와 Langfuse DB/Redis는 공유하지 않는다.

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

운영 스택은 image 기반이다. 로컬 시연용 기본 명령에 포함하지 않는다.

```bash
docker compose -f infra/docker/docker-compose.prod.yml up -d
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

앱과 Langfuse가 통신해야 할 경우에는 `ai-health-shared` network를 통해 연결한다.

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
