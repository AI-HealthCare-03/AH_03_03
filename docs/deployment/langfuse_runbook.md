# Langfuse Runbook

이 문서는 Langfuse self-host를 Health Ladder 앱과 분리해서 안전하게 실행하고 점검하는 절차를 정리한다.

## 핵심 원칙

- Langfuse는 관측 도구이며 앱 실행 필수 구성요소가 아니다.
- Langfuse 장애가 FastAPI, AI Worker, RAG, 식단 분석 장애가 되면 안 된다.
- 앱의 Langfuse trace 기록은 best-effort로 동작해야 하며, 실패해도 warning/no-op으로 끝나야 한다.
- Langfuse와 Health Ladder 메인 앱은 서로 다른 Docker Compose 파일을 사용한다.
- `docker compose down -v`는 Langfuse 데이터 볼륨을 삭제하므로 운영/공유 환경에서 실행하지 않는다.

## Compose 구분

Health Ladder 메인 compose:

```text
infra/docker/docker-compose.dev.yml
infra/docker/docker-compose.prod.yml
docker-compose.yml
```

Langfuse compose:

```text
infra/langfuse/docker-compose.yml
```

프로젝트 루트에서 `docker compose ...`를 실행하면 루트 `docker-compose.yml` 기준으로 동작한다. 따라서 루트에서 아래 명령을 실행하면 Langfuse 서비스가 없어 실패한다.

```bash
docker compose restart langfuse-web
```

Langfuse 서비스 조작은 반드시 `infra/langfuse` compose를 대상으로 실행한다.

## 실행 절차

처음 1회:

```bash
cd infra/langfuse
cp .env.example .env
```

`infra/langfuse/.env`의 `CHANGEME_*` 값을 교체한다. 이 파일은 커밋하지 않는다.

Makefile 사용:

```bash
make langfuse-up
make langfuse-ps
make langfuse-health
make langfuse-logs
```

직접 실행:

```bash
cd infra/langfuse
docker compose up -d
docker compose ps
curl -fsS http://127.0.0.1:3000 > /dev/null
docker compose logs --tail=100 langfuse-web langfuse-worker
```

## 재시작 절차

Langfuse Web이 일시적으로 응답하지 않거나 readiness 문제가 의심될 때:

```bash
make langfuse-restart
make langfuse-health
make langfuse-logs
```

직접 실행:

```bash
cd infra/langfuse
docker compose restart langfuse-web langfuse-worker
curl -v http://127.0.0.1:3000
```

`HTTP/1.1 200 OK`가 보이면 Langfuse Web은 응답 가능한 상태다.

## 종료 절차

일반 종료:

```bash
make langfuse-down
```

직접 실행:

```bash
cd infra/langfuse
docker compose down
```

금지:

```bash
docker compose down -v
```

`down -v`는 Langfuse Postgres, Redis, ClickHouse, MinIO 볼륨을 삭제할 수 있다. 운영/공유 환경에서는 사용하지 않는다.

## 정상 확인 명령

서비스 목록:

```bash
make langfuse-ps
```

HTTP health:

```bash
make langfuse-health
curl -v http://127.0.0.1:3000
```

포트 점검:

```bash
lsof -nP -iTCP:3000 -sTCP:LISTEN
```

최근 로그:

```bash
make langfuse-logs
cd infra/langfuse && docker compose logs --tail=100 langfuse-web langfuse-worker
```

## 앱 연동 환경변수

실제 값은 문서나 PR에 넣지 않는다.

로컬 `uv` 실행:

```env
LANGFUSE_ENABLED=true
LANGFUSE_BASE_URL=http://localhost:3000
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=<langfuse-public-key>
LANGFUSE_SECRET_KEY=<langfuse-secret-key>
```

앱 Docker 실행, host port 접근:

```env
LANGFUSE_ENABLED=true
LANGFUSE_BASE_URL=http://host.docker.internal:3000
LANGFUSE_HOST=http://host.docker.internal:3000
LANGFUSE_PUBLIC_KEY=<langfuse-public-key>
LANGFUSE_SECRET_KEY=<langfuse-secret-key>
```

같은 Docker network에서 service name 접근:

```env
LANGFUSE_ENABLED=true
LANGFUSE_BASE_URL=http://langfuse-web:3000
LANGFUSE_HOST=http://langfuse-web:3000
LANGFUSE_PUBLIC_KEY=<langfuse-public-key>
LANGFUSE_SECRET_KEY=<langfuse-secret-key>
```

`infra/docker/docker-compose.dev.yml`과 `infra/langfuse/docker-compose.yml`은 `ai-health-shared` network를 사용한다. 이 네트워크는 HTTP 접근용이며 앱 Postgres/Redis와 Langfuse Postgres/Redis를 공유하지 않는다.

## 최근 장애 원인 분석

관측된 현상:

- 루트에서 `docker compose restart langfuse-web` 실행 시 `no such service: langfuse-web`
- 루트에서 `docker compose up -d` 실행 시 메인 Redis `6379` 포트 충돌
- `infra/langfuse`에서 `docker compose restart langfuse-web langfuse-worker` 후 `curl -v http://127.0.0.1:3000`은 `HTTP/1.1 200 OK`

판단:

- Langfuse 완전 장애라기보다 compose 실행 위치 혼동과 Web readiness/restart 문제가 원인이다.
- 루트 compose는 Health Ladder 메인 앱용이고, Langfuse compose는 `infra/langfuse/docker-compose.yml`이다.
- `docker compose up -d`는 이미 running 중인 Web 컨테이너를 강제 재시작하지 않는다.
- Langfuse Web은 Redis/Postgres readiness 전후로 일시적으로 empty response를 줄 수 있다.
- 과거 Redis 로그와 현재 로그를 구분해야 한다. 로그는 `--tail=100` 또는 시간 기준으로 확인한다.

## 앱 안전성 확인

현재 앱의 Langfuse 연동은 다음 구조다.

- `LANGFUSE_ENABLED=false`이면 trace 없이 정상 동작한다.
- Langfuse 설정이 없거나 client 생성에 실패하면 no-op으로 처리한다.
- `record_langfuse_event()`는 예외를 서비스 코드로 전파하지 않고 `False`를 반환한다.
- OpenAI generation trace wrapper도 Langfuse observation 생성 실패 시 일반 OpenAI 호출로 fallback한다.

따라서 Langfuse 장애는 앱 장애가 되면 안 된다.

## 흔한 문제와 대응

### `no such service: langfuse-web`

원인: 루트 compose에서 Langfuse 서비스를 찾고 있다.

대응:

```bash
cd infra/langfuse
docker compose ps
```

또는:

```bash
make langfuse-ps
```

### Redis `6379` 포트 충돌

원인: 루트 메인 compose를 실행했거나, 앱 Redis와 별도 로컬 Redis가 충돌한다.

대응:

- Langfuse만 점검하려면 `make langfuse-*` 명령을 사용한다.
- 루트에서 `docker compose up -d`를 실행하지 않는다.

### `curl` empty response

가능 원인:

- Web이 재시작 직후 readiness 전 상태
- Redis/Postgres/ClickHouse 의존 서비스 연결 지연

대응:

```bash
make langfuse-restart
sleep 5
make langfuse-health
make langfuse-logs
```

### trace가 안 보임

확인:

- `LANGFUSE_ENABLED=true`
- `LANGFUSE_BASE_URL`이 실행 방식에 맞는 주소인지
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`가 같은 Langfuse 프로젝트에서 발급된 값인지
- 실제 trace 대상 기능이 켜져 있는지. 예: `CHATBOT_USE_REAL_LLM`, `RAG_ENABLED`, GPT Vision 관련 flag

## 배포 전 점검

```bash
make langfuse-ps
make langfuse-health
make langfuse-logs
```

이후 앱에서 trace 대상 기능을 1회 실행하고 Langfuse UI에서 trace가 생성되는지 확인한다.
