# Langfuse Self-Host

이 문서는 `infra/langfuse/`의 Langfuse Docker Compose 설정을 설명한다. Langfuse는 LLM/GPT Vision/RAG/챗봇 관측 도구다.

Langfuse는 앱 필수 실행 요소가 아니다. `LANGFUSE_ENABLED=false`이면 FastAPI와 AI Worker는 trace 없이 정상 동작한다. 또한 Langfuse는 RAG/LLM 실행 엔진이 아니라 trace, prompt, evaluation metadata를 관리하는 관측 도구다. 현재 keyword RAG도 루트 앱 환경변수 `RAG_ENABLED=false`가 기본값이라 비활성화되어 있다.

## 왜 별도 compose로 분리하나요?

Langfuse는 자체 Web, Worker, Postgres, Redis, ClickHouse, MinIO가 필요하다. 우리 서비스의 Postgres/Redis와 공유하면 데이터 수명주기, 장애 영향, 포트, 백업 정책이 섞인다.

따라서 Langfuse는 `infra/langfuse/docker-compose.yml`에서 별도 compose project로 실행한다.

## 구성 요소

- `langfuse-web`: Langfuse UI/API, `http://localhost:3000`
- `langfuse-worker`: trace ingestion/background worker
- `postgres`: Langfuse 전용 metadata DB
- `redis`: Langfuse 전용 queue/cache
- `clickhouse`: trace/observation analytics storage
- `minio`: Langfuse 전용 blob storage

우리 서비스와 공유하지 않는 항목:

- app Postgres
- app Redis
- app Docker network
- app migration

## 포트

- Langfuse Web: `localhost:3000`
- Postgres/Redis/ClickHouse/MinIO: host에 노출하지 않음

우리 서비스가 사용하는 `80`, `8000`, `5432`, `6379`, `5173`과 충돌하지 않게 구성했다.

Langfuse media upload 기능에서 브라우저가 MinIO에 직접 접근해야 하는 운영 구성이 필요하면 MinIO API를 별도 포트로 노출해야 할 수 있다. 로컬 LLM/RAG trace 관측 목적에서는 우선 내부 네트워크 전용으로 둔다.

## 실행

```bash
cd infra/langfuse
cp .env.example .env
```

`.env`의 `CHANGEME` 값을 교체한다.

특히 아래 값은 반드시 바꾼다.

- `NEXTAUTH_SECRET`
- `SALT`
- `ENCRYPTION_KEY`
- `POSTGRES_PASSWORD`
- `REDIS_AUTH`
- `CLICKHOUSE_PASSWORD`
- `MINIO_ROOT_USER`
- `MINIO_ROOT_PASSWORD`
- `LANGFUSE_INIT_USER_PASSWORD`

권장 생성 예:

```bash
openssl rand -base64 32
openssl rand -hex 32
```

설정 확인:

```bash
docker compose config --services
```

주의: `docker compose config` 전체 출력에는 `.env` secret이 펼쳐질 수 있으므로 화면 공유나 문서 공유에 사용하지 않는다.

실행:

```bash
docker compose up -d
```

접속:

```bash
open http://localhost:3000
```

## 종료

```bash
cd infra/langfuse
docker compose down
```

볼륨까지 초기화:

주의: 아래 명령은 Langfuse Postgres, Redis, ClickHouse, MinIO 볼륨을 삭제할 수 있다. 운영/공유 환경에서는 실행하지 않는다. 로컬 실험 데이터를 완전히 버릴 때만 사용한다.

```bash
docker compose down -v
```

## 우리 앱 연동

Langfuse UI에서 project API key를 발급한 뒤 루트 앱 환경변수에 넣는다.

예:

```env
LANGFUSE_ENABLED=true
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_BASE_URL=http://localhost:3000
LANGFUSE_PUBLIC_KEY=<langfuse-public-key>
LANGFUSE_SECRET_KEY=<langfuse-secret-key>
```

우리 서비스도 Docker compose로 실행하고 Langfuse와 같은 서버에서 통신하려면 external network를 공유한다.

```bash
docker network create ai-health-shared
```

`infra/docker/docker-compose.dev.yml`과 `infra/langfuse/docker-compose.yml`은 모두 `ai-health-shared`를 사용하도록 구성되어 있다.
이 네트워크는 `fastapi`/`ai-worker`가 Langfuse Web에 HTTP로 접근하기 위한 통로일 뿐이며, Postgres/Redis를 공유하지 않는다.

Docker network 내부에서 접근할 때:

```env
LANGFUSE_HOST=http://langfuse-web:3000
LANGFUSE_BASE_URL=http://langfuse-web:3000
```

호스트에서 직접 접근할 때:

```env
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_BASE_URL=http://localhost:3000
```

현재 앱에서 관측 가능한 대상:

- keyword RAG retrieval trace. 단, `RAG_ENABLED=true`일 때만 생성된다.
- GPT Vision/LLM 관련 trace metadata. 단, 해당 provider flag와 `LANGFUSE_ENABLED=true`가 함께 켜져야 한다.
- key가 없거나 disabled 상태면 no-op으로 동작해야 하며 앱 응답을 깨지 않는다.

## 참고

- 공식 Langfuse self-host Docker Compose 구성을 기준으로 Web, Worker, Postgres, Redis, ClickHouse, MinIO topology를 분리 적용했다.
- 운영 배포에서는 secret 관리, backup, TLS, 외부 접근 제어, retention 정책을 별도로 설계해야 한다.
