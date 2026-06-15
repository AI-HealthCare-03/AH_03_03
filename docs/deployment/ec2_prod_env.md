# EC2 Production Deployment Checklist

이 문서는 EC2에서 `infra/docker/docker-compose.prod.yml` 기준으로 운영 스택을 띄울 때 확인할 항목을 정리한 것입니다. 실제 secret, API key, private key 원문은 이 문서나 Git tracked 파일에 작성하지 않습니다.

## Compose 기준

- 시연/운영 스택 기준 파일: `infra/docker/docker-compose.prod.yml`
- 로컬 개발 스택 기준 파일: `infra/docker/docker-compose.dev.yml`
- 이번 부트캠프 데모 운영 범위는 AWS `EC2 + S3`만 사용합니다.
- EC2 단일 서버에서 Docker Compose로 `nginx`, `frontend`, `fastapi`, `ai-worker`, `postgres`, `redis` 컨테이너를 실행합니다.
- PostgreSQL은 EC2 내부 `postgres` 컨테이너와 `postgres_data` Docker volume으로 운영합니다.
- Redis Stream은 EC2 내부 `redis` 컨테이너와 `redis_data` Docker volume으로 운영합니다.
- 업로드 파일은 S3 private bucket에 저장합니다.
- RDS와 ElastiCache는 이번 데모 운영 범위에서 제외합니다. 장기 운영 전환 시 별도 운영 설계로 검토할 수 있습니다.
- prod compose는 외부에 `nginx:80`, `nginx:443`만 노출합니다.
- Postgres와 Redis는 Docker network 내부 `expose`만 사용하며 EC2 보안 그룹에서 직접 열지 않습니다.
- FastAPI와 ai-worker는 각각 DB connection pool을 만들기 때문에 `DB_POOL_MAX_SIZE`는 작게 시작합니다.
- HTTPS 운영 기본 Nginx 설정은 `infra/nginx/prod_https.conf`입니다.

## EC2에서 준비할 환경 파일

운영 서버에서는 예시 파일을 복사해서 별도 파일로 관리합니다.

```bash
cp envs/example.prod.env prod.env
```

반드시 운영자가 교체해야 하는 주요 항목:

- `SECRET_KEY`
- `COOKIE_DOMAIN`
- `CORS_ALLOW_ORIGINS`
- `FRONTEND_BASE_URL`
- `DB_PASSWORD`
- `SMTP_HOST`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM_EMAIL`
- `OPENAI_API_KEY` 또는 사용하는 LLM provider secret
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`를 사용하는 경우
- `S3_BUCKET_NAME`
- `NGINX_CONF`는 기본 `../nginx/prod_https.conf`, 최초 인증서 발급 전에는 `../nginx/prod_http.conf`

작성하지 말아야 하는 값:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- SMTP password 원문을 문서/README/예시 파일에 직접 작성한 값
- OpenAI, Langfuse secret 원문

## 배포 이미지 빌드 기준

`infra/docker/docker-compose.prod.yml`은 app, ai-worker, frontend 이미지를 EC2에서 build하지 않고 registry에서 pull한다. 따라서 배포 전 로컬 또는 CI에서 아래 3개 이미지를 같은 tag 규칙으로 build/push해야 한다.

| 이미지 | prod compose tag | Dockerfile | build context |
| --- | --- | --- | --- |
| FastAPI app | `${DOCKER_USER}/${DOCKER_REPOSITORY}:app-${APP_VERSION}` | `app/Dockerfile` | repo root |
| AI Worker | `${DOCKER_USER}/${DOCKER_REPOSITORY}:ai-${AI_WORKER_VERSION}` | `ai_runtime/Dockerfile` | repo root |
| Frontend | `${DOCKER_USER}/${DOCKER_REPOSITORY}:frontend-${FRONTEND_VERSION}` | `frontend/Dockerfile` | repo root |

로컬 build 검증:

```bash
make image-tags
make image-build-check
```

`make image-build-check`는 배포 검증용 alias이며 기본적으로 `make image-buildx-amd64-check`를 실행한다. EC2 Ubuntu t3 계열 배포 대상은 `linux/amd64`이므로 배포 전 검증은 amd64 buildx 기준으로 맞춘다.

Apple Silicon에서 Docker Desktop 기본 native build를 실행하면 `linux/arm64/aarch64`로 빌드될 수 있다. `paddlepaddle`은 Linux aarch64 wheel이 제공되지 않는 버전이 있어 ai-worker 이미지 native build가 실패할 수 있다. 이 실패는 EC2 amd64 배포 이미지 실패와 다르므로, 배포 검증에는 아래 buildx target을 사용한다.

```bash
make image-buildx-amd64-check
```

로컬 아키텍처 자체를 확인하고 싶을 때만 native build target을 사용한다.

```bash
make image-build-native-check
```

필요하면 platform을 명시적으로 바꿀 수 있다. EC2 amd64 배포에서는 기본값을 유지한다.

```bash
DOCKER_PLATFORM=linux/amd64 make image-buildx-amd64-check
```

Docker Hub push 전에는 로컬 이미지 architecture가 `amd64`인지 확인한다.

```bash
docker image inspect ${DOCKER_USER}/${DOCKER_REPOSITORY}:app-${APP_VERSION} \
  --format '{{.Os}}/{{.Architecture}}'
docker image inspect ${DOCKER_USER}/${DOCKER_REPOSITORY}:ai-${AI_WORKER_VERSION} \
  --format '{{.Os}}/{{.Architecture}}'
docker image inspect ${DOCKER_USER}/${DOCKER_REPOSITORY}:frontend-${FRONTEND_VERSION} \
  --format '{{.Os}}/{{.Architecture}}'
```

각 출력이 `linux/amd64`인지 확인한 뒤 push한다.

Docker Hub push는 build 검증과 tag 확인 후 별도로 실행한다. push 전에는 Docker Hub 계정/레포/tag가 prod compose와 일치하는지 확인한다.

```bash
docker push ${DOCKER_USER}/${DOCKER_REPOSITORY}:app-${APP_VERSION}
docker push ${DOCKER_USER}/${DOCKER_REPOSITORY}:ai-${AI_WORKER_VERSION}
docker push ${DOCKER_USER}/${DOCKER_REPOSITORY}:frontend-${FRONTEND_VERSION}
```

### Frontend build args

`frontend/Dockerfile`은 Vite 정적 번들을 build한다. `VITE_*` 값은 runtime 환경변수가 아니라 build-time 값이며, 빌드 결과 JS bundle에 포함된다.

사용하는 build arg:

- `VITE_API_BASE_URL`

`VITE_*`에는 브라우저 public config만 넣는다. `OPENAI_API_KEY`, `SMTP_PASSWORD`, `LANGFUSE_SECRET_KEY` 같은 server secret은 절대 build arg로 넘기지 않는다.

### Secret bake-in 점검

backend Dockerfile은 `.env`를 copy하지 않고, root `.dockerignore`는 `.env`, private env 파일, service account JSON, node_modules, cache/output 계열 파일을 제외한다. 그래도 배포 전 아래 명령으로 image metadata와 파일 포함 여부를 확인한다.

```bash
docker history --no-trunc ${DOCKER_USER}/${DOCKER_REPOSITORY}:app-${APP_VERSION}
docker history --no-trunc ${DOCKER_USER}/${DOCKER_REPOSITORY}:ai-${AI_WORKER_VERSION}
docker history --no-trunc ${DOCKER_USER}/${DOCKER_REPOSITORY}:frontend-${FRONTEND_VERSION}

docker inspect ${DOCKER_USER}/${DOCKER_REPOSITORY}:app-${APP_VERSION}
docker inspect ${DOCKER_USER}/${DOCKER_REPOSITORY}:ai-${AI_WORKER_VERSION}
docker inspect ${DOCKER_USER}/${DOCKER_REPOSITORY}:frontend-${FRONTEND_VERSION}
```

이미지 내부에 `.env`나 service account JSON이 들어가지 않았는지 확인한다.

```bash
docker run --rm ${DOCKER_USER}/${DOCKER_REPOSITORY}:app-${APP_VERSION} \
  sh -lc 'find /app -name ".env" -o -name "*service-account*.json"'
docker run --rm ${DOCKER_USER}/${DOCKER_REPOSITORY}:ai-${AI_WORKER_VERSION} \
  sh -lc 'find /app -name ".env" -o -name "*service-account*.json"'
```

frontend bundle에 server secret 키워드가 섞이지 않았는지도 확인한다. server secret 값이나 server secret 변수명은 없어야 한다.

```bash
docker run --rm ${DOCKER_USER}/${DOCKER_REPOSITORY}:frontend-${FRONTEND_VERSION} \
  sh -lc 'grep -R "OPENAI_API_KEY\\|SMTP_PASSWORD\\|LANGFUSE_SECRET_KEY\\|PRIVATE_KEY" -n /usr/share/nginx/html || true'
```

## Storage 기준

운영에서는 `STORAGE_BACKEND=s3`를 기본으로 사용합니다. 업로드 파일은 DB backup에 포함되지 않고 S3 bucket에 별도로 보존됩니다.

```env
STORAGE_BACKEND=s3
S3_BUCKET_NAME=
S3_REGION=ap-northeast-2
S3_PREFIX=ai-health/
S3_PRESIGNED_URL_EXPIRES_SECONDS=3600
```

S3 object는 private 전제로 사용합니다. EC2에는 S3 접근 권한을 가진 IAM Role을 붙이고, AWS access key/secret을 `prod.env`에 넣지 않는 구성을 우선합니다.

### Local storage

로컬 개발 또는 S3 준비 전 임시 확인에서는 `STORAGE_BACKEND=local`을 사용할 수 있습니다. 다만 이번 데모 운영 기준은 S3입니다.

```env
STORAGE_BACKEND=local
LOCAL_STORAGE_ROOT=var/storage
```

local backend는 public URL을 만들지 않으며, prod compose는 `/app/var/storage`를 Docker volume으로 보존합니다. EC2 인스턴스나 volume이 삭제되면 local 파일도 함께 사라질 수 있으므로 운영 업로드 파일은 S3에 저장합니다.

## Database and Redis 기준

이번 데모 운영은 RDS/ElastiCache를 사용하지 않습니다.

```env
DB_HOST=postgres
DB_PORT=5432
REDIS_HOST=redis
REDIS_PORT=6379
```

- `postgres`와 `redis`는 compose service name입니다.
- 두 서비스는 Docker network 내부에서만 접근합니다.
- EC2 security group에서 `5432`, `6379`를 열지 않습니다.
- 장기 운영에서 managed service가 필요해지면 RDS/ElastiCache 전환을 별도 작업으로 검토합니다.

### 데이터 보존 기준

- PostgreSQL 데이터는 Docker named volume `postgres_data`에 저장됩니다.
- Redis 데이터는 Docker named volume `redis_data`에 저장됩니다. prod compose는 Redis AOF를 켜서 Stream/job 상태가 컨테이너 재시작에 최대한 보존되도록 합니다.
- `docker compose up -d`, 이미지 pull, 컨테이너 재생성, EC2 재부팅만으로는 named volume이 삭제되지 않습니다.
- `docker compose down`만 실행해도 named volume은 기본적으로 남습니다.
- `docker compose down -v`, Docker volume 수동 삭제, EBS/EC2 인스턴스 삭제를 하면 DB/Redis 데이터가 사라질 수 있습니다.
- EC2 종료, EBS 정리, compose volume 삭제 전에는 PostgreSQL 백업을 먼저 남깁니다.
- Redis는 캐시/큐 성격이 강하므로 PostgreSQL처럼 장기 원장으로 보지 않습니다. 처리 중인 job이 중요한 시점에는 배포 전 worker idle 상태와 DLQ를 확인합니다.

## Domain and HTTPS

### DNS

도메인 DNS에서 EC2 public IPv4 주소를 가리키는 A record를 설정합니다.

- 예: `your-domain.example A <EC2_PUBLIC_IP>`
- `COOKIE_DOMAIN`, `CORS_ALLOW_ORIGINS`, `FRONTEND_BASE_URL`, Nginx `server_name`은 같은 운영 도메인 기준으로 맞춥니다.

### Nginx 설정

- HTTP bootstrap 설정: `infra/nginx/prod_http.conf`
- HTTPS 운영 설정: `infra/nginx/prod_https.conf`
- prod compose는 `NGINX_CONF` 값으로 mount할 Nginx 설정 파일을 고를 수 있습니다.

`prod_https.conf`의 placeholder는 운영 도메인으로 교체해야 합니다.

- `server_name your-domain.example;`
- `/etc/letsencrypt/live/your-domain.example/fullchain.pem`
- `/etc/letsencrypt/live/your-domain.example/privkey.pem`

실제 인증서 파일은 Git에 커밋하지 않습니다. `certbot-conf` Docker volume 또는 운영 서버의 secret mount로 관리합니다.

### Certbot bootstrap 예시

인증서가 아직 없으면 HTTPS Nginx가 시작되지 않을 수 있습니다. 처음에는 HTTP 설정으로 Nginx를 띄운 뒤 certbot으로 인증서를 발급합니다.

```bash
NGINX_CONF=../nginx/prod_http.conf \
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml up -d nginx
```

certbot webroot 발급 예시:

```bash
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml run --rm certbot \
  certonly --webroot -w /var/www/certbot \
  -d your-domain.example \
  --email admin@example.com \
  --agree-tos \
  --no-eff-email
```

인증서 발급 후 `prod.env`의 `NGINX_CONF=../nginx/prod_https.conf`를 확인하고 Nginx를 재시작합니다.

```bash
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml up -d nginx
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml exec nginx nginx -t
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml exec nginx nginx -s reload
```

### Security headers

`prod_https.conf`는 HTTPS server block에서 다음 헤더를 적용합니다.

- `Strict-Transport-Security`
- `X-Content-Type-Options`
- `X-Frame-Options`
- `Referrer-Policy`
- `Permissions-Policy`

`Strict-Transport-Security`는 브라우저가 HTTPS를 강하게 기억하게 하므로, 도메인/인증서/HTTPS 운영이 안정화된 뒤 적용 상태를 유지하세요. Content-Security-Policy는 프론트 asset과 API endpoint를 모두 확인한 뒤 별도 강화하는 편이 안전합니다.

## 실행 순서

이미지를 먼저 push한 뒤 EC2에서 실행합니다. prod compose는 pull-only 구조이므로 EC2에서 app/frontend/worker 이미지를 build하지 않는다.

`docker-compose.prod.yml`은 외부 Docker network `ai-health-shared`를 사용합니다. EC2 최초 배포 전 한 번 생성합니다.

```bash
docker network create ai-health-shared || true
```

EC2에서는 `envs/example.prod.env`를 복사한 `prod.env`를 준비하고, 실제 운영 값은 `prod.env`에만 채운다. 최초 IP/HTTP 확인 또는 Let's Encrypt 인증서 발급 전에는 HTTPS 설정 대신 HTTP bootstrap 설정을 사용한다.

```env
NGINX_CONF=../nginx/prod_http.conf
```

이미지를 pull하고 컨테이너를 실행한다.

```bash
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml pull
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml up -d
```

`REFRESH_TOKEN_COOKIE_SECURE=true` 상태에서는 HTTP/IP 접속 smoke test에서 refresh cookie 기반 로그인 유지가 제한될 수 있다. 이 값은 HTTPS/도메인 운영 기준으로는 true를 유지해야 하며, HTTP 임시 테스트에서만 쿠키 동작 제약을 감안한다.

마이그레이션:

```bash
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml exec fastapi \
  uv run --no-sync aerich upgrade
```

챌린지 seed가 필요한 초기 환경:

```bash
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml exec fastapi \
  uv run --no-sync python scripts/seed_mvp_challenges.py
```

기본 health check:

```bash
curl -fsS http://localhost/api/v1/system/health
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml ps
```

도메인 DNS, 인증서, Nginx HTTPS 설정이 준비되면 `prod.env`를 HTTPS 설정으로 전환한다.

```env
NGINX_CONF=../nginx/prod_https.conf
```

이후 Nginx 설정을 재적용하고 외부 HTTPS health check를 확인한다.

```bash
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml up -d nginx
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml exec nginx nginx -t
curl -fsS https://your-domain.example/api/v1/system/health
```

## PostgreSQL 백업/복구

EC2 내부 postgres 컨테이너를 사용하므로 EC2 인스턴스 종료, EBS 삭제, volume 삭제 전에는 반드시 DB 백업을 남깁니다.

운영 편의 스크립트:

```bash
./scripts/ops/backup_postgres.sh
./scripts/ops/restore_postgres.sh var/backups/postgres/backup.sql
```

기본값:

- env file: `prod.env`
- compose file: `infra/docker/docker-compose.prod.yml`
- backup dir: `var/backups/postgres/`

다른 경로를 쓰는 경우:

```bash
ENV_FILE=/home/ubuntu/app/prod.env BACKUP_DIR=/home/ubuntu/backups ./scripts/ops/backup_postgres.sh
ENV_FILE=/home/ubuntu/app/prod.env ./scripts/ops/restore_postgres.sh /home/ubuntu/backups/backup.sql
```

직접 백업 명령:

```bash
mkdir -p var/backups/postgres
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml exec -T postgres \
  sh -lc 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  > var/backups/postgres/ai_health_$(date +%Y%m%d_%H%M%S).sql
```

직접 복구 명령:

```bash
cat var/backups/postgres/backup.sql | docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml exec -T postgres \
  sh -lc 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
```

복구는 대상 DB 상태를 확인한 뒤 진행합니다. 기존 데이터가 있는 DB에 그대로 복구하면 충돌이 날 수 있습니다. `restore_postgres.sh`는 파일 경로 인자를 요구하고 확인 문구를 입력해야 진행됩니다.

S3 업로드 파일은 S3 bucket에 저장되므로 PostgreSQL dump에는 포함되지 않습니다. DB 백업과 S3 bucket 보존/버전 관리 정책은 별도로 관리해야 합니다. 백업 파일을 S3에 올리는 자동화는 이번 데모 운영 필수 범위가 아니며, 필요하면 운영자가 별도 CLI/cron으로 구성합니다.

## Health check

운영 편의 스크립트:

```bash
./scripts/ops/check_prod_health.sh
```

도메인 연결 후 외부 HTTPS까지 확인하려면:

```bash
PUBLIC_BASE_URL=https://your-domain.example ./scripts/ops/check_prod_health.sh
```

수동 확인:

```bash
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml ps
curl -fsS http://localhost/api/v1/system/health
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml exec postgres \
  sh -lc 'pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml exec redis redis-cli ping
```

외부 도메인을 연결한 뒤에는 다음도 확인합니다.

```bash
curl -fsS https://your-domain.example/api/v1/system/health
```

## 로그 확인 루틴

서비스별 로그:

```bash
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml logs --tail=100 fastapi
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml logs --tail=100 ai-worker
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml logs --tail=100 nginx
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml logs --tail=100 postgres
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml logs --tail=100 redis
```

실시간 추적:

```bash
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml logs -f fastapi
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml logs -f ai-worker
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml logs -f nginx
```

장애 상황별 첫 확인 위치:

- API 500: `fastapi` 로그, PostgreSQL 연결 상태, 최근 migration 적용 여부
- 프론트 접속 불가: `nginx` 로그, `frontend` 컨테이너 상태, 80/443 보안 그룹, 인증서 경로
- OCR/식단/분석 job 멈춤: `ai-worker` 로그, `redis` ping, Redis Stream pending/DLQ 상태
- Redis Stream 처리 지연: `ai-worker` 로그와 `redis` 로그를 함께 확인
- DB 연결 실패: `postgres` 로그, `pg_isready`, `prod.env`의 `DB_HOST=postgres` 확인
- S3 업로드 실패: `fastapi` 또는 `ai-worker` 로그, `STORAGE_BACKEND=s3`, `S3_BUCKET_NAME`, EC2 IAM Role 권한 확인

## EC2 보안 그룹 포트

열어야 하는 포트:

- `80/tcp`: nginx HTTP
- `443/tcp`: nginx HTTPS
- `22/tcp`: SSH, 운영자 IP로 제한

열지 말아야 하는 포트:

- `5432/tcp`: Postgres
- `6379/tcp`: Redis
- `8000/tcp`: FastAPI container
- frontend 내부 포트: nginx container에서만 접근

## 배포 전 검증

로컬 또는 CI에서 먼저 확인합니다.

```bash
git diff --check
uv run ruff check app scripts ai_runtime tests
uv run ruff format app scripts ai_runtime tests --check
CHATBOT_USE_REAL_LLM=false OPENAI_API_KEY= LANGFUSE_ENABLED=false RAG_ENABLED=false uv run pytest tests
cd frontend && npm run build && cd ..
```

prod compose config 렌더링도 확인합니다. 출력에 secret 원문이 포함될 수 있으므로 공유하거나 PR에 붙이지 않습니다.

```bash
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml config
```

## 운영 체크 포인트

- `EMAIL_VERIFICATION_DEBUG=false`
- `PASSWORD_RESET_DEBUG=false`
- `REFRESH_TOKEN_COOKIE_SECURE=true`
- `CORS_ALLOW_ORIGINS`는 실제 프론트 도메인만 포함
- `STORAGE_BACKEND=s3` 사용 시 private bucket과 IAM Role 권한 확인
- AI provider flag는 비용/관측/secret 준비 후 단계적으로 활성화
- Redis/Postgres는 EC2 외부로 직접 노출하지 않음
