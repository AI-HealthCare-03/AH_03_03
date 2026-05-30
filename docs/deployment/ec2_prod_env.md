# EC2 Production Deployment Checklist

이 문서는 EC2에서 `infra/docker/docker-compose.prod.yml` 기준으로 운영 스택을 띄울 때 확인할 항목을 정리한 것입니다. 실제 secret, API key, private key 원문은 이 문서나 Git tracked 파일에 작성하지 않습니다.

## Compose 기준

- 시연/운영 스택 기준 파일: `infra/docker/docker-compose.prod.yml`
- 로컬 개발 스택 기준 파일: `infra/docker/docker-compose.dev.yml`
- prod compose는 외부에 `nginx:80`만 노출합니다.
- Postgres와 Redis는 Docker network 내부 `expose`만 사용하며 EC2 보안 그룹에서 직접 열지 않습니다.
- FastAPI와 ai-worker는 각각 DB connection pool을 만들기 때문에 `DB_POOL_MAX_SIZE`는 작게 시작합니다.

## EC2에서 준비할 환경 파일

운영 서버에서는 예시 파일을 복사해서 별도 파일로 관리합니다.

```bash
cp envs/example.prod.env .env.prod
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
- `CLOVA_OCR_API_URL`, `CLOVA_OCR_SECRET_KEY`를 사용하는 경우
- `S3_BUCKET_NAME`
- Firebase Admin SDK를 쓰는 경우 `GOOGLE_APPLICATION_CREDENTIALS` 경로

작성하지 말아야 하는 값:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- Firebase service account JSON 원문
- SMTP password 원문을 문서/README/예시 파일에 직접 작성한 값
- OpenAI, Langfuse, Clova secret 원문

## Storage 선택 기준

### S3 권장

운영에서는 `STORAGE_BACKEND=s3`를 권장합니다.

```env
STORAGE_BACKEND=s3
S3_BUCKET_NAME=your-private-bucket-name
S3_REGION=ap-northeast-2
S3_PREFIX=ai-health/prod
S3_PRESIGNED_URL_EXPIRES_SECONDS=3600
```

S3 object는 private 전제로 사용합니다. EC2에는 S3 접근 권한을 가진 IAM Role을 붙이고, AWS access key/secret을 `.env.prod`에 넣지 않는 구성을 우선합니다.

### Local storage

단일 EC2 임시 운영이나 S3 준비 전에는 `STORAGE_BACKEND=local`을 사용할 수 있습니다.

```env
STORAGE_BACKEND=local
LOCAL_STORAGE_ROOT=var/storage
```

local backend는 public URL을 만들지 않으며, prod compose는 `/app/var/storage`를 Docker volume으로 보존합니다. 다중 인스턴스나 무중단 배포에는 S3로 전환해야 합니다.

## 실행 순서

이미지를 먼저 push한 뒤 EC2에서 실행합니다.

```bash
docker compose --env-file .env.prod -f infra/docker/docker-compose.prod.yml pull
docker compose --env-file .env.prod -f infra/docker/docker-compose.prod.yml up -d
```

마이그레이션:

```bash
docker compose --env-file .env.prod -f infra/docker/docker-compose.prod.yml exec fastapi \
  uv run --no-sync aerich upgrade
```

챌린지 seed가 필요한 초기 환경:

```bash
docker compose --env-file .env.prod -f infra/docker/docker-compose.prod.yml exec fastapi \
  uv run --no-sync python scripts/seed_mvp_challenges.py
```

## Health check

```bash
docker compose --env-file .env.prod -f infra/docker/docker-compose.prod.yml ps
curl -fsS http://localhost/api/v1/system/health
```

외부 도메인을 연결한 뒤에는 다음도 확인합니다.

```bash
curl -fsS https://your-domain.example/api/v1/system/health
```

## EC2 보안 그룹 포트

기본 운영 HTTP 구성:

- `80/tcp`: nginx HTTP
- `22/tcp`: SSH, 운영자 IP로 제한

TLS를 직접 붙이는 경우:

- `443/tcp`: HTTPS

열지 말아야 하는 포트:

- `5432/tcp`: Postgres
- `6379/tcp`: Redis
- `8000/tcp`: FastAPI container

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
docker compose --env-file .env.prod -f infra/docker/docker-compose.prod.yml config
```

## 운영 체크 포인트

- `EMAIL_VERIFICATION_DEBUG=false`
- `PASSWORD_RESET_DEBUG=false`
- `REFRESH_TOKEN_COOKIE_SECURE=true`
- `CORS_ALLOW_ORIGINS`는 실제 프론트 도메인만 포함
- `STORAGE_BACKEND=s3` 사용 시 private bucket과 IAM Role 권한 확인
- AI provider flag는 비용/관측/secret 준비 후 단계적으로 활성화
- Redis/Postgres는 EC2 외부로 직접 노출하지 않음
