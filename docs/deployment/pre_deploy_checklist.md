# Health Ladder 배포 전 최종 점검표

이 문서는 Health Ladder 배포 전/후 확인 항목을 빠르게 점검하기 위한 체크리스트입니다.
실제 secret, API key, password, private key 값은 이 문서나 Git tracked 파일에 작성하지 않습니다.

EC2 + Docker Compose + DuckDNS + HTTPS 전체 절차는 `docs/deployment/ec2_docker_deploy_guide.md`를 기준으로 확인합니다.

## 1. 로컬 검증

- [ ] 현재 브랜치와 변경 상태 확인

```bash
git status -sb
```

- [ ] whitespace / diff 오류 확인

```bash
git diff --check
```

- [ ] Python lint 확인

```bash
uv run ruff check app ai_runtime scripts tests
```

- [ ] Python format 확인

```bash
uv run ruff format --check app ai_runtime scripts tests
```

- [ ] Python test 확인

```bash
uv run pytest -q
```

- [ ] Frontend production build 확인

```bash
cd frontend && npm run build && cd ..
```

## 2. Dev Stack 확인

- [ ] dev stack health 확인

```bash
make dev-health
```

- [ ] backend / ai-worker 수정 후 dev stack에 재반영

```bash
make dev-rebuild-api
```

- [ ] dev Nginx 502 또는 upstream 꼬임 복구

```bash
make dev-restart-nginx
```

## 3. Docker 이미지 검증

운영 compose는 EC2에서 이미지를 build하지 않고 Docker Hub에서 pull합니다.
배포 전 app, ai-worker, frontend image가 같은 tag 규칙으로 준비되어야 합니다.
같은 `v1.0.0` tag를 반복 재사용하지 말고 배포마다 새 version을 정합니다.

- [ ] 배포용 이미지 build 검증

```bash
export APP_VERSION=v1.0.1
export AI_WORKER_VERSION=v1.0.1
export FRONTEND_VERSION=v1.0.1
make image-tags
make image-build-check
```

- [ ] 이미지 architecture 확인

```bash
docker image inspect kdu0312/ai-health:app-${APP_VERSION} --format '{{.Os}}/{{.Architecture}}'
docker image inspect kdu0312/ai-health:ai-${AI_WORKER_VERSION} --format '{{.Os}}/{{.Architecture}}'
docker image inspect kdu0312/ai-health:frontend-${FRONTEND_VERSION} --format '{{.Os}}/{{.Architecture}}'
```

기대값:

```text
linux/amd64
```

- [ ] Docker Hub push

```bash
make image-push
```

- [ ] EC2 `.prod.env`의 image version 갱신

```env
APP_VERSION=v1.0.1
AI_WORKER_VERSION=v1.0.1
FRONTEND_VERSION=v1.0.1
```

## 4. .prod.env 확인

`.prod.env`는 운영 서버별 secret과 설정을 담는 로컬 파일입니다. 절대 commit하지 않습니다.
`envs/example.prod.env`는 템플릿 파일로 유지합니다.

- [ ] 템플릿 복사

```bash
cp envs/example.prod.env .prod.env
```

- [ ] Git ignore 확인

```bash
git check-ignore -v .prod.env
```

- [ ] 필수 변수 이름 확인

아래 값은 실제 운영 값으로 채우되, 값 자체는 문서/PR/로그에 남기지 않습니다.

- `SECRET_KEY`
- `COOKIE_DOMAIN`
- `CORS_ALLOW_ORIGINS`
- `FRONTEND_BASE_URL`
- `DB_HOST`
- `DB_PORT`
- `DB_USER`
- `DB_PASSWORD`
- `DB_NAME`
- `REDIS_HOST`
- `REDIS_PORT`
- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_FROM_EMAIL`
- `SMTP_FROM_NAME`
- `OPENAI_API_KEY`
- `LANGFUSE_BASE_URL`
- `LANGFUSE_HOST`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `VITE_API_BASE_URL`
- `NGINX_CONF`

`healthladder.duckdns.org` 기준으로 아래 값이 맞는지 확인합니다.

```env
COOKIE_DOMAIN=healthladder.duckdns.org
CORS_ALLOW_ORIGINS=https://healthladder.duckdns.org
FRONTEND_BASE_URL=https://healthladder.duckdns.org
VITE_API_BASE_URL=/api/v1
```

`OPENAI_API_KEY`, `LANGFUSE_*`, SMTP 값은 사용하는 provider를 켤 때만 실제 값이 필요합니다. prod에서는 debug/mock/test 계열 flag를 기본 false로 두고, `EMAIL_VERIFICATION_DEBUG=false`, `PASSWORD_RESET_DEBUG=false`, `REFRESH_TOKEN_COOKIE_SECURE=true`를 유지합니다.

DuckDNS token은 secret입니다. `.prod.env`, 문서, PR, issue, shell history, 배포 로그에 원문을 남기지 않습니다.

## 5. EC2 배포 순서

- [ ] 서버에서 최신 코드 반영

```bash
git pull origin release
```

- [ ] `.prod.env` 작성

```bash
cp envs/example.prod.env .prod.env
# .prod.env 실제 운영값 수정
```

최초 HTTPS 인증서 발급 전에는 HTTP bootstrap을 위해 `.prod.env`에서 아래 값을 사용합니다.

```env
NGINX_CONF=../nginx/prod_http.conf
NGINX_HTTP_PORT=80
```

- [ ] DuckDNS DNS 확인

`healthladder.duckdns.org`가 현재 EC2 public IP를 가리키는지 확인합니다.

```bash
nslookup healthladder.duckdns.org
curl -I http://healthladder.duckdns.org
```

- [ ] Docker Hub 이미지 pull

```bash
make prod-pull
```

- [ ] prod stack 실행

```bash
make prod-up
```

`make prod-up`은 prod compose의 external network인 `ai-health-shared`가 없으면 생성합니다. 수동 compose 명령을 사용하는 경우에는 먼저 `docker network inspect ai-health-shared >/dev/null 2>&1 || docker network create ai-health-shared`로 보장합니다.

- [ ] DB migration 실행

```bash
make prod-migrate
```

챌린지 seed가 필요한 초기 환경에서는 운영자가 명시적으로 승인한 뒤 danger target을 별도로 실행합니다. 일반 배포 흐름에서는 seed를 자동 실행하지 않습니다.

```bash
make danger-prod-seed-challenges
```

FAQ 테이블은 migration으로 생성되지만 FAQ row는 별도 seed입니다. FAQ 목록이 비어 있고 초기 FAQ seed가 필요할 때만 실행합니다.

```bash
make danger-prod-seed-faqs
```

- [ ] 컨테이너 상태 확인

```bash
make prod-ps
```

- [ ] 서버 내부 health check

```bash
make prod-health
```

`make prod-health`는 `.prod.env`의 `NGINX_HTTP_PORT`를 읽고, 값이 없으면 운영 기본 `80`으로 `http://localhost:80/api/v1/system/health`를 확인합니다. HTTPS 전환 전에는 HTTP health가 먼저 성공해야 합니다.

- [ ] HTTPS 적용 후 외부 health 확인

```bash
curl -I https://healthladder.duckdns.org
curl -fsS https://healthladder.duckdns.org/api/v1/system/health
```

## 6. 배포 후 기능 확인

- [ ] 프론트 접속 확인
- [ ] 회원가입 이메일 인증 코드 발송 확인
- [ ] 가족 초대 이메일 발송 확인
- [ ] ai-worker 로그 확인

```bash
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml logs --tail=100 ai-worker
```

- [ ] nginx 로그 확인

```bash
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml logs --tail=100 nginx
```

- [ ] DB / Redis health 확인

```bash
curl -fsS http://localhost/api/v1/system/health
```

## 7. Worker / 알림 확인

현재 MVP 배포에서는 별도 `notification-worker`, `email-worker`, `scheduler-worker`를 실행하지 않습니다. `ai-worker` 하나가 Redis Stream consumer를 실행하며 AI/OCR/ML job과 service job을 함께 처리합니다.

현재 `ai-worker`가 처리하는 service job:

- `email.verification.send`
- `password_reset.email.send`
- `family.invite.email.send`
- `family.notification.create`

`SCHEDULER_ENABLED=true`이면 notification scheduler도 `ai-worker` 안에서 실행됩니다.

### 현재 기준 체크

- [ ] `ai-worker` running 확인

```bash
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml ps ai-worker
```

- [ ] `ai-worker` logs 확인

```bash
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml logs --tail=100 ai-worker
```

- [ ] 이메일 인증 job 처리 확인
  - 회원가입 이메일 인증 코드 발송을 요청한다.
  - `ai-worker` 로그에서 service job 처리 실패가 없는지 확인한다.

- [ ] `SCHEDULER_ENABLED` 값 확인
  - 예약 알림/리마인더를 운영에서 사용할 때만 `true`로 둔다.
  - 현재 구조에서는 scheduler가 `ai-worker` 내부에서 함께 실행된다.

### 향후 worker 분리 후 체크

worker를 분리한 뒤에는 아래 항목을 별도로 확인합니다. 현재 compose에는 아직 해당 service가 없습니다.

- [ ] `email-worker` running 확인
- [ ] `notification-worker` running 확인
- [ ] `scheduler-worker` running 확인
- [ ] 각 worker별 Redis Stream consumer group 확인
- [ ] 각 worker별 logs 확인
- [ ] email/notification/scheduler job이 AI/OCR job 적체와 독립적으로 처리되는지 확인

## 8. 자주 나는 문제

### Mac에서 prod-pull 시 linux/arm64 관련 오류

EC2 배포 대상은 `linux/amd64` 기준입니다. Mac에서 pull/platform 문제가 나면 아래처럼 확인합니다.

```bash
DOCKER_DEFAULT_PLATFORM=linux/amd64 make prod-pull
```

### dev Nginx 502

FastAPI 컨테이너 재생성 후 dev Nginx가 이전 upstream IP를 물고 있으면 아래 명령으로 복구합니다.

```bash
make dev-restart-nginx
```

### .prod.env 누락

`.prod.env`가 없으면 `make prod-pull`, `make prod-up`이 실패합니다.

```bash
cp envs/example.prod.env .prod.env
git check-ignore -v .prod.env
```

### SMTP 535 Authentication failed

Brevo SMTP 인증 실패입니다. 아래 값을 다시 확인합니다.

- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_FROM_EMAIL`
- Brevo SMTP 활성화 여부
- Brevo sender 인증 여부
