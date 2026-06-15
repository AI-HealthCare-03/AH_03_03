# Health Ladder 배포 전 최종 점검표

이 문서는 Health Ladder 배포 전/후 확인 항목을 빠르게 점검하기 위한 체크리스트입니다.
실제 secret, API key, password, private key 값은 이 문서나 Git tracked 파일에 작성하지 않습니다.

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

- [ ] 배포용 이미지 build 검증

```bash
make image-build-check
```

- [ ] 이미지 architecture 확인

```bash
docker image inspect kdu0312/ai-health:app-v1.0.0 --format '{{.Os}}/{{.Architecture}}'
docker image inspect kdu0312/ai-health:ai-v1.0.0 --format '{{.Os}}/{{.Architecture}}'
docker image inspect kdu0312/ai-health:frontend-v1.0.0 --format '{{.Os}}/{{.Architecture}}'
```

기대값:

```text
linux/amd64
```

- [ ] Docker Hub push

```bash
docker push kdu0312/ai-health:app-v1.0.0
docker push kdu0312/ai-health:ai-v1.0.0
docker push kdu0312/ai-health:frontend-v1.0.0
```

## 4. prod.env 확인

`prod.env`는 운영 서버별 secret과 설정을 담는 로컬 파일입니다. 절대 commit하지 않습니다.
`envs/example.prod.env`는 템플릿 파일로 유지합니다.

- [ ] 템플릿 복사

```bash
cp envs/example.prod.env prod.env
```

- [ ] Git ignore 확인

```bash
git check-ignore -v prod.env
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
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `VITE_API_BASE_URL`
- `NGINX_CONF`

## 5. EC2 배포 순서

- [ ] 서버에서 최신 코드 반영

```bash
git pull origin feature/kdu
```

- [ ] `prod.env` 작성

```bash
cp envs/example.prod.env prod.env
# prod.env 실제 운영값 수정
```

- [ ] Docker Hub 이미지 pull

```bash
make prod-pull
```

- [ ] prod stack 실행

```bash
make prod-up
```

- [ ] DB migration과 seed 명시 실행

```bash
make prod-release-db
```

명시적으로 나누어 실행하려면 아래 순서로 실행합니다.

```bash
make prod-migrate
make prod-seed
```

- [ ] 컨테이너 상태 확인

```bash
make prod-ps
```

- [ ] 서버 내부 health check

```bash
make prod-health
```

## 6. 배포 후 기능 확인

- [ ] 프론트 접속 확인
- [ ] 회원가입 이메일 인증 코드 발송 확인
- [ ] 가족 초대 이메일 발송 확인
- [ ] ai-worker 로그 확인

```bash
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml logs --tail=100 ai-worker
```

- [ ] nginx 로그 확인

```bash
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml logs --tail=100 nginx
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
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml ps ai-worker
```

- [ ] `ai-worker` logs 확인

```bash
docker compose --env-file prod.env -f infra/docker/docker-compose.prod.yml logs --tail=100 ai-worker
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

### prod.env 누락

`prod.env`가 없으면 `make prod-pull`, `make prod-up`이 실패합니다.

```bash
cp envs/example.prod.env prod.env
git check-ignore -v prod.env
```

### SMTP 535 Authentication failed

Brevo SMTP 인증 실패입니다. 아래 값을 다시 확인합니다.

- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_FROM_EMAIL`
- Brevo SMTP 활성화 여부
- Brevo sender 인증 여부
