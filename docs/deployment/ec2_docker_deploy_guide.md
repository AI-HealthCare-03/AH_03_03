# EC2 Docker Compose Deployment Guide

이 문서는 Health Ladder MVP를 `EC2 + Docker Compose + DuckDNS + HTTPS` 기준으로 배포할 때 따라가는 실행 가이드입니다. 실제 secret, DuckDNS token, SMTP password, OpenAI key, DB password는 문서/PR/이슈/로그에 남기지 않습니다.

기준 파일:

- prod compose: `infra/docker/docker-compose.prod.yml`
- dev compose: `infra/docker/docker-compose.dev.yml`
- prod env: `.prod.env`
- prod env template: `envs/example.prod.env`
- prod domain: `healthladder.duckdns.org`
- HTTPS Nginx: `infra/nginx/prod_https.conf`
- HTTP bootstrap Nginx: `infra/nginx/prod_http.conf`

## 1. 배포 구조 개요

```text
Client
  ↓
Nginx :80 / :443
  ├─ /        → frontend:80
  └─ /api/*   → fastapi:8000
                 ↓
              PostgreSQL
              Redis
              ai-worker
```

- 프론트 정적 파일은 `frontend` 이미지 안의 Nginx가 서빙하고, 외부 Nginx가 같은 도메인의 `/`로 proxy합니다.
- API는 외부 Nginx가 `/api/` 요청을 `fastapi:8000`으로 proxy합니다.
- 외부 health check는 Nginx 경유 기준인 `https://healthladder.duckdns.org/api/v1/system/health`로 확인합니다.
- Swagger는 Nginx 경유 기준으로 `/api/docs`를 사용합니다.
- 현재 프론트 API client는 versioned base path를 사용하므로 운영 `VITE_API_BASE_URL=/api/v1`을 유지합니다. 일반적인 same-origin API base를 `/api`로 두는 구조와 다릅니다.

## 2. EC2 인스턴스 준비

권장 시작점:

- Ubuntu LTS
- Docker Engine과 Docker Compose plugin 사용
- EC2 public IPv4 확인
- EBS/volume 삭제 전 PostgreSQL backup 계획 수립
- MVP local storage 기준으로 Docker volume 보존 계획 수립. S3 storage로 전환할 때만 private bucket과 EC2 IAM Role 권한 준비

EC2 내부에서 `postgres`, `redis`, `fastapi`, `ai-worker`, `frontend`, `nginx`, `certbot` 컨테이너를 Compose로 실행합니다. 이번 MVP 배포 기준에서는 RDS/ElastiCache가 아니라 EC2 내부 Compose service를 사용합니다.

## 3. 보안 그룹 설정

외부 inbound:

| Port | 용도 | 권장 source |
| --- | --- | --- |
| 22 | SSH | 내 IP만 허용 |
| 80 | HTTP / Let's Encrypt webroot | `0.0.0.0/0`, 필요 시 IPv6도 허용 |
| 443 | HTTPS | `0.0.0.0/0`, 필요 시 IPv6도 허용 |

열지 않는 포트:

- `8000`: FastAPI 직접 노출 금지
- `5432`: PostgreSQL 직접 노출 금지
- `6379`: Redis 직접 노출 금지
- frontend 내부 `80`: 외부 직접 노출 금지

외부 공개는 Nginx `80/443`만 기준으로 합니다.

## 4. Docker Engine 설치

EC2에는 Docker Engine과 Compose plugin을 설치합니다. 설치 후 아래 정도만 확인합니다.

```bash
docker --version
docker compose version
```

이 문서에서는 설치 스크립트를 고정하지 않습니다. AWS/Ubuntu 공식 절차 또는 Docker 공식 문서를 기준으로 설치하고, 배포 전에는 Docker daemon이 재부팅 후 자동 시작되는지 확인합니다.

## 5. Docker login

prod compose는 EC2에서 이미지를 build하지 않고 registry에서 pull합니다. private repository를 쓰면 EC2에서 Docker login이 필요합니다.

```bash
docker login
```

Docker Hub token/password는 shell history나 문서에 남기지 않습니다.

## 6. DuckDNS 도메인 연결

운영 도메인:

```text
healthladder.duckdns.org
```

DuckDNS에서 이 도메인이 EC2 public IPv4를 가리키게 합니다. DuckDNS token은 secret입니다. token이 보이는 화면 캡처를 공유하지 말고, 문서/PR/README/이슈/로그에 기록하지 않습니다.

확인 예시:

```bash
nslookup healthladder.duckdns.org
curl -I http://healthladder.duckdns.org
```

`curl -I http://...`는 인증서 발급 전 HTTP bootstrap 확인용입니다.

## 7. .prod.env 작성

표준 생성:

```bash
cp envs/example.prod.env .prod.env
```

`.prod.env`는 운영 서버 로컬 secret 파일이며 Git에 올리지 않습니다. 실제 값은 화면 공유/로그/PR에 노출하지 않습니다.

중요 값:

```env
ENV=prod
COOKIE_DOMAIN=healthladder.duckdns.org
FRONTEND_BASE_URL=https://healthladder.duckdns.org
CORS_ALLOW_ORIGINS=https://healthladder.duckdns.org
VITE_API_BASE_URL=/api/v1
EMAIL_ENABLED=true
EMAIL_VERIFICATION_DEBUG=false
PASSWORD_RESET_DEBUG=false
REFRESH_TOKEN_COOKIE_SECURE=true
REFRESH_TOKEN_COOKIE_SAMESITE=lax
NGINX_CONF=../nginx/prod_https.conf
```

`ENV`는 코드상 `prod`와 `production`을 모두 production 모드로 처리합니다. 현재 `envs/example.prod.env`와 prod compose 표준은 `ENV=prod`입니다.

운영자가 실제 값으로 채워야 하는 secret/환경값:

- `SECRET_KEY`
- `DB_*`
- `REDIS_*`
- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_FROM_EMAIL`
- `OPENAI_API_KEY` 또는 사용하는 LLM provider secret
- `LANGFUSE_*`를 사용하는 경우
- `S3_BUCKET_NAME`은 `STORAGE_BACKEND=s3`를 선택할 때만 필요

Brevo를 쓰는 경우에도 이 프로젝트에서는 전용 Brevo API key 연동이 아니라 SMTP provider 설정으로 취급합니다. Twilio/SMS는 현재 MVP 운영 필수 범위가 아니며, 관련 값은 optional로 둡니다.

MVP 운영 기준 storage:

```env
STORAGE_BACKEND=local
UPLOAD_STORAGE_DIR=var/uploads
LOCAL_STORAGE_ROOT=var/storage
S3_BUCKET_NAME=
```

local storage 사용 시 `S3_BUCKET_NAME`은 비어 있어도 됩니다. OCR/검진 업로드가 500으로 실패하면 먼저 운영 `.prod.env`의 storage 값을 확인합니다.

식단 이미지 분석은 실제 provider가 준비되어야 합니다. `DIET_GPT_VISION_ENABLED=false`, `DIET_VISION_PROVIDER=rule_based`, `DIET_DEMO_FALLBACK_ENABLED=false` 조합에서는 더미 음식/점수를 저장하지 않고 service unavailable로 종료됩니다. 운영에서 GPT Vision을 사용할 때만 아래 값을 `.prod.env`에 설정합니다.

```env
OPENAI_API_KEY=<SET_IN_PROD>
DIET_VISION_PROVIDER=gpt_vision
DIET_GPT_VISION_ENABLED=true
DIET_GPT_VISION_MODEL=gpt-4o
GPT_VISION_FALLBACK_ENABLED=true
DIET_DEMO_FALLBACK_ENABLED=false
```

건강검진 OCR도 실제 provider가 준비되어야 합니다. 운영 기본값이 아래처럼 provider 미설정 상태라면 더미 검진값을 저장하지 않고 OCR job이 실패합니다.

```env
EXAM_OCR_PROVIDER=fallback
EXAM_GPT_VISION_ENABLED=false
PADDLE_OCR_ENABLED=false
GPT_VISION_FALLBACK_ENABLED=false
```

운영에서 GPT Vision 건강검진 OCR을 사용할 때만 아래 값을 `.prod.env`에 설정합니다.

```env
OPENAI_API_KEY=<SET_IN_PROD>
EXAM_OCR_PROVIDER=gpt_vision
EXAM_GPT_VISION_ENABLED=true
EXAM_GPT_VISION_MODEL=gpt-4o
GPT_VISION_FALLBACK_ENABLED=true
```

`EXAM_OCR_PROVIDER`는 `fallback`, `auto`, `paddleocr`, `gpt_vision` 값을 사용합니다. `fallback`은 OCR provider 후보가 없다는 뜻이며 성공용 더미 provider가 아닙니다. OCR 측정값 조회 시 실제 FK 컬럼명은 `exam_report_id`입니다.

## 8. Docker image pull/build 정책

EC2는 pull-only 운영입니다. EC2에서 app/frontend/worker 이미지를 build하지 않습니다.

배포 전 로컬 또는 CI에서 이미지를 준비합니다.

```bash
make image-tags
make image-build-check
```

EC2에서는 pull:

```bash
make prod-pull
```

운영 서버가 실제로 pull하는 이미지 tag는 `.prod.env`의 `APP_VERSION`, `AI_WORKER_VERSION`, `FRONTEND_VERSION` 기준을 따릅니다. 같은 tag를 반복 재사용하면 EC2 `main` 브랜치는 최신이어도 예전 frontend image가 계속 서빙될 수 있으므로, 배포마다 `main-<short-sha>`, `v1.0.1`, `20260616-1` 같은 새 tag를 발급하고 Docker Hub push 후 `.prod.env`의 세 version 값을 함께 갱신합니다.

frontend 정적 bundle 반영 여부는 배포 후 컨테이너 내부 asset hash로 확인합니다.

```bash
docker inspect frontend --format 'IMAGE={{.Config.Image}} CREATED={{.Created}}'
docker exec frontend ls -al /usr/share/nginx/html/assets | grep -E 'index-|ExamOcrPage|AdminDashboard|AdminFaq' || true
```

## 9. Docker Compose prod 실행

표준 흐름:

```bash
cp envs/example.prod.env .prod.env
# 최초 HTTPS 인증서 발급 전에는 .prod.env에서 NGINX_CONF=../nginx/prod_http.conf 사용
make prod-pull
make prod-up
make prod-migrate
make prod-health
make prod-logs
```

`make prod-up`은 prod compose의 external network인 `ai-health-shared`가 없으면 먼저 생성합니다. 직접 compose를 실행하는 경우에는 같은 이름의 external network를 먼저 보장해야 합니다.

직접 compose를 써야 할 때도 반드시 prod compose와 `.prod.env`를 명시합니다.

```bash
docker network inspect ai-health-shared >/dev/null 2>&1 || docker network create ai-health-shared
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml up -d
```

루트 `docker-compose.yml`은 legacy/minimal backend/AI 검증용입니다. 운영 표준으로 쓰지 않습니다.

## 10. migration 실행

운영 migration은 명시적으로만 실행합니다.

```bash
make prod-migrate
```

`prod-release-db`는 하위 호환용으로 남아 있지만 현재는 migration만 실행합니다. 운영 배포에서 seed는 자동 실행하지 않습니다.

챌린지 seed가 꼭 필요한 초기 환경에서는 운영자가 명시적으로 승인한 뒤 danger target으로만 실행합니다.

```bash
make danger-prod-seed-challenges
```

FAQ 테이블은 migration으로 생성되지만 FAQ 목록 row는 별도 seed입니다. FAQ 목록이 비어 있고 초기 FAQ seed가 필요할 때만 실행합니다.

```bash
make danger-prod-seed-faqs
```

## 11. Nginx reverse proxy 구조

prod compose의 Nginx는 외부 `80/443`을 노출합니다.

- `/api/` → `fastapi:8000`
- `/` → `frontend:80`
- `/.well-known/acme-challenge/` → certbot webroot

`infra/nginx/prod_http.conf`는 인증서 발급 전 HTTP bootstrap용입니다. `infra/nginx/prod_https.conf`는 HTTPS 운영용입니다.

`server_name`과 인증서 경로는 `healthladder.duckdns.org` 기준입니다.

## 12. HTTPS/Certbot 적용 절차

실제 certbot은 이 문서 작업 중 실행하지 않습니다. 운영 서버에서만 아래 흐름을 사용합니다.

1. DuckDNS가 EC2 public IP를 가리키는지 확인합니다.
2. 보안 그룹에서 80 포트가 열려 있는지 확인합니다.
3. 최초 인증서 발급 전에는 `NGINX_CONF=../nginx/prod_http.conf`로 HTTP Nginx를 띄웁니다.
4. `make prod-health` 또는 `curl -fsS http://localhost/api/v1/system/health`로 HTTP health를 확인합니다.
5. `/.well-known/acme-challenge/`가 Nginx에서 webroot로 제공되는지 확인합니다.
6. certbot webroot 방식으로 인증서를 발급합니다.
7. `make prod-certbot-tls-assets`로 Nginx HTTPS 보조 TLS 파일을 certbot volume에 준비합니다.
8. `NGINX_CONF=../nginx/prod_https.conf`로 전환합니다.
9. Nginx를 재시작하거나 reload합니다.
10. HTTPS health check를 확인합니다.

certbot 예시:

```bash
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml run --rm certbot \
  certonly --webroot -w /var/www/certbot \
  -d healthladder.duckdns.org \
  --email admin@example.com \
  --agree-tos \
  --no-eff-email
```

인증서 경로:

```text
/etc/letsencrypt/live/healthladder.duckdns.org/fullchain.pem
/etc/letsencrypt/live/healthladder.duckdns.org/privkey.pem
```

`scripts/certbot.sh`는 legacy interactive reference입니다. 현재 표준 배포 흐름은 `.prod.env`와 `infra/docker/docker-compose.prod.yml` 기준 문서를 따릅니다.

인증서 발급과 HTTPS 전환의 실제 운영 절차는 `docs/deployment/release_deploy_runbook.md`를 우선 확인합니다.

## 13. health check

Nginx 경유 기준:

```bash
curl -fsS https://healthladder.duckdns.org/api/v1/system/health
```

HTTP bootstrap 중:

```bash
curl -I http://healthladder.duckdns.org
curl -fsS http://healthladder.duckdns.org/api/v1/system/health
```

HTTPS 적용 후:

```bash
curl -I https://healthladder.duckdns.org
curl -fsS https://healthladder.duckdns.org/api/v1/system/health
```

Makefile:

```bash
make prod-health
```

`make prod-health`는 `.prod.env`의 `NGINX_HTTP_PORT`를 읽고, 값이 없으면 운영 기본 포트 `80`으로 `http://localhost:80/api/v1/system/health`를 확인합니다.

## 14. 배포 후 QA

브라우저에서 확인합니다.

- 로그인
- 회원가입 이메일 인증
- 건강 분석
- 정밀 분석
- 식단 이미지 분석
- 챗봇 질문
- 추천 챌린지 표시
- 대시보드 챌린지 제목
- 알림 목록
- 비밀번호 재설정 링크

이메일 알림 smoke는 실제 발송이므로 운영자가 명시적으로 허용한 경우에만 별도 QA 절차로 실행합니다.

```bash
TEST_NOTIFICATION_EMAIL="user@example.com" \
uv run python scripts/qa/smoke_notification_email.py --confirm-send
```

위 명령의 이메일 주소는 실제 운영자가 직접 지정합니다. 테스트/문서에 실제 사용자 이메일을 남기지 않습니다.

## 15. 운영 주의사항

- `.prod.env`와 secret은 Git에 올리지 않습니다.
- DuckDNS token은 절대 기록하지 않습니다.
- `docker compose config` 원문 출력은 secret 노출 위험이 있어 사용하지 않습니다.
- 필요한 경우 `config --quiet` 또는 `config --services`만 사용합니다.
- `docker compose down -v`는 DB/Redis volume을 삭제할 수 있으므로 일반 운영 절차에 넣지 않습니다.
- 운영 seed, RAG ingest, OpenAI embedding apply는 danger/cost target으로만 다룹니다.
- `8000`, `5432`, `6379`는 외부 보안 그룹에서 열지 않습니다.

안전한 compose 확인:

```bash
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml config --quiet
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml config --services
```

## 16. rollback / 로그 확인

로그:

```bash
make prod-logs
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml logs --tail=100 nginx
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml logs --tail=100 fastapi
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml logs --tail=100 ai-worker
```

Rollback 기본 방향:

1. 이전에 정상 동작하던 image tag로 `APP_VERSION`, `AI_WORKER_VERSION`, `FRONTEND_VERSION`을 되돌립니다.
2. `make prod-pull`로 이미지를 다시 받습니다.
3. `make prod-up`으로 컨테이너를 재생성합니다.
4. `make prod-health`와 HTTPS health check를 확인합니다.

DB schema migration이 포함된 배포는 rollback 전에 migration policy를 별도로 확인합니다. DB volume 삭제나 `down -v`는 rollback 방법이 아닙니다.
