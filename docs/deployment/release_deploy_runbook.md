# Health Ladder 운영 배포 Runbook

이 문서는 `main` 브랜치 기준 운영 자동배포와, 장애 대응 시 사용할 수 있는 EC2 수동 배포 체크리스트입니다. 실제 secret 값은 이 문서, PR, issue, 채팅, 스크린샷, 배포 로그에 남기지 않습니다.

## 1. 개요

- 운영 배포 기준 브랜치: `main`
- 기본 배포 방식: GitHub Actions `deploy-prod` workflow
- 운영 env 파일: `.prod.env`
- prod compose: `infra/docker/docker-compose.prod.yml`
- 운영 도메인: `healthladder.duckdns.org`
- 기본 흐름: HTTP bootstrap으로 먼저 기동과 health를 확인한 뒤 certbot/HTTPS로 전환
- 최종 HTTPS 기준: `.prod.env`의 `NGINX_CONF=../nginx/prod_https.conf`

Docker image tag 예시:

```bash
kdu0312/ai-health:app-v1.0.0
kdu0312/ai-health:ai-v1.0.0
kdu0312/ai-health:frontend-v1.0.0
```

운영 compose가 pull하는 image version은 `.prod.env`의 아래 값으로 결정합니다.

```env
APP_VERSION=v1.0.1
AI_WORKER_VERSION=v1.0.1
FRONTEND_VERSION=v1.0.1
```

같은 tag(`v1.0.0` 등)를 반복해서 재사용하면 EC2의 `main` 브랜치는 최신이어도 Docker Hub의 frontend image가 예전 build일 수 있습니다. GitHub Actions 자동배포는 `main-<short-sha>` tag를 매번 새로 만들고, EC2 `.prod.env`의 `APP_VERSION`, `AI_WORKER_VERSION`, `FRONTEND_VERSION`만 같은 tag로 갱신합니다.

최종 운영 확인 결과 기준:

```bash
curl -fsS http://healthladder.duckdns.org/api/v1/system/health
curl -fsS https://healthladder.duckdns.org/api/v1/system/health
curl -I https://healthladder.duckdns.org
```

성공 기준:

- health 응답의 `status`가 `ok`
- HTTPS 응답이 `HTTP/2 200`
- `strict-transport-security` 헤더 확인

## 2. GitHub Actions 운영 자동배포

운영 배포는 `main` push를 기준으로 합니다.

필요한 GitHub Secrets:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
- `EC2_HOST`
- `EC2_USER`
- `EC2_SSH_KEY`

자동배포 workflow:

```text
push to main
-> checkout
-> uv/Python 준비
-> ruff format check, ruff check, pytest
-> frontend build
-> docker buildx 준비
-> Docker Hub login
-> app/ai-worker/frontend image build & push
-> EC2 SSH 접속
-> git fetch origin && git reset --hard origin/main
-> .prod.env의 APP_VERSION, AI_WORKER_VERSION, FRONTEND_VERSION만 새 tag로 갱신
-> make prod-pull
-> make prod-up
-> make prod-migrate
-> make prod-health
-> curl -fsS https://healthladder.duckdns.org/api/v1/system/health
```

자동 생성되는 image tag 예:

```text
main-<short-sha>
```

Docker Hub image 예:

```text
kdu0312/ai-health:app-main-<short-sha>
kdu0312/ai-health:ai-main-<short-sha>
kdu0312/ai-health:frontend-main-<short-sha>
```

workflow는 `.prod.env`의 실제 secret 값을 생성하거나 덮어쓰지 않습니다. 운영 서버에 이미 존재하는 `.prod.env`에서 image version key 세 개만 수정합니다.

실패 시에도 `docker compose down -v`나 volume 삭제 명령은 실행하지 않습니다. `make prod-up`은 새 container 재생성을 시도하며, pull/build/migration/health 중 실패하면 workflow가 실패합니다.

## 3. 로컬 배포 전 검증

`main` push 전에 로컬에서 확인합니다.

```bash
uv run ruff format app scripts ai_runtime tests --check
uv run ruff check app scripts ai_runtime tests
uv run pytest -q
cd frontend && npm run build && cd ..
git diff --check
docker compose --env-file envs/example.prod.env -f infra/docker/docker-compose.prod.yml config --quiet
```

`main` 브랜치 상태 확인과 push:

```bash
git status -sb
git log --oneline -5
git push origin main
```

아래 수동 image build/push는 GitHub Actions를 우회해야 하는 경우에만 사용합니다. 일반 운영 배포는 `main` push 자동배포를 사용합니다. 수동 build/push를 할 때도 세 image를 같은 배포 tag로 맞춥니다.
Makefile image target은 `.prod.env` 전체를 자동으로 source하지 않습니다. secret 노출을 피하기 위해 build/push에 필요한 version 값만 shell에서 export하거나 command line 변수로 넘깁니다.

```bash
export APP_VERSION=v1.0.1
export AI_WORKER_VERSION=v1.0.1
export FRONTEND_VERSION=v1.0.1
make image-tags
make image-build-check
make image-check-ai-ocr-import
make image-push
```

날짜 tag를 쓸 때도 세 값을 같은 배포 단위로 맞춥니다.

```bash
export APP_VERSION=20260616-1
export AI_WORKER_VERSION=20260616-1
export FRONTEND_VERSION=20260616-1
make image-tags
make image-build-push
```

`make image-check-ai-ocr-import`는 build된 ai-worker image에서 `cv2`와 `paddleocr` import를 확인합니다. `make image-push`와 `make image-build-push`는 app, ai-worker, frontend 세 image를 모두 대상으로 합니다. Docker Hub token/password는 shell history나 문서에 남기지 않습니다.

## 4. EC2 반영 절차

이 절차는 자동배포 장애 대응이나 수동 점검용입니다. 운영 서버의 실제 `.prod.env`는 Git에 없으므로 먼저 백업합니다.

```bash
cp .prod.env ~/.prod.env.backup.$(date +%Y%m%d_%H%M%S)
```

`main` 최신화:

```bash
git fetch origin
git reset --hard origin/main
ls -al .prod.env
```

compose 유효성 확인:

```bash
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml config --quiet
```

새 image tag를 배포하는 경우 `.prod.env`에서 image version만 새 tag로 맞춥니다. 값 자체는 운영 서버의 `.prod.env`에만 저장하고 Git에 커밋하지 않습니다.

```env
APP_VERSION=v1.0.1
AI_WORKER_VERSION=v1.0.1
FRONTEND_VERSION=v1.0.1
```

MVP 운영 storage는 local 기준으로 맞춥니다. OCR/검진 파일 업로드가 S3 설정 누락 때문에 500으로 실패하지 않도록 아래 값을 확인합니다.

```env
STORAGE_BACKEND=local
UPLOAD_STORAGE_DIR=var/uploads
LOCAL_STORAGE_ROOT=var/storage
S3_BUCKET_NAME=
```

local storage 사용 시 `S3_BUCKET_NAME`은 비어 있어도 됩니다. S3로 전환할 때만 `STORAGE_BACKEND=s3`와 private bucket/IAM 권한을 준비합니다.

식단 이미지 분석은 운영에서 실제 provider가 준비되어야 합니다. 아래 조합처럼 GPT Vision이 꺼져 있고 demo fallback도 꺼져 있으면 `/api/v1/diets/analyze`는 더미 결과를 저장하지 않고 service unavailable로 종료됩니다.

```env
DIET_VISION_PROVIDER=rule_based
DIET_GPT_VISION_ENABLED=false
DIET_DEMO_FALLBACK_ENABLED=false
```

운영에서 GPT Vision 식단 분석을 사용하려면 `.prod.env`에 아래 값을 준비합니다. 실제 key 값은 문서나 채팅에 남기지 않습니다.

```env
OPENAI_API_KEY=<SET_IN_PROD>
DIET_VISION_PROVIDER=gpt_vision
DIET_GPT_VISION_ENABLED=true
DIET_GPT_VISION_MODEL=gpt-4o
GPT_VISION_FALLBACK_ENABLED=true
DIET_DEMO_FALLBACK_ENABLED=false
```

`DIET_DEMO_FALLBACK_ENABLED=true`는 시연/개발용 더미 fallback을 명시적으로 허용하는 설정입니다. 운영 기본값은 false로 유지합니다.

건강검진 OCR도 실제 provider가 준비되어야 합니다. 운영 기본값이 아래처럼 provider 미설정 상태라면 더미 검진값을 저장하지 않고 OCR job이 실패합니다.

```env
EXAM_OCR_PROVIDER=fallback
EXAM_GPT_VISION_ENABLED=false
PADDLE_OCR_ENABLED=false
GPT_VISION_FALLBACK_ENABLED=false
```

운영에서 GPT Vision 건강검진 OCR을 사용할 때는 `.prod.env`에 아래 값을 준비합니다. 실제 key 값은 문서나 채팅에 남기지 않습니다.

```env
OPENAI_API_KEY=<SET_IN_PROD>
EXAM_OCR_PROVIDER=gpt_vision
EXAM_GPT_VISION_ENABLED=true
EXAM_GPT_VISION_MODEL=gpt-4o
GPT_VISION_FALLBACK_ENABLED=true
```

PaddleOCR을 건강검진 OCR 주기능으로 쓰고 GPT Vision을 fallback으로 둘 때는 ai-worker image가 OCR dependency를 포함한 새 tag로 배포되어 있어야 합니다. 배포 후 아래 import smoke가 통과하는지 먼저 확인합니다.

```bash
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml exec -T ai-worker \
  uv run --no-sync python -c "import cv2; import paddleocr; print('ocr imports ok')"
```

import가 통과하면 운영자가 `.prod.env`에서 아래 OCR provider 조합으로 전환할 수 있습니다.

```env
EXAM_OCR_PROVIDER=auto
PADDLE_OCR_ENABLED=true
EXAM_GPT_VISION_ENABLED=true
GPT_VISION_FALLBACK_ENABLED=true
```

`EXAM_OCR_PROVIDER`는 현재 `fallback`, `auto`, `paddleocr`, `gpt_vision` 값을 사용합니다. `fallback`은 운영 안전 기본값이며 OCR provider 후보가 없다는 뜻입니다. 이 상태에서는 `/api/v1/exams/{exam_id}/ocr` job이 성공처럼 보이지 않아야 하고, `exam_reports.ocr_status=FAILED` 및 `exam_measurements` 빈 결과로 확인됩니다.

검진표 OCR 측정값 확인 시 실제 FK 컬럼명은 `exam_report_id`입니다.

```sql
select id, exam_report_id, measurement_key, value, unit, created_at
from exam_measurements
where exam_report_id = <EXAM_REPORT_ID>
order by id;
```

## 5. Docker 권한 확인

현재 계정의 Docker 권한을 확인합니다.

```bash
groups
ls -l /var/run/docker.sock
docker ps
```

`ubuntu` 사용자가 `docker` 그룹에 없다면:

```bash
sudo usermod -aG docker ubuntu
newgrp docker
```

권한 반영에는 SSH 재접속이 필요할 수 있습니다.

compose 상태 확인:

```bash
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml ps
```

## 6. 운영 기동

`make prod-up`은 `ai-health-shared` external network가 없으면 먼저 생성합니다.

```bash
make prod-pull
make prod-up
make prod-migrate
```

`make prod-pull`은 `infra/docker/docker-compose.prod.yml`에 정의된 service image를 pull합니다. 이때 `.prod.env`의 `APP_VERSION`, `AI_WORKER_VERSION`, `FRONTEND_VERSION` 값이 Docker Hub에 push된 tag와 일치해야 합니다.

`make prod-migrate`는 FAQ 테이블을 만들지만 FAQ row를 넣지는 않습니다. FAQ 목록 데이터는 migration이 아니라 별도 seed입니다.

최초 운영 DB에 챌린지 seed가 꼭 필요하고 운영자가 명시적으로 승인한 경우에만 실행합니다.

```bash
make danger-prod-seed-challenges
```

챌린지 seed 확인은 실제 컬럼 기준으로 집계합니다. `challenges` 테이블에는 `is_active` 컬럼이 없으므로 `status` 기준으로 확인합니다.

```bash
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml exec -T postgres \
  sh -lc 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -P pager=off -c "
select status, count(*)
from challenges
group by status
order by status;
"'
```

FAQ 목록이 비어 있고 운영자가 초기 FAQ seed를 승인한 경우에만 실행합니다.

```bash
make danger-prod-seed-faqs
```

health 확인:

```bash
make prod-health
curl -fsS http://localhost/api/v1/system/health
curl -fsS http://healthladder.duckdns.org/api/v1/system/health
```

frontend image가 최신 build인지 asset hash를 확인합니다.

```bash
docker inspect frontend --format 'IMAGE={{.Config.Image}} CREATED={{.Created}}'
docker exec frontend ls -al /usr/share/nginx/html/assets | grep -E 'index-|ExamOcrPage|AdminDashboard|AdminFaq' || true
```

로컬 최신 build의 `frontend/dist/assets` hash와 EC2 frontend 컨테이너의 hash가 다르면, EC2 코드가 최신이어도 frontend Docker image tag가 예전 build를 가리키는 상태입니다. 예전 admin asset(`AdminDashboard`, `AdminFaq`)이 남아 있으면 stale frontend image를 우선 의심합니다. 이 경우 새 frontend tag를 build/push하고 `.prod.env`의 `FRONTEND_VERSION`을 새 tag로 바꾼 뒤 다시 pull/up 합니다.

## 7. HTTP Bootstrap

최초 HTTPS 인증서 발급 전에는 `.prod.env`에서 HTTP Nginx 설정을 사용합니다.

```env
NGINX_CONF=../nginx/prod_http.conf
```

Nginx 재생성:

```bash
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml up -d --force-recreate nginx
```

HTTP health가 성공한 뒤 certbot 발급과 HTTPS 전환으로 넘어갑니다.

## 8. DuckDNS 확인

DuckDNS가 EC2 public IP를 가리키는지 확인합니다.

```bash
nslookup healthladder.duckdns.org
dig +short healthladder.duckdns.org
```

기대값은 현재 EC2 public IP입니다. 다르면 DuckDNS의 current ip를 EC2 public IP로 갱신합니다. DuckDNS token은 secret이므로 문서, shell history, 채팅, issue에 남기지 않습니다.

## 9. ACME Challenge 확인

certbot 발급 전에 HTTP bootstrap Nginx가 ACME challenge 경로를 외부에 노출하는지 확인합니다.

Nginx 컨테이너에서 challenge 파일 생성:

```bash
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml exec -T nginx \
  sh -lc 'mkdir -p /var/www/certbot/.well-known/acme-challenge && echo ok > /var/www/certbot/.well-known/acme-challenge/ping'
```

외부 확인:

```bash
curl -fsS http://healthladder.duckdns.org/.well-known/acme-challenge/ping
```

기대값:

```text
ok
```

## 10. Certbot 발급

`certbot` compose service의 기본 command가 renew loop일 수 있으므로, 신규 발급은 `--entrypoint certbot`으로 override합니다. 이메일은 운영자가 소유한 주소를 넣되 문서에는 실제 주소를 남기지 않습니다.

dry-run:

```bash
timeout 180s docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml run --rm --no-deps --entrypoint certbot certbot \
  certonly --webroot \
  --webroot-path=/var/www/certbot \
  -d healthladder.duckdns.org \
  --email <EMAIL> \
  --agree-tos \
  --no-eff-email \
  --non-interactive \
  --dry-run \
  -v
```

dry-run 성공 기준:

- `The dry run was successful.`
- `EXIT_CODE=0`

실제 발급:

```bash
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml run --rm --no-deps --entrypoint certbot certbot \
  certonly --webroot \
  --webroot-path=/var/www/certbot \
  -d healthladder.duckdns.org \
  --email <EMAIL> \
  --agree-tos \
  --no-eff-email \
  --non-interactive
```

발급 성공 기준:

- `Successfully received certificate.`
- `/etc/letsencrypt/live/healthladder.duckdns.org/fullchain.pem`
- `/etc/letsencrypt/live/healthladder.duckdns.org/privkey.pem`

`No renewals were attempted`만 출력되면 신규 발급 command가 아니라 renew loop가 실행된 것일 수 있습니다. 이 경우 위처럼 `--entrypoint certbot`을 사용했는지 확인합니다.

## 11. Certbot 보조 TLS 파일

certbot 인증서 파일(`fullchain.pem`, `privkey.pem`)은 Git에 넣지 않습니다. 인증서와 보조 TLS 파일은 Docker named volume인 `certbot-conf` 안에서만 관리합니다.

인증서는 발급됐지만 Nginx HTTPS 설정에서 아래 파일이 없어 죽을 수 있습니다.

- `/etc/letsencrypt/options-ssl-nginx.conf`
- `/etc/letsencrypt/ssl-dhparams.pem`

장애 로그 예:

```text
open() "/etc/letsencrypt/options-ssl-nginx.conf" failed
```

`options-ssl-nginx.conf`, `ssl-dhparams.pem`은 Nginx HTTPS 설정에 필요한 보조 파일입니다. 새 EC2, `certbot-conf` volume 삭제, `docker compose down -v` 이후에는 다시 생성해야 합니다.

certbot Docker volume 안에 보조 TLS 파일을 준비합니다. 이 target은 인증서 파일을 생성하거나 복사하지 않습니다.

```bash
make prod-certbot-tls-assets
```

파일 확인:

```bash
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml run --rm --no-deps --entrypoint sh certbot \
  -c 'ls -l /etc/letsencrypt/options-ssl-nginx.conf /etc/letsencrypt/ssl-dhparams.pem'
```

## 12. HTTPS 전환

HTTPS 전환 전 순서:

1. HTTP bootstrap 성공
2. ACME challenge path 확인
3. certbot dry-run 성공
4. certbot 실제 인증서 발급 성공
5. `make prod-certbot-tls-assets`
6. `.prod.env`에서 `NGINX_CONF=../nginx/prod_https.conf`
7. Nginx 재생성
8. HTTPS health 확인

인증서와 보조 TLS 파일이 준비되면 `.prod.env`를 HTTPS 설정으로 바꿉니다.

```env
NGINX_CONF=../nginx/prod_https.conf
```

Nginx 재생성:

```bash
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml up -d --force-recreate nginx
```

HTTPS health:

```bash
curl -fsS https://healthladder.duckdns.org/api/v1/system/health
curl -I https://healthladder.duckdns.org
```

성공 기준:

- health 응답의 `status`가 `ok`
- `HTTP/2 200`
- `strict-transport-security` 헤더 확인

## 13. 자주 발생한 장애 대응

### EC2 main은 최신인데 화면이 예전 버전으로 보임

Git branch 최신화는 compose 파일과 문서만 갱신합니다. 브라우저에 보이는 정적 파일은 `frontend` Docker image 안에 들어 있으므로 Docker Hub의 `frontend-${FRONTEND_VERSION}` image가 최신 `main` build여야 합니다.

확인:

```bash
docker inspect frontend --format 'IMAGE={{.Config.Image}} CREATED={{.Created}}'
docker exec frontend ls -al /usr/share/nginx/html/assets | grep -E 'index-|ExamOcrPage|AdminDashboard|AdminFaq' || true
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml images
```

대응:

1. `main`에서 새 `FRONTEND_VERSION` tag로 frontend를 포함한 세 image를 build/push합니다.
2. EC2 `.prod.env`의 `APP_VERSION`, `AI_WORKER_VERSION`, `FRONTEND_VERSION`을 새 tag로 맞춥니다.
3. EC2에서 image를 다시 pull하고 컨테이너를 재생성합니다.

```bash
make prod-pull
make prod-up
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml up -d --force-recreate frontend nginx
```

### OCR 또는 검진 업로드가 500으로 실패

MVP 운영 기준은 local storage입니다. S3를 쓰지 않는 상태에서 `STORAGE_BACKEND=s3`이거나 `S3_BUCKET_NAME`이 필요한 경로로 동작하면 OCR/업로드가 500으로 실패할 수 있습니다.

운영 `.prod.env`에서 아래 값을 확인합니다. 실제 secret 값은 출력하지 않습니다.

```env
STORAGE_BACKEND=local
UPLOAD_STORAGE_DIR=var/uploads
LOCAL_STORAGE_ROOT=var/storage
S3_BUCKET_NAME=
```

local storage 사용 시 `S3_BUCKET_NAME`은 비어 있어도 됩니다.

### OCR job은 끝났지만 검진 결과가 비어 있음

`exam_reports.ocr_status=FAILED`이고 `exam_measurements`가 비어 있으면 provider 설정 또는 업로드 파일 문제입니다. `EXAM_OCR_PROVIDER=fallback`, `EXAM_GPT_VISION_ENABLED=false`, `PADDLE_OCR_ENABLED=false` 조합은 더미 결과를 만들지 않는 안전 기본값입니다.

GPT Vision OCR을 운영에서 사용하려면 `OPENAI_API_KEY`, `EXAM_OCR_PROVIDER=gpt_vision`, `EXAM_GPT_VISION_ENABLED=true`, `EXAM_GPT_VISION_MODEL`, `GPT_VISION_FALLBACK_ENABLED=true`를 확인합니다. 값 자체는 출력하지 않습니다.

PaddleOCR을 운영 주기능으로 쓰려면 ai-worker image에 `cv2`와 `paddleocr`가 설치되어 있어야 합니다. 새 worker image tag 배포 후 아래 smoke가 실패하면 `.prod.env`만 켜지 말고 worker image를 다시 build/push/deploy합니다.

```bash
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml exec -T ai-worker \
  uv run --no-sync python -c "import cv2; import paddleocr; print('ocr imports ok')"
```

PaddleOCR 주기능 + GPT Vision fallback 조합은 `EXAM_OCR_PROVIDER=auto`, `PADDLE_OCR_ENABLED=true`, `EXAM_GPT_VISION_ENABLED=true`, `GPT_VISION_FALLBACK_ENABLED=true`입니다.

측정값 조회 SQL은 `report_id`가 아니라 `exam_report_id`를 사용합니다.

```sql
select id, exam_report_id, measurement_key, value, unit, created_at
from exam_measurements
where exam_report_id = <EXAM_REPORT_ID>
order by id;
```

### SECRET_KEY must be set to a strong non-default value in production

`.prod.env`의 `SECRET_KEY`가 기본값이거나 너무 약한 값이면 발생합니다. 운영용 strong secret을 새로 생성해 `.prod.env`에만 반영합니다.

### COOKIE_DOMAIN must not point to localhost in production

production에서 `COOKIE_DOMAIN=localhost`가 적용되면 FastAPI 또는 ai-worker가 기동하지 않습니다. `fastapi`와 `ai-worker` 모두 `COOKIE_DOMAIN`, `CORS_ALLOW_ORIGINS`, `REFRESH_TOKEN_COOKIE_*` 환경변수를 받아야 합니다.

### email_service: misconfigured

SMTP 설정 누락 여부를 확인합니다.

- `SMTP_HOST`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_FROM_EMAIL`

값 자체는 출력하지 않습니다.

### Aerich old format of migration file detected

`make prod-migrate`에서 아래 오류가 발생하면 migration 파일의 `MODELS_STATE` 포맷이 현재 Aerich 버전과 맞지 않는 상태입니다.

```text
Old format of migration file detected, run `aerich fix-migrations` to upgrade format
```

운영 DB나 volume을 삭제하지 않습니다. 서버에서 임의로 migration 파일을 고치거나 `down -v`를 실행하지 말고, 로컬 코드에서 해당 migration 파일 포맷을 수정한 뒤 새 image로 재배포합니다.

로컬 확인 절차:

```bash
uv run ruff check app scripts ai_runtime tests
uv run pytest -q tests/exams/test_exam_confirm_to_health_record.py tests/jobs/test_async_job_skeleton.py
uv run aerich history
git diff --check
```

수정이 main에 반영되면 GitHub Actions 자동배포 또는 수동 image build/push 후 EC2에서 다시 실행합니다.

```bash
make prod-pull
make prod-up
make prod-migrate
make prod-health
```

### network ai-health-shared declared as external, but could not be found

`make prod-up`은 `prod-network`를 먼저 실행해 network를 보장합니다. 수동 compose 명령을 쓰는 경우에는 아래를 먼저 실행합니다.

```bash
docker network inspect ai-health-shared >/dev/null 2>&1 || docker network create ai-health-shared
```

### fastapi 재생성 후 외부 요청 502

Nginx가 이전 upstream IP를 들고 있을 수 있습니다. Nginx를 재생성합니다.

```bash
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml up -d --force-recreate nginx
```

### nginx가 HTTPS config에서 죽는 경우

certbot 인증서 파일 또는 보조 TLS 파일이 없는 상태에서 `prod_https.conf`를 쓰고 있을 수 있습니다. 최초 인증서 발급 전에는 `.prod.env`에서 아래 값을 사용합니다.

```env
NGINX_CONF=../nginx/prod_http.conf
```

인증서 발급 후에도 Nginx가 죽으면 다음 파일이 `certbot-conf` volume 안에 있는지 확인합니다.

- `options-ssl-nginx.conf`
- `ssl-dhparams.pem`

### certbot run이 No renewals were attempted만 출력

compose service의 기본 command가 renew loop라서 그럴 수 있습니다. 신규 발급은 `docker compose run ... --entrypoint certbot certbot certonly ...` 형태로 실행합니다.

## 14. HTTPS 전환 전 체크

- [ ] HTTP health OK
- [ ] DuckDNS IP OK
- [ ] 80 포트 외부 접근 OK
- [ ] ACME challenge URL이 `ok` 반환
- [ ] certbot volume 준비
- [ ] `fullchain.pem`, `privkey.pem` 발급 확인
- [ ] `options-ssl-nginx.conf`, `ssl-dhparams.pem` 준비
- [ ] `make prod-certbot-tls-assets` 실행
- [ ] 인증서 발급 후 `.prod.env`에서 `NGINX_CONF=../nginx/prod_https.conf` 전환

전환 후 확인:

```bash
curl -I https://healthladder.duckdns.org
curl -fsS https://healthladder.duckdns.org/api/v1/system/health
```

## 15. 보안 주의

- `.prod.env` 커밋 금지
- 실제 API key, SMTP password, DuckDNS token, DB password, SSH private key 문서화 금지
- 채팅/스크린샷/터미널에 노출된 SMTP key, DuckDNS token, SSH private key, OpenAI key 등은 운영 안정화 후 재발급 또는 rotation 필요
- SSH private key는 shell로 실행하지 않습니다.
- `sh -i ~/.ssh/healthladder.pem ...` 형태로 실행하지 않습니다.

올바른 SSH 접속 예:

```bash
ssh -i ~/.ssh/healthladder.pem ubuntu@<EC2_IP>
```
