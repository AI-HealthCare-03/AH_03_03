# Health Ladder 운영 배포 Runbook

이 문서는 release 브랜치 기준 운영 배포자가 EC2에서 수동 배포할 때 따르는 체크리스트입니다. 실제 secret 값은 이 문서, PR, issue, 채팅, 스크린샷, 배포 로그에 남기지 않습니다.

## 1. 개요

- 대상 브랜치: `release`
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

## 2. 로컬 배포 전 검증

release 브랜치 push 전에 로컬에서 확인합니다.

```bash
uv run ruff format app scripts ai_runtime tests --check
uv run ruff check app scripts ai_runtime tests
uv run pytest -q
cd frontend && npm run build && cd ..
git diff --check
docker compose --env-file envs/example.prod.env -f infra/docker/docker-compose.prod.yml config --quiet
```

release 브랜치 상태 확인과 push:

```bash
git status -sb
git log --oneline -5
git push origin release
```

## 3. EC2 반영 절차

운영 서버의 실제 `.prod.env`는 Git에 없으므로 먼저 백업합니다.

```bash
cp .prod.env ~/.prod.env.backup.$(date +%Y%m%d_%H%M%S)
```

release 최신화:

```bash
git fetch origin
git reset --hard origin/release
ls -al .prod.env
```

compose 유효성 확인:

```bash
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml config --quiet
```

## 4. Docker 권한 확인

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

## 5. 운영 기동

`make prod-up`은 `ai-health-shared` external network가 없으면 먼저 생성합니다.

```bash
make prod-up
make prod-migrate
```

최초 운영 DB에 챌린지 seed가 꼭 필요하고 운영자가 명시적으로 승인한 경우에만 실행합니다.

```bash
make danger-prod-seed
```

health 확인:

```bash
make prod-health
curl -fsS http://localhost/api/v1/system/health
curl -fsS http://healthladder.duckdns.org/api/v1/system/health
```

## 6. HTTP Bootstrap

최초 HTTPS 인증서 발급 전에는 `.prod.env`에서 HTTP Nginx 설정을 사용합니다.

```env
NGINX_CONF=../nginx/prod_http.conf
```

Nginx 재생성:

```bash
docker compose --env-file .prod.env -f infra/docker/docker-compose.prod.yml up -d --force-recreate nginx
```

HTTP health가 성공한 뒤 certbot 발급과 HTTPS 전환으로 넘어갑니다.

## 7. DuckDNS 확인

DuckDNS가 EC2 public IP를 가리키는지 확인합니다.

```bash
nslookup healthladder.duckdns.org
dig +short healthladder.duckdns.org
```

기대값은 현재 EC2 public IP입니다. 다르면 DuckDNS의 current ip를 EC2 public IP로 갱신합니다. DuckDNS token은 secret이므로 문서, shell history, 채팅, issue에 남기지 않습니다.

## 8. ACME Challenge 확인

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

## 9. Certbot 발급

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

## 10. Certbot 보조 TLS 파일

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

## 11. HTTPS 전환

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

## 12. 자주 발생한 장애 대응

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

## 13. HTTPS 전환 전 체크

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

## 14. 보안 주의

- `.prod.env` 커밋 금지
- 실제 API key, SMTP password, DuckDNS token, DB password, SSH private key 문서화 금지
- 채팅/스크린샷/터미널에 노출된 SMTP key, DuckDNS token, SSH private key, OpenAI key 등은 운영 안정화 후 재발급 또는 rotation 필요
- SSH private key는 shell로 실행하지 않습니다.
- `sh -i ~/.ssh/healthladder.pem ...` 형태로 실행하지 않습니다.

올바른 SSH 접속 예:

```bash
ssh -i ~/.ssh/healthladder.pem ubuntu@<EC2_IP>
```
