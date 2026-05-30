#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ENV_FILE="${ENV_FILE:-.env.prod}"
COMPOSE_FILE="${COMPOSE_FILE:-infra/docker/docker-compose.prod.yml}"
LOCAL_HEALTH_URL="${LOCAL_HEALTH_URL:-http://localhost/api/v1/system/health}"
PUBLIC_BASE_URL="${PUBLIC_BASE_URL:-}"

if [[ "${ENV_FILE}" = /* ]]; then
  ENV_PATH="${ENV_FILE}"
else
  ENV_PATH="${PROJECT_ROOT}/${ENV_FILE}"
fi

if [[ "${COMPOSE_FILE}" = /* ]]; then
  COMPOSE_PATH="${COMPOSE_FILE}"
else
  COMPOSE_PATH="${PROJECT_ROOT}/${COMPOSE_FILE}"
fi

compose() {
  docker compose --env-file "${ENV_PATH}" -f "${COMPOSE_PATH}" "$@"
}

service_running() {
  local service="$1"
  compose ps --status running "${service}" | awk 'NR > 1 { found = 1 } END { exit found ? 0 : 1 }'
}

failures=0

check() {
  local label="$1"
  shift
  echo
  echo "==> ${label}"
  if "$@"; then
    echo "OK: ${label}"
  else
    echo "FAIL: ${label}" >&2
    failures=$((failures + 1))
  fi
}

if [[ ! -f "${ENV_PATH}" ]]; then
  echo "env file not found: ${ENV_PATH}" >&2
  exit 1
fi

if [[ ! -f "${COMPOSE_PATH}" ]]; then
  echo "compose file not found: ${COMPOSE_PATH}" >&2
  exit 1
fi

echo "Using env file: ${ENV_PATH}"
echo "Using compose file: ${COMPOSE_PATH}"

check "docker compose ps" compose ps
check "nginx container running" service_running nginx
check "fastapi local health endpoint" curl -fsS "${LOCAL_HEALTH_URL}"
check "ai-worker container running" service_running ai-worker
check "postgres readiness" compose exec -T postgres sh -lc 'pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
check "redis ping" compose exec -T redis redis-cli ping

if [[ -n "${PUBLIC_BASE_URL}" ]]; then
  check "public domain health endpoint" curl -fsS "${PUBLIC_BASE_URL%/}/api/v1/system/health"
else
  echo
  echo "PUBLIC_BASE_URL is not set. Skipping domain health check."
  echo "Example: PUBLIC_BASE_URL=https://your-domain.example $0"
fi

cat <<EOF

Useful log commands:
  docker compose --env-file ${ENV_FILE} -f ${COMPOSE_FILE} logs --tail=100 fastapi
  docker compose --env-file ${ENV_FILE} -f ${COMPOSE_FILE} logs --tail=100 ai-worker
  docker compose --env-file ${ENV_FILE} -f ${COMPOSE_FILE} logs --tail=100 nginx
  docker compose --env-file ${ENV_FILE} -f ${COMPOSE_FILE} logs --tail=100 postgres
  docker compose --env-file ${ENV_FILE} -f ${COMPOSE_FILE} logs --tail=100 redis

EOF

if [[ "${failures}" -gt 0 ]]; then
  echo "${failures} health check(s) failed." >&2
  exit 1
fi

echo "All health checks passed."
