#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ENV_FILE="${ENV_FILE:-.prod.env}"
COMPOSE_FILE="${COMPOSE_FILE:-infra/docker/docker-compose.prod.yml}"
BACKUP_DIR="${BACKUP_DIR:-var/backups/postgres}"

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

if [[ "${BACKUP_DIR}" = /* ]]; then
  BACKUP_PATH="${BACKUP_DIR}"
else
  BACKUP_PATH="${PROJECT_ROOT}/${BACKUP_DIR}"
fi

if [[ ! -f "${ENV_PATH}" ]]; then
  echo "env file not found: ${ENV_PATH}" >&2
  echo "Set ENV_FILE=/path/to/.prod.env if your production env file is elsewhere." >&2
  exit 1
fi

if [[ ! -f "${COMPOSE_PATH}" ]]; then
  echo "compose file not found: ${COMPOSE_PATH}" >&2
  exit 1
fi

mkdir -p "${BACKUP_PATH}"

timestamp="$(date +%Y%m%d_%H%M%S)"
backup_file="${BACKUP_PATH}/ai_health_${timestamp}.sql"

echo "Creating PostgreSQL backup from the prod compose postgres service..."
echo "Output: ${backup_file}"

docker compose --env-file "${ENV_PATH}" -f "${COMPOSE_PATH}" exec -T postgres \
  sh -lc 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  > "${backup_file}"

echo "Backup complete."
du -h "${backup_file}"
