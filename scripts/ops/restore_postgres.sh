#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ENV_FILE="${ENV_FILE:-.prod.env}"
COMPOSE_FILE="${COMPOSE_FILE:-infra/docker/docker-compose.prod.yml}"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 path/to/backup.sql" >&2
  echo "This restores into the current prod compose PostgreSQL database." >&2
  exit 1
fi

BACKUP_FILE="$1"
if [[ "${BACKUP_FILE}" != /* ]]; then
  BACKUP_FILE="${PROJECT_ROOT}/${BACKUP_FILE}"
fi

if [[ ! -f "${BACKUP_FILE}" ]]; then
  echo "backup file not found: ${BACKUP_FILE}" >&2
  exit 1
fi

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

if [[ ! -f "${ENV_PATH}" ]]; then
  echo "env file not found: ${ENV_PATH}" >&2
  echo "Set ENV_FILE=/path/to/.prod.env if your production env file is elsewhere." >&2
  exit 1
fi

if [[ ! -f "${COMPOSE_PATH}" ]]; then
  echo "compose file not found: ${COMPOSE_PATH}" >&2
  exit 1
fi

cat <<EOF
You are about to restore this SQL dump into the prod compose PostgreSQL database:

  ${BACKUP_FILE}

This can overwrite or conflict with existing data. Take a fresh backup first.
Type RESTORE_DATABASE to continue.
EOF

if [[ "${ASSUME_YES:-false}" != "true" ]]; then
  read -r confirmation
  if [[ "${confirmation}" != "RESTORE_DATABASE" ]]; then
    echo "Restore canceled."
    exit 1
  fi
fi

cat "${BACKUP_FILE}" | docker compose --env-file "${ENV_PATH}" -f "${COMPOSE_PATH}" exec -T postgres \
  sh -lc 'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"'

echo "Restore complete."
