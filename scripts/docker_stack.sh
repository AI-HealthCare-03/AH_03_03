#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

APP_COMPOSE="${ROOT_DIR}/docker-compose.yml"
DEV_COMPOSE="${ROOT_DIR}/infra/docker/docker-compose.dev.yml"
LANGFUSE_COMPOSE="${ROOT_DIR}/infra/langfuse/docker-compose.yml"
SHARED_NETWORK="ai-health-shared"
APP_IMAGES=(
  "ozcodingschool/ai-health:app-v1.0.0"
  "test-ai-health:latest"
)

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/docker_stack.sh app up
  ./scripts/docker_stack.sh app up-full
  ./scripts/docker_stack.sh app worker-up
  ./scripts/docker_stack.sh app build
  ./scripts/docker_stack.sh app worker-build
  ./scripts/docker_stack.sh app rebuild
  ./scripts/docker_stack.sh app clean-image
  ./scripts/docker_stack.sh app down
  ./scripts/docker_stack.sh app ps
  ./scripts/docker_stack.sh app logs
  ./scripts/docker_stack.sh app worker-logs
  ./scripts/docker_stack.sh dev up
  ./scripts/docker_stack.sh dev down
  ./scripts/docker_stack.sh dev ps
  ./scripts/docker_stack.sh langfuse up
  ./scripts/docker_stack.sh langfuse down
  ./scripts/docker_stack.sh langfuse ps
  ./scripts/docker_stack.sh langfuse logs

Stacks:
  app       Legacy/minimal root docker-compose.yml for backend/AI checks.
  dev       infra/docker/docker-compose.dev.yml. Standard full dev stack.
  langfuse  infra/langfuse/docker-compose.yml. Optional Langfuse self-host stack.
USAGE
}

ensure_shared_network() {
  if ! docker network inspect "${SHARED_NETWORK}" >/dev/null 2>&1; then
    echo "Creating Docker network: ${SHARED_NETWORK}"
    docker network create "${SHARED_NETWORK}" >/dev/null
  fi
}

run_app() {
  local action="$1"
  case "${action}" in
    up)
      docker compose -f "${APP_COMPOSE}" up -d --build postgres redis fastapi
      ;;
    up-full)
      docker compose -f "${APP_COMPOSE}" up -d --build postgres redis fastapi ai-worker
      ;;
    worker-up)
      docker compose -f "${APP_COMPOSE}" up -d --build ai-worker
      ;;
    build)
      docker compose -f "${APP_COMPOSE}" build fastapi
      ;;
    worker-build)
      docker compose -f "${APP_COMPOSE}" build ai-worker
      ;;
    rebuild)
      docker compose -f "${APP_COMPOSE}" build --no-cache fastapi
      ;;
    clean-image)
      clean_app_images
      ;;
    down)
      docker compose -f "${APP_COMPOSE}" down
      ;;
    ps)
      docker compose -f "${APP_COMPOSE}" ps
      ;;
    logs)
      docker compose -f "${APP_COMPOSE}" logs --tail=100 fastapi
      ;;
    worker-logs)
      docker compose -f "${APP_COMPOSE}" logs --tail=100 ai-worker
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

clean_app_images() {
  for image in "${APP_IMAGES[@]}"; do
    if docker image inspect "${image}" >/dev/null 2>&1; then
      echo "Removing image: ${image}"
      docker image rm "${image}"
    else
      echo "Image not found, skipping: ${image}"
    fi
  done
}

run_dev() {
  local action="$1"
  case "${action}" in
    up)
      ensure_shared_network
      docker compose -f "${DEV_COMPOSE}" up -d
      ;;
    down)
      docker compose -f "${DEV_COMPOSE}" down
      ;;
    ps)
      docker compose -f "${DEV_COMPOSE}" ps
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

run_langfuse() {
  local action="$1"
  case "${action}" in
    up)
      ensure_shared_network
      docker compose -f "${LANGFUSE_COMPOSE}" up -d
      ;;
    down)
      docker compose -f "${LANGFUSE_COMPOSE}" down
      ;;
    ps)
      docker compose -f "${LANGFUSE_COMPOSE}" ps
      ;;
    logs)
      docker compose -f "${LANGFUSE_COMPOSE}" logs --tail=100 langfuse-web langfuse-worker
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main() {
  if [[ $# -ne 2 ]]; then
    usage
    exit 1
  fi

  local stack="$1"
  local action="$2"

  case "${stack}" in
    app)
      run_app "${action}"
      ;;
    dev)
      run_dev "${action}"
      ;;
    langfuse)
      run_langfuse "${action}"
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
