DEV_ENV ?= .env
PROD_ENV ?= prod.env
DEV_COMPOSE = docker compose --env-file $(DEV_ENV) -f infra/docker/docker-compose.dev.yml
PROD_COMPOSE = docker compose --env-file $(PROD_ENV) -f infra/docker/docker-compose.prod.yml
COMPOSE_DEV = $(DEV_COMPOSE)
DOCKER_USER ?= kdu0312
DOCKER_REPOSITORY ?= ai-health
APP_VERSION ?= v1.0.0
AI_WORKER_VERSION ?= v1.0.0
FRONTEND_VERSION ?= v1.0.0
IMAGE_REPO = $(DOCKER_USER)/$(DOCKER_REPOSITORY)
APP_IMAGE = $(IMAGE_REPO):app-$(APP_VERSION)
AI_WORKER_IMAGE = $(IMAGE_REPO):ai-$(AI_WORKER_VERSION)
FRONTEND_IMAGE = $(IMAGE_REPO):frontend-$(FRONTEND_VERSION)
DOCKER_PLATFORM ?= linux/amd64
VITE_API_BASE_URL ?= /api/v1

# Legacy/minimal app stack
# Root docker-compose.yml wrapper for backend/AI checks only.
# Standard local dev/demo execution should use dev-* or demo-* targets below.
.PHONY: app-up app-up-full app-worker-up app-build app-worker-build app-rebuild app-clean-image app-down app-ps app-logs app-worker-logs
app-up:
	./scripts/docker_stack.sh app up

app-up-full:
	./scripts/docker_stack.sh app up-full

app-worker-up:
	./scripts/docker_stack.sh app worker-up

app-build:
	./scripts/docker_stack.sh app build

app-worker-build:
	./scripts/docker_stack.sh app worker-build

app-rebuild:
	./scripts/docker_stack.sh app rebuild

app-clean-image:
	./scripts/docker_stack.sh app clean-image

app-down:
	./scripts/docker_stack.sh app down

app-ps:
	./scripts/docker_stack.sh app ps

app-logs:
	./scripts/docker_stack.sh app logs

app-worker-logs:
	./scripts/docker_stack.sh app worker-logs

# Standard dev/demo stack
# Full stack via infra/docker/docker-compose.dev.yml: postgres, redis, fastapi, ai-worker, frontend, nginx.
.PHONY: dev-network dev-up dev-stack dev-front dev-local dev-down dev-ps dev-logs dev-migrate dev-seed dev-health dev-rebuild-api dev-rebuild-frontend dev-rebuild-all dev-restart-nginx dev-config-check
dev-network:
	docker network inspect ai-health-shared >/dev/null 2>&1 || docker network create ai-health-shared >/dev/null

dev-up: dev-network
	$(DEV_COMPOSE) up -d --build

dev-stack: dev-network
	@echo "Starting Docker backend stack for local Vite frontend development..."
	@echo "Backend stack: http://localhost:8080"
	@echo "Frontend dev server: http://localhost:5173"
	@echo "Note: nginx depends on the frontend service in compose, so Docker may start it as a dependency."
	$(DEV_COMPOSE) up -d postgres redis fastapi ai-worker nginx
	curl -fsS http://localhost:8080/api/v1/system/health

dev-front:
	@echo "Starting local Vite frontend dev server..."
	@echo "Frontend dev server: http://localhost:5173"
	@echo "API proxy target: http://localhost:8080"
	cd frontend && npm run dev

dev-local: dev-stack
	@echo "Docker backend stack is ready."
	@echo "Open the frontend at http://localhost:5173"
	cd frontend && npm run dev

dev-down:
	$(DEV_COMPOSE) down --remove-orphans

dev-ps:
	$(DEV_COMPOSE) ps

dev-logs:
	$(DEV_COMPOSE) logs --tail=100 fastapi ai-worker frontend nginx

dev-migrate:
	$(DEV_COMPOSE) exec fastapi uv run --no-sync aerich upgrade

dev-seed:
	$(DEV_COMPOSE) exec fastapi uv run --no-sync python scripts/seed_mvp_challenges.py

dev-health:
	curl -fsS http://localhost:8080/api/v1/system/health

dev-rebuild-api:
	$(DEV_COMPOSE) up -d --build fastapi ai-worker
	$(DEV_COMPOSE) up -d --force-recreate nginx
	sleep 5
	curl -fsS http://localhost:8080/api/v1/system/health

dev-rebuild-frontend:
	$(DEV_COMPOSE) up -d --build frontend
	$(DEV_COMPOSE) up -d --force-recreate nginx
	sleep 3
	curl -fsS http://localhost:8080/api/v1/system/health

dev-rebuild-all:
	$(DEV_COMPOSE) up -d --build
	$(DEV_COMPOSE) up -d --force-recreate nginx
	sleep 5
	curl -fsS http://localhost:8080/api/v1/system/health

dev-restart-nginx:
	$(DEV_COMPOSE) up -d --force-recreate nginx
	sleep 3
	curl -fsS http://localhost:8080/api/v1/system/health

dev-config-check:
	$(DEV_COMPOSE) config --quiet

.PHONY: demo-up demo-down demo-ps demo-logs demo-health
demo-up: dev-up

demo-down: dev-down

demo-ps: dev-ps

demo-logs: dev-logs

demo-health: dev-health

# Langfuse stack
# Optional self-hosted observability stack via infra/langfuse/docker-compose.yml.
.PHONY: langfuse-up langfuse-down langfuse-ps langfuse-logs
langfuse-up:
	./scripts/docker_stack.sh langfuse up

langfuse-down:
	./scripts/docker_stack.sh langfuse down

langfuse-ps:
	./scripts/docker_stack.sh langfuse ps

langfuse-logs:
	./scripts/docker_stack.sh langfuse logs

# Prod compose convenience
# Uses infra/docker/docker-compose.prod.yml and pulls prebuilt registry images.
.PHONY: prod-pull prod-up prod-ps prod-logs prod-migrate prod-seed prod-health prod-release-db
prod-pull:
	$(PROD_COMPOSE) pull

prod-up:
	$(PROD_COMPOSE) up -d

prod-ps:
	$(PROD_COMPOSE) ps

prod-logs:
	$(PROD_COMPOSE) logs -f

prod-migrate:
	$(PROD_COMPOSE) exec -T fastapi uv run --no-sync aerich upgrade

prod-seed:
	$(PROD_COMPOSE) exec -T fastapi uv run --no-sync python scripts/seed_mvp_challenges.py

prod-health:
	curl -fsS http://localhost:$${NGINX_HTTP_PORT:-8080}/api/v1/system/health

prod-release-db: prod-migrate prod-seed

# Release image build
# prod compose is pull-only; these targets verify the images that will be pushed separately.
.PHONY: image-build-check image-build-native-check image-build-app image-build-ai image-build-frontend image-buildx-amd64-check image-tags
# Deployment build check alias. Uses linux/amd64 buildx for EC2 Ubuntu amd64 targets.
image-build-check: image-buildx-amd64-check

# Native architecture build check. On Apple Silicon, ai-worker can fail because paddlepaddle has no Linux arm64 wheel.
image-build-native-check: image-build-app image-build-ai image-build-frontend

image-build-app:
	docker build -f app/Dockerfile -t $(APP_IMAGE) .

image-build-ai:
	docker build -f ai_runtime/Dockerfile -t $(AI_WORKER_IMAGE) .

image-build-frontend:
	docker build -f frontend/Dockerfile -t $(FRONTEND_IMAGE) \
		--build-arg VITE_API_BASE_URL=$(VITE_API_BASE_URL) \
		.

image-buildx-amd64-check:
	docker buildx build --platform $(DOCKER_PLATFORM) -f app/Dockerfile -t $(APP_IMAGE) --load .
	docker buildx build --platform $(DOCKER_PLATFORM) -f ai_runtime/Dockerfile -t $(AI_WORKER_IMAGE) --load .
	docker buildx build --platform $(DOCKER_PLATFORM) -f frontend/Dockerfile -t $(FRONTEND_IMAGE) --load \
		--build-arg VITE_API_BASE_URL=$(VITE_API_BASE_URL) \
		.

image-tags:
	@printf '%s\n' "$(APP_IMAGE)" "$(AI_WORKER_IMAGE)" "$(FRONTEND_IMAGE)"
