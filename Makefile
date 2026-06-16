DEV_ENV ?= .env
PROD_ENV_FILE ?= .prod.env
DEV_COMPOSE = docker compose --env-file $(DEV_ENV) -f infra/docker/docker-compose.dev.yml
PROD_COMPOSE = docker compose --env-file $(PROD_ENV_FILE) -f infra/docker/docker-compose.prod.yml
COMPOSE_DEV = $(DEV_COMPOSE)
FASTAPI_EXEC = $(DEV_COMPOSE) exec fastapi uv run --no-sync
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
RAG_OPENAI_LIMIT = $(if $(LIMIT),$(LIMIT),1)
RAG_VECTOR_TOP_K = $(if $(TOP_K),$(TOP_K),3)

# Target groups:
# dev-*     Local Docker development stack.
# qa-*      Local validation commands.
# prod-*    Production/deployment commands; no seed/RAG ingest is automatic.
# rag-*     RAG management. Write/costly actions route through danger-* aliases.
# danger-*  DB write, provider cost, or destructive local maintenance actions.
# image-*   Release image build/tag checks.

# Legacy/minimal app stack
# Root docker-compose.yml wrapper for backend/AI checks only.
# Standard local dev/demo execution should use dev-* or demo-* targets below.
.PHONY: app-up app-up-full app-worker-up app-build app-worker-build app-rebuild app-clean-image danger-app-clean-image app-down app-ps app-logs app-worker-logs
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

app-clean-image: danger-app-clean-image

danger-app-clean-image:
	@echo "DANGER: removing local Docker images. Stop containers first if Docker refuses removal."
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
.PHONY: dev-network dev-up dev-stack dev-front dev-local dev-down dev-ps dev-logs dev-migrate dev-seed dev-health dev-rebuild-api dev-rebuild-worker dev-rebuild-frontend dev-rebuild-all dev-restart-nginx dev-config-check
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
	@echo "WARNING: dev-seed writes challenge seed data to the development DB."
	$(DEV_COMPOSE) exec fastapi uv run --no-sync python scripts/seed_mvp_challenges.py

dev-health:
	curl -fsS http://localhost:8080/api/v1/system/health

dev-rebuild-api:
	$(DEV_COMPOSE) up -d --build fastapi ai-worker
	$(DEV_COMPOSE) up -d --force-recreate nginx
	sleep 5
	curl -fsS http://localhost:8080/api/v1/system/health

dev-rebuild-worker:
	$(DEV_COMPOSE) up -d --build ai-worker
	sleep 3
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
	@echo "Checking rendered dev compose validity with config --quiet. No compose config contents are printed."
	$(DEV_COMPOSE) config --quiet

.PHONY: qa qa-backend qa-frontend qa-diff
qa: qa-diff qa-backend qa-frontend

qa-backend:
	uv run ruff check app scripts ai_runtime tests
	uv run pytest -q

qa-frontend:
	cd frontend && npm run build

qa-diff:
	git diff --check

.PHONY: rag-preview rag-ingest-dry-run rag-ingest-apply danger-rag-ingest-apply rag-embed-dry-run rag-embed-apply-openai-dry-run rag-embed-apply-openai danger-rag-embed-apply-openai rag-vector-query
rag-preview:
	uv run python scripts/rag/preview_rag_chunks.py --json

rag-ingest-dry-run:
	$(FASTAPI_EXEC) python scripts/rag/ingest_rag_chunks.py --json

# Backward-compatible alias. Prefer danger-rag-ingest-apply for explicit DB writes.
rag-ingest-apply: danger-rag-ingest-apply

# DANGER: DB write 발생. dry-run 결과 확인 후 명시적으로 실행하세요.
danger-rag-ingest-apply:
	@echo "DANGER: writing RAG chunks to the development DB."
	$(FASTAPI_EXEC) python scripts/rag/ingest_rag_chunks.py --apply --json

rag-embed-dry-run:
	$(FASTAPI_EXEC) python scripts/rag/embed_rag_chunks.py --provider disabled --json

# OpenAI API 비용이 발생할 수 있음: 기본 LIMIT=1, 예: make rag-embed-apply-openai-dry-run LIMIT=1
rag-embed-apply-openai-dry-run:
	@echo "WARNING: this may call OpenAI for embeddings. Default LIMIT=$(RAG_OPENAI_LIMIT)."
	$(FASTAPI_EXEC) python scripts/rag/embed_rag_chunks.py --provider openai --json --limit $(RAG_OPENAI_LIMIT)

# Backward-compatible alias. Prefer danger-rag-embed-apply-openai for explicit DB writes + OpenAI cost.
rag-embed-apply-openai: danger-rag-embed-apply-openai

# DANGER: DB write + OpenAI API 비용이 발생할 수 있음: 기본 LIMIT=1, 예: make danger-rag-embed-apply-openai LIMIT=1
danger-rag-embed-apply-openai:
	@echo "DANGER: this writes embeddings to DB and may call OpenAI. Default LIMIT=$(RAG_OPENAI_LIMIT)."
	$(FASTAPI_EXEC) python scripts/rag/embed_rag_chunks.py --provider openai --apply --json --limit $(RAG_OPENAI_LIMIT)

# Read-only vector search check. OpenAI query embedding cost may occur.
rag-vector-query:
	@echo "WARNING: read-only query, but OpenAI query embedding cost may occur."
	$(FASTAPI_EXEC) python scripts/rag/query_vector_rag.py --query "$(QUERY)" --top-k $(RAG_VECTOR_TOP_K) --provider openai --json

.PHONY: demo-up demo-down demo-ps demo-logs demo-health
demo-up: dev-up

demo-down: dev-down

demo-ps: dev-ps

demo-logs: dev-logs

demo-health: dev-health

# Langfuse stack
# Optional self-hosted observability stack via infra/langfuse/docker-compose.yml.
.PHONY: langfuse-up langfuse-down langfuse-ps langfuse-logs langfuse-health langfuse-restart
langfuse-up:
	./scripts/docker_stack.sh langfuse up

langfuse-down:
	./scripts/docker_stack.sh langfuse down

langfuse-ps:
	./scripts/docker_stack.sh langfuse ps

langfuse-logs:
	./scripts/docker_stack.sh langfuse logs

langfuse-health:
	curl -fsS http://127.0.0.1:3000 > /dev/null
	@echo "Langfuse health OK: http://127.0.0.1:3000"

langfuse-restart:
	cd infra/langfuse && docker compose restart langfuse-web langfuse-worker

# Prod compose convenience
# Uses infra/docker/docker-compose.prod.yml and pulls prebuilt registry images.
.PHONY: prod-pull prod-up prod-ps prod-logs prod-migrate prod-seed danger-prod-seed prod-health prod-release-db
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

prod-seed: danger-prod-seed

danger-prod-seed:
	@echo "DANGER: this writes challenge seed data to the production DB. Run only for an explicitly approved initial seed."
	$(PROD_COMPOSE) exec -T fastapi uv run --no-sync python scripts/seed_mvp_challenges.py

prod-health:
	curl -fsS http://localhost:$${NGINX_HTTP_PORT:-8080}/api/v1/system/health

prod-release-db: prod-migrate
	@echo "prod-release-db now runs migration only. Use danger-prod-seed separately when production seed is explicitly approved."

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
