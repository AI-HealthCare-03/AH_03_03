.PHONY: app-up app-up-full app-worker-up app-build app-worker-build app-rebuild app-clean-image app-down app-ps app-logs app-worker-logs
.PHONY: dev-network dev-up dev-down dev-ps dev-logs dev-migrate dev-seed dev-health
.PHONY: demo-up demo-down demo-ps demo-logs demo-health
.PHONY: langfuse-up langfuse-down langfuse-ps langfuse-logs

COMPOSE_DEV = docker compose --env-file .env -f infra/docker/docker-compose.dev.yml

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

dev-network:
	docker network inspect ai-health-shared >/dev/null 2>&1 || docker network create ai-health-shared >/dev/null

dev-up: dev-network
	$(COMPOSE_DEV) up -d --build

dev-down:
	$(COMPOSE_DEV) down --remove-orphans

dev-ps:
	$(COMPOSE_DEV) ps

dev-logs:
	$(COMPOSE_DEV) logs --tail=100 fastapi ai-worker frontend nginx

dev-migrate:
	$(COMPOSE_DEV) exec fastapi uv run --no-sync aerich upgrade

dev-seed:
	$(COMPOSE_DEV) exec fastapi uv run --no-sync python scripts/seed_mvp_challenges.py

dev-health:
	curl -fsS http://localhost:8080/api/v1/system/health

demo-up: dev-up

demo-down: dev-down

demo-ps: dev-ps

demo-logs: dev-logs

demo-health: dev-health

langfuse-up:
	./scripts/docker_stack.sh langfuse up

langfuse-down:
	./scripts/docker_stack.sh langfuse down

langfuse-ps:
	./scripts/docker_stack.sh langfuse ps

langfuse-logs:
	./scripts/docker_stack.sh langfuse logs
