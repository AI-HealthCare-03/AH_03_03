.PHONY: app-up app-up-full app-worker-up app-build app-worker-build app-rebuild app-clean-image app-down app-ps app-logs app-worker-logs
.PHONY: dev-up dev-down dev-ps
.PHONY: demo-up demo-down demo-ps demo-logs demo-health
.PHONY: langfuse-up langfuse-down langfuse-ps langfuse-logs

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

dev-up:
	./scripts/docker_stack.sh dev up

dev-down:
	./scripts/docker_stack.sh dev down

dev-ps:
	./scripts/docker_stack.sh dev ps

demo-up:
	docker network inspect ai-health-shared >/dev/null 2>&1 || docker network create ai-health-shared >/dev/null
	docker compose -f infra/docker/docker-compose.dev.yml up -d --build

demo-down:
	docker compose -f infra/docker/docker-compose.dev.yml down

demo-ps:
	docker compose -f infra/docker/docker-compose.dev.yml ps

demo-logs:
	docker compose -f infra/docker/docker-compose.dev.yml logs --tail=100 nginx frontend fastapi

demo-health:
	curl -fsS http://localhost:8080/api/v1/system/health

langfuse-up:
	./scripts/docker_stack.sh langfuse up

langfuse-down:
	./scripts/docker_stack.sh langfuse down

langfuse-ps:
	./scripts/docker_stack.sh langfuse ps

langfuse-logs:
	./scripts/docker_stack.sh langfuse logs
