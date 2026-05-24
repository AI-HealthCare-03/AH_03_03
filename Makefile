.PHONY: app-up app-up-full app-worker-up app-build app-worker-build app-rebuild app-clean-image app-down app-ps app-logs app-worker-logs
.PHONY: dev-up dev-down dev-ps
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

langfuse-up:
	./scripts/docker_stack.sh langfuse up

langfuse-down:
	./scripts/docker_stack.sh langfuse down

langfuse-ps:
	./scripts/docker_stack.sh langfuse ps

langfuse-logs:
	./scripts/docker_stack.sh langfuse logs
