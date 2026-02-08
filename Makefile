# Check tool configuration (single source of truth for tool parameters)
RUFF_TARGETS = penny/
PYTEST_ARGS = penny/tests/ -v

.PHONY: up prod kill build fmt lint fix typecheck check pytest token

# --- Docker Compose ---

up:
	docker compose --profile team up --build

prod:
	docker compose -f docker-compose.yml up --build penny

kill:
	docker compose --profile team down --rmi local --remove-orphans

build:
	docker compose build penny

# Print a GitHub App installation token for use with gh CLI
# Usage: GH_TOKEN=$(make token) gh pr create ...
token:
	@docker compose run --rm --no-deps --entrypoint "" pm uv run /repo/agents/github_app.py 2>/dev/null

# --- Code quality (auto-detects host vs container via LOCAL env var) ---

ifdef LOCAL
# Inside a container — run tools directly from app/ subdir
RUN = cd app &&
else
# On host — run tools inside the penny container
RUN = docker compose run --rm penny
endif

fmt: $(if $(LOCAL),,build)
	$(RUN) ruff format $(RUFF_TARGETS)

lint: $(if $(LOCAL),,build)
	$(RUN) ruff check $(RUFF_TARGETS)

fix: $(if $(LOCAL),,build)
	$(RUN) ruff format $(RUFF_TARGETS)
	$(RUN) ruff check --fix $(RUFF_TARGETS)

typecheck: $(if $(LOCAL),,build)
	$(RUN) ty check $(RUFF_TARGETS)

check: $(if $(LOCAL),,build)
	$(RUN) ruff format --check $(RUFF_TARGETS)
	$(RUN) ruff check $(RUFF_TARGETS)
	$(RUN) ty check $(RUFF_TARGETS)
	$(RUN) pytest $(PYTEST_ARGS)

pytest: $(if $(LOCAL),,build)
	$(RUN) pytest $(PYTEST_ARGS)
