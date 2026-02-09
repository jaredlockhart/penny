# Check tool configuration (single source of truth for tool parameters)
RUFF_TARGETS = penny/
PYTEST_ARGS = penny/tests/ -v
TEAM_RUFF_TARGETS = penny_team/
TEAM_PYTEST_ARGS = tests/ -v

.PHONY: up prod kill build team-build fmt lint fix typecheck check pytest token migrate-test migrate-validate

# --- Docker Compose ---

up:
	GIT_COMMIT=$$(git rev-parse --short HEAD 2>/dev/null || echo unknown) docker compose --profile team up --build

prod:
	GIT_COMMIT=$$(git rev-parse --short HEAD 2>/dev/null || echo unknown) docker compose -f docker-compose.yml up --build penny

kill:
	docker compose --profile team down --rmi local --remove-orphans

build:
	GIT_COMMIT=$$(git rev-parse --short HEAD 2>/dev/null || echo unknown) docker compose build penny

team-build:
	docker compose build team

# Print a GitHub App installation token for use with gh CLI
# Usage: GH_TOKEN=$(make token) gh pr create ...
token:
	@docker compose run --rm --no-deps --entrypoint "" pm uv run /repo/penny-team/penny_team/utils/github_app.py 2>/dev/null

# --- Code quality (auto-detects host vs container via LOCAL env var) ---

ifdef LOCAL
# Inside a container — run tools directly
RUN = cd penny &&
TEAM_RUN = cd penny-team &&
else
# On host — run tools inside Docker containers
RUN = docker compose run --rm penny
TEAM_RUN = docker compose run --rm team
endif

fmt: $(if $(LOCAL),,build team-build)
	$(RUN) ruff format $(RUFF_TARGETS)
	$(TEAM_RUN) ruff format $(TEAM_RUFF_TARGETS)

lint: $(if $(LOCAL),,build team-build)
	$(RUN) ruff check $(RUFF_TARGETS)
	$(TEAM_RUN) ruff check $(TEAM_RUFF_TARGETS)

fix: $(if $(LOCAL),,build team-build)
	$(RUN) ruff format $(RUFF_TARGETS)
	$(RUN) ruff check --fix $(RUFF_TARGETS)
	$(TEAM_RUN) ruff format $(TEAM_RUFF_TARGETS)
	$(TEAM_RUN) ruff check --fix $(TEAM_RUFF_TARGETS)

typecheck: $(if $(LOCAL),,build team-build)
	$(RUN) ty check $(RUFF_TARGETS)
	$(TEAM_RUN) ty check $(TEAM_RUFF_TARGETS)

check: $(if $(LOCAL),,build team-build)
	$(RUN) ruff format --check $(RUFF_TARGETS)
	$(RUN) ruff check $(RUFF_TARGETS)
	$(RUN) ty check $(RUFF_TARGETS)
	$(RUN) python -m penny.database.migrate --validate
	$(RUN) pytest $(PYTEST_ARGS)
	$(TEAM_RUN) ruff format --check $(TEAM_RUFF_TARGETS)
	$(TEAM_RUN) ruff check $(TEAM_RUFF_TARGETS)
	$(TEAM_RUN) ty check $(TEAM_RUFF_TARGETS)
	$(TEAM_RUN) pytest $(TEAM_PYTEST_ARGS)

pytest: $(if $(LOCAL),,build team-build)
	$(RUN) pytest $(PYTEST_ARGS)
	$(TEAM_RUN) pytest $(TEAM_PYTEST_ARGS)

migrate-test: $(if $(LOCAL),,build)
	$(RUN) python -m penny.database.migrate --test

migrate-validate: $(if $(LOCAL),,build)
	$(RUN) python -m penny.database.migrate --validate
