DC = docker compose run --rm penny

# Check tool configuration (single source of truth for tool parameters)
RUFF_TARGETS = penny/
PYTEST_ARGS = penny/tests/ -v

.PHONY: up prod kill build fmt lint fix typecheck check pytest fmt-local fix-local check-local

# --- Docker-based (host / CI) ---

up:
	docker compose --profile team up --build

prod:
	cp .env.prod .env
	docker compose -f docker-compose.yml up --build

kill:
	docker compose --profile team down --rmi local --remove-orphans

build:
	docker compose build penny

fmt: build
	$(DC) ruff format $(RUFF_TARGETS)

lint: build
	$(DC) ruff check $(RUFF_TARGETS)

fix: build
	$(DC) ruff format $(RUFF_TARGETS)
	$(DC) ruff check --fix $(RUFF_TARGETS)

typecheck: build
	$(DC) ty check $(RUFF_TARGETS)

check: build
	$(DC) ruff format --check $(RUFF_TARGETS)
	$(DC) ruff check $(RUFF_TARGETS)
	$(DC) ty check $(RUFF_TARGETS)
	$(DC) pytest $(PYTEST_ARGS)

pytest: build
	$(DC) pytest $(PYTEST_ARGS)

# --- Direct execution (agent containers) ---

fmt-local:
	cd app && ruff format $(RUFF_TARGETS)

fix-local:
	cd app && ruff format $(RUFF_TARGETS)
	cd app && ruff check --fix $(RUFF_TARGETS)

check-local:
	cd app && ruff format --check $(RUFF_TARGETS)
	cd app && ruff check $(RUFF_TARGETS)
	cd app && ty check $(RUFF_TARGETS)
	cd app && pytest $(PYTEST_ARGS)
