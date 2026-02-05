DC = docker-compose run --rm penny

.PHONY: up test prod kill build fmt lint fix typecheck check pytest

up:
	docker-compose up --build

test:
	cp data/penny.db data/test.db 2>/dev/null || true
	cp .env.test .env
	docker-compose up --build

prod:
	cp .env.prod .env
	docker-compose up --build

kill:
	docker-compose down --rmi local --remove-orphans

build:
	docker-compose build penny

fmt: build
	$(DC) ruff format penny/

lint: build
	$(DC) ruff check penny/

fix: build
	$(DC) ruff format penny/
	$(DC) ruff check --fix penny/

typecheck: build
	$(DC) ty check penny/

check: build
	$(DC) ruff format --check penny/
	$(DC) ruff check penny/
	$(DC) ty check penny/
	$(DC) pytest penny/tests/ -v

pytest: build
	$(DC) pytest penny/tests/ -v
