DC = docker-compose run --rm penny

.PHONY: up test prod kill fmt lint fix typecheck check

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

fmt:
	$(DC) ruff format penny/

lint:
	$(DC) ruff check penny/

fix:
	$(DC) ruff format penny/
	$(DC) ruff check --fix penny/

typecheck:
	$(DC) ty check penny/

check:
	$(DC) ruff format --check penny/
	$(DC) ruff check penny/
	$(DC) ty check penny/
