PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e .[dev]

collect-tests:
	pytest --collect-only -q

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .

unit:
	bash infra/scripts/run_unit_tests.sh

integration:
	bash infra/scripts/run_integration_tests.sh

validate:
	bash infra/scripts/run_full_validation.sh

run-api:
	python -m uvicorn apps.api.main:create_app --factory --reload --port 8000

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	bash infra/scripts/clean_repo.sh
