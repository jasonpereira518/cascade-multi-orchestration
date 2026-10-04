.PHONY: sync check lint format typecheck test smoke docker-test

sync:
	uv sync --locked

check: lint typecheck test

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run pyright

test:
	uv run pytest -m "not smoke"

smoke:
	uv run pytest -m smoke

docker-test:
	uv run pytest -m docker
