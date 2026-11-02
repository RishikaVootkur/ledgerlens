.PHONY: setup ingest index eval test lint dev up down web

setup:  ## Install Python and web dependencies, pull the embedding model
	uv sync --all-extras
	uv run pre-commit install
	cd web && npm ci
	ollama pull nomic-embed-text

ingest:  ## Download filings from EDGAR, parse, chunk and index them
	uv run ledgerlens ingest run
	uv run ledgerlens index build

eval:  ## Run the dev-split eval with the default config
	uv run ledgerlens eval run --config eval/configs/default.yaml --split dev

test:  ## Lint, type-check and unit tests
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run pytest -m "not integration" --cov --cov-report=term-missing

dev:  ## Start Qdrant and Phoenix in Docker, then the API and web dev servers
	docker compose up -d qdrant phoenix
	uv run uvicorn ledgerlens.api.app:app --reload --port 8000 & cd web && npm run dev

up:  ## Full stack in Docker
	docker compose up -d --build

down:
	docker compose down
