# Observability: tracing into Arize Phoenix

LedgerLens can send OpenTelemetry traces with OpenInference attributes to the Phoenix service
in `docker-compose.yml`. Each question becomes one trace:

- `agent` (AGENT): the question, the final answer, the model, filters and usage totals.
- One CHAIN span per graph step: `plan`, `tools`, `answer`, `verify`, `self_check`, `repair`.
- One span per tool call: RETRIEVER for `search_filings` (query, method, filters, chunk ids),
  TOOL for `get_company_facts`, `get_table` and `calculate`.
- One LLM span per model call: model, provider, prompt/completion/total tokens, provider
  latency, list-price-equivalent cost (`llm.cost.*`, labeled `ledgerlens.cost_label`), and
  whether the response came from the cache.

Traces hold full prompts, questions and retrieved filing text. Send them only to a collector
you trust; the local Phoenix is the intended one.

## Tracing is opt-in

Phoenix 21 starts with authentication on and rejects unauthenticated span exports with 401.
LedgerLens therefore traces only when both `PHOENIX_ENDPOINT` and `PHOENIX_API_KEY` are set.
With either one missing, `setup_tracing()` is a no-op: it logs `tracing.disabled` once and
nothing is exported. That covers the API, `ledgerlens ask` and later eval runs.

## Create a Phoenix system API key

1. Start Phoenix: `docker compose up -d phoenix`, then open http://localhost:6006.
2. Sign in as the admin user. On a fresh Phoenix the first login is `admin@localhost` with
   password `admin`, and Phoenix asks you to choose a new password straight away.
3. Open **Settings** (the gear in the left sidebar) and find the **API Keys** section.
4. Create a **System** key (system keys belong to the deployment, not to one user, which
   suits a service), give it a name such as `ledgerlens`, and copy the key. Phoenix shows it
   only once.

## Configure LedgerLens

Add both values to `.env` in the repository root (never commit `.env`):

```bash
PHOENIX_ENDPOINT=http://localhost:6006/v1/traces
PHOENIX_API_KEY=<the system key>
```

- Local runs (`make dev`, `uv run ledgerlens ask ...`) send to `localhost:6006`.
- In Docker, the `api` service turns any non-empty `PHOENIX_ENDPOINT` into
  `http://phoenix:6006/v1/traces` (the service name on the compose network) and reads the key
  from `.env`. Restart it after changing `.env`: `docker compose up -d api`.

## Check that spans arrive

Ask one question, for example
`uv run ledgerlens ask "What was Coca-Cola net income in fiscal 2025?" --ticker KO`, then open
the `ledgerlens` project in the Phoenix UI. The API logs `tracing.enabled` at startup when
tracing is on, and `tracing.disabled` when it is skipped. If the key is wrong, the OpenTelemetry
exporter logs `Failed to export spans batch code: 401`. Answers are never affected.

Shutdown flushes pending spans for at most 2 seconds, so a slow or stopped Phoenix never holds
up `docker compose stop` or the end of a CLI command.
