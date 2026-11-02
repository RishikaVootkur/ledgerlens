# LedgerLens API image: FastAPI + the agent, no torch (fastembed runs on ONNX).
# Build: docker compose build api

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /src
# Dependencies first, so code changes do not reinstall them.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin ledgerlens \
    && mkdir -p /app/data /app/eval /app/.cache \
    && chown -R ledgerlens:ledgerlens /app
COPY --from=builder --chown=ledgerlens:ledgerlens /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    LEDGERLENS_HOME=/app \
    LOG_JSON=true \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    HF_HOME=/app/.cache/huggingface
WORKDIR /app
USER ledgerlens
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/models', timeout=4)"]
CMD ["uvicorn", "ledgerlens.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-proxy-headers"]
