# Image for the compose `api` and `worker` services (profile "service", P6).
# Only the service + ingest extras: torch/ml extras are added when the worker
# starts running verification (P6), so P0-P5 builds stay small.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first so code edits don't invalidate the dependency layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --extra service --extra ingest

COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev --extra service --extra ingest

CMD ["python", "-c", "import app; print('app image ok')"]
