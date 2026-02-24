# Agent-UniRAG API with uv (Astral)
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN uv sync --no-dev

EXPOSE 8000

ENV HOST=0.0.0.0
ENV PORT=8000

# Single worker: agent and PDF retriever are in-memory; multiple workers would each have separate state.
CMD ["uv", "run", "gunicorn", "agent_unirag.api:app", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
