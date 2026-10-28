FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends docker.io ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV UV_LINK_MODE=copy PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
    NASDAQ_ENV_FILE=none PATH="/app/.venv/bin:$PATH"
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra ui --no-dev
COPY nasdaq_agent ./nasdaq_agent
COPY config.yaml main.py ./
EXPOSE 8000
CMD ["uvicorn", "nasdaq_agent.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
