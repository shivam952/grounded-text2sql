FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency and project metadata files first (layer cache)
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/

# Install dependencies with uv (frozen lockfile)
RUN uv sync --frozen --no-dev

# Expose port
EXPOSE 8080

# Run with uvicorn
CMD ["uv", "run", "uvicorn", "groundedsql.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
