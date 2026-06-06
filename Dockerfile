FROM python:3.12-slim

# Install uv
RUN pip install uv --quiet

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml .
COPY setup.cfg .

# Install project dependencies (no dev tools)
RUN uv venv && uv sync --no-dev

# Copy source code, config, and pre-trained models
COPY src/ src/
COPY conf/ conf/

# Models must be mounted or baked in at build time.
# For local usage: docker run -v $(pwd)/models:/app/models ...
# For production: COPY models/ models/

# Expose API port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uv", "run", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
