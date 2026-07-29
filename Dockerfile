FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system packages required by OpenCV and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Install dependencies from uv.lock
RUN uv sync --frozen --no-dev

# Copy application source
COPY app.py .
COPY routers ./routers
COPY utils ./utils

EXPOSE 8502

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8502"]