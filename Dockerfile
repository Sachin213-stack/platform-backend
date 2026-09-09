# ============================================
# AI-CTO Backend — Multi-stage Production Dockerfile
# ============================================

# -------------------------------------------------------------
# Stage 1: Build & Dependency Compilation Stage
# -------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies required for compiling C extensions (asyncpg, bcrypt, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies into virtual environment
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------------------
# Stage 2: Minimal Runtime Stage
# -------------------------------------------------------------
FROM python:3.12-slim AS runtime

# System environment flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install minimal runtime system dependencies (libgomp1 required for scikit-learn OpenMP acceleration)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create unprivileged system user and logs directory
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -s /bin/bash -m appuser && \
    mkdir -p /app/logs && \
    chown -R appuser:appuser /app

# Copy application code into container (filtered by .dockerignore)
COPY --chown=appuser:appuser . /app

# Run as non-root user
USER appuser

# Expose FastAPI listening port
EXPOSE 8000

# Container health probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, httpx; port = os.environ.get('PORT', '8000'); r = httpx.get(f'http://localhost:{port}/api/health'); exit(0 if r.status_code == 200 else 1)" || exit 1

# Launch production server with dynamic port support (defaults to 8000 for local Docker)
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

