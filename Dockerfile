# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – dependency builder
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some wheels (e.g. opencv headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libstdc++6 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Non-root user for security
RUN useradd --create-home appuser \
 && mkdir -p /app/checkpoints \
 && chown -R appuser:appuser /app

USER appuser

# Application source
COPY --chown=appuser:appuser app.py .

# Environment defaults (overridden by .env / docker-compose / CI)
ENV LOG_LEVEL=INFO \
    MODEL_PATH=/app/checkpoints/best_model.pth \
    IMAGE_SIZE=256 \
    FLASK_ENV=production

EXPOSE 5000

# Use Waitress for production WSGI serving
CMD ["python", "-m", "waitress", "--host=0.0.0.0", "--port=5000", "app:app"]
