FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git curl && \
    rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd --create-home --shell /bin/bash habla
WORKDIR /app

COPY habla/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY habla/ .
# Assets shared with the Android app (prompts, ASR profile, idioms, languages)
COPY shared/ /app/shared/
ENV HABLA_SHARED_DIR=/app/shared

# Ensure data directories exist and are writable
RUN mkdir -p data/audio data/idioms && chown -R habla:habla /app

USER habla

EXPOSE 8002

# Health check — Docker will restart unhealthy containers with restart policy
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8002"]
