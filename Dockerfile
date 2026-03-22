FROM python:3.12-slim

# System deps for cryptography (Kalshi RSA signing) and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# BGE-M3 model will download on first run and be cached in /app/.cache
# Mount a volume here for persistence across restarts
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Non-root user for security
RUN useradd -m -u 1000 bot && chown -R bot:bot /app
USER bot

CMD ["python", "main.py"]
