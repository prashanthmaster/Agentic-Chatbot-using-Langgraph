FROM python:3.11-slim

# Prevent Python cache files and enable immediate container logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Where the SQLite checkpointer and FAISS indexes are written.
# Mount a volume here in production, or all conversations are lost on redeploy.
ENV DATA_DIR=/app/data

WORKDIR /app

# libgomp1 is the OpenMP runtime that faiss-cpu links against at import time.
# build-essential was REMOVED: every dependency in requirements.txt ships a
# prebuilt manylinux wheel, so no compiler is needed. Saves ~250MB per image
# and several minutes per build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so the (slow) dependency layer is cached and only
# rebuilt when requirements.txt changes - not on every edit to app.py.
COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application. .dockerignore keeps .env, .git and runtime state out.
COPY . .

# Run as a non-root user. Create the data directory first and hand it over,
# so a mounted volume inherits workable ownership.
RUN mkdir -p "$DATA_DIR" \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Lets Docker itself report health, independently of the CI workflow's check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5)" || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
