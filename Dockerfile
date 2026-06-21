# =============================================================================
# Stage 1 – dependency builder
# Separate stage so the heavy pip install layer is cached independently
# from source-code changes.
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps needed to compile some Python packages (pymupdf, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# =============================================================================
# Stage 2 – runtime image
# =============================================================================
FROM python:3.12-slim AS runtime

WORKDIR /app

# Carry over compiled packages from builder
COPY --from=builder /install /usr/local

# System runtime libs (same as build stage, minus build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy application source
COPY . .

# Install the project itself in editable-equivalent mode (non-editable for prod)
RUN pip install --no-cache-dir -e .

# Streamlit listens on 8501 by default
EXPOSE 8501

# Health check — curl the Streamlit health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# Run the app
ENTRYPOINT ["streamlit", "run", "app/main.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
