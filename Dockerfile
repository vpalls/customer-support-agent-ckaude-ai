# ============================================
# Customer Support Agent — Docker Image
# Python 3.12 + Chainlit + Claude + ChromaDB
# ============================================
FROM python:3.12-slim

# ── System dependencies ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Set working directory ──
WORKDIR /app

# ── Install Python deps (cached layer) ──
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Pre-download the embedding model (cached layer) ──
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── Copy application code ──
COPY app.py agent.py knowledge_base_loader.py ./
COPY chainlit.md ./
COPY .chainlit/ ./.chainlit/
COPY public/ ./public/
COPY data/ ./data/

# ── Create persistent volume mount points ──
RUN mkdir -p /app/chroma_db

# ── Environment ──
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/data
ENV CHROMA_PERSIST_DIR=/app/chroma_db

# ── Expose Chainlit port ──
EXPOSE 8000

# ── Health check ──
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# ── Start Chainlit ──
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
