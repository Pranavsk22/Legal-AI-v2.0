FROM python:3.11-slim

# Installing system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

# ── HF cache dirs (create *before* chown) ────────────────────
ENV HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/tmp/huggingface/sentence-transformers
RUN mkdir -p /tmp/huggingface/transformers /tmp/huggingface/sentence-transformers

# ── non‑root user & basic dirs ───────────────────────────────
RUN useradd -m user && \
    mkdir -p /app && \
    chown -R user:user /app /tmp/huggingface

WORKDIR /app

# ── deps (still as root, no cache) ───────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── copy code & switch user ─────────────────────────────────
COPY --chown=user . .
USER user
ENV PATH="/home/user/.local/bin:$PATH"

EXPOSE 7860 
# ── start FastAPI on whatever port HF assigns ───────────────
#CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "7860"]

HEALTHCHECK CMD curl -fs http://localhost:7860/healthz || exit 1

CMD uvicorn backend.api.main:app --host 0.0.0.0 --port 7860 --lifespan on --log-level info
