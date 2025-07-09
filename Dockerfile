# ─────────────────────────  Dockerfile  ─────────────────────────
FROM python:3.11-slim

# --- system libs ------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# --- HF caches (world‑writable) --------------------------------
ENV HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/tmp/huggingface/sentence-transformers
RUN mkdir -p $HF_HOME/transformers $HF_HOME/sentence-transformers && \
    chmod -R 777 $HF_HOME          # ensure the non-root user can write

# --- python deps ------------------------------------------------
WORKDIR /tmp
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- non‑root user ---------------------------------------------
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

# --- copy code --------------------------------------------------
WORKDIR /app
COPY --chown=user . .

# --- tell Docker & HF which port we use ------------------------
EXPOSE 7860

# --- start ------------------------------------------------------
# ── start ------------------------------------------------------
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
