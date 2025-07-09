FROM python:3.11-slim

# ── Install system packages ──
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# ── Hugging Face cache environment ──
ENV HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/tmp/huggingface/sentence-transformers

# ── Create dirs & user ──
RUN mkdir -p /tmp/huggingface/transformers /tmp/huggingface/sentence-transformers && \
    useradd -m -u 1000 user && \
    chown -R user:user /tmp/huggingface

# ── Install Python dependencies before switching to user ──
WORKDIR /tmp
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Switch to non-root user ──
ENV PATH="/home/user/.local/bin:${PATH}"
USER user

# ── Copy application code ──
WORKDIR /app
COPY --chown=user . .

# ── Run the app ──
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
