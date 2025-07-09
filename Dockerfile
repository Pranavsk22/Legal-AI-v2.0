FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set Hugging Face cache path
ENV HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/tmp/huggingface/sentence-transformers

# Create user and directories
RUN useradd -m user && \
    mkdir -p /app && \
    chown -R user:user /app /tmp/huggingface

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY --chown=user . .

USER user
ENV PATH="/home/user/.local/bin:$PATH"

# 🧠 KEY CHANGE: use $PORT env var!
CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT"]
