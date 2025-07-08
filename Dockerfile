FROM python:3.11-slim

# System deps for torch / cv2; trim if you don’t need opencv
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PORT=7860  HF_HUB_DISABLE_SYMLINKS_WARNING=1
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
