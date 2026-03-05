FROM python:3.11-slim

WORKDIR /app

# Install system deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download model weights from Hugging Face at build time (if HF_REPO_ID is set)
# Or weights can be baked in by placing .pth files in checkpoints/ before building
ARG HF_REPO_ID=""
RUN if [ -n "$HF_REPO_ID" ]; then \
    HF_REPO_ID=$HF_REPO_ID python download_weights.py; \
    fi

EXPOSE 8000

# Run weight download at startup (in case they weren't baked in), then start server
CMD python download_weights.py && uvicorn app:app --host 0.0.0.0 --port 8000
