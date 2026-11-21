FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache \
    CHECKPOINT_PATH=/app/model_weights/emotion2vec_final_model.pth \
    EMOTION2VEC_MODEL_PATH=/app/models/emotion2vec_plus_base

# FFmpeg and libsndfile are required for audio conversion and Emotion2Vec+ Base
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch first (matches verified Beyond_Words versions, much smaller than CUDA)
RUN pip install --no-cache-dir \
    torch==2.15.0 torchaudio==2.11.1 \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download Emotion2Vec+ Base snapshot into the image so Cloud Run starts fast with no HF runtime dependency
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='emotion2vec/emotion2vec_plus_base', local_dir='/app/models/emotion2vec_plus_base')"

COPY model_weights/ model_weights/
COPY predict_voice.py main.py index.html ARCHITECTURE.md ./

EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --log-level info"]
