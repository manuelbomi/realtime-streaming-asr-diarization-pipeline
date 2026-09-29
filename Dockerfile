# Linux CPU image. For GPU serving, swap the base image for a CUDA-enabled
# one and set ASR_DEVICE=cuda / ASR_COMPUTE_TYPE=float16 (see docs/PRODUCTION.md).
FROM python:3.11-slim

WORKDIR /app

# libgomp1: required by onnxruntime (webrtcvad-wheels/CTranslate2 dependency chain)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY asr/ asr/
COPY diarization/ diarization/
COPY api/ api/

ENV ASR_MODEL_SIZE=tiny.en \
    ASR_DEVICE=cpu \
    ASR_COMPUTE_TYPE=int8

EXPOSE 8000
CMD ["uvicorn", "api.ws_server:app", "--host", "0.0.0.0", "--port", "8000"]
