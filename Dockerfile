FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    WHISPER_MODEL=base \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    HF_HOME=/var/data/huggingface \
    CLASSLENS_MODEL_CACHE=/var/data/classlens

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        ffmpeg \
        libavformat-dev \
        libavcodec-dev \
        libavdevice-dev \
        libavutil-dev \
        libswscale-dev \
        libswresample-dev \
        libavfilter-dev \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -c "import av; print(f'PyAV {av.__version__} installed')" \
    && ffmpeg -version

COPY assessor ./assessor
COPY web ./web
COPY server.py config.json ./

RUN useradd --create-home --shell /usr/sbin/nologin classlens \
    && mkdir -p /home/classlens/.cache /var/data /tmp/classlens_uploads "$NUMBA_CACHE_DIR" \
    && chown -R classlens:classlens /app /home/classlens /var/data /tmp/classlens_uploads "$NUMBA_CACHE_DIR" \
    && su -s /bin/sh classlens -c 'python -c "import librosa, numpy as np; librosa.yin(np.zeros(22050, dtype=np.float32), fmin=65, fmax=500, sr=22050); print(\"librosa YIN smoke test passed\")"'
COPY docker-entrypoint.sh /usr/local/bin/classlens-entrypoint
RUN chmod 755 /usr/local/bin/classlens-entrypoint
ENTRYPOINT ["/usr/local/bin/classlens-entrypoint"]

EXPOSE 10000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "10000"]
