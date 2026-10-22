FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    CLASSLENS_DEFAULT_WHISPER_MODEL=tiny \
    WHISPER_MODEL=tiny \
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
    && mkdir -p /home/classlens/.cache /var/data /tmp/classlens_uploads \
    && chown -R classlens:classlens /app /home/classlens /var/data /tmp/classlens_uploads \
    && su -s /bin/sh classlens -c 'python -c "import numpy as np; from assessor.features import _yin_pitch; pitch = _yin_pitch(np.sin(2 * np.pi * 220 * np.arange(16000, dtype=np.float32) / 16000), 16000, frame_length=1024); assert len(pitch) and abs(float(np.median(pitch)) - 220) < 5; print(\"NumPy pitch smoke test passed\")"'
COPY docker-entrypoint.sh /usr/local/bin/classlens-entrypoint
RUN chmod 755 /usr/local/bin/classlens-entrypoint
ENTRYPOINT ["/usr/local/bin/classlens-entrypoint"]

EXPOSE 10000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0"]
