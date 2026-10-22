FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    HF_HOME=/home/classlens/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg tesseract-ocr libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY assessor ./assessor
COPY web ./web
COPY server.py config.json ./

RUN useradd --create-home --shell /usr/sbin/nologin classlens \
    && mkdir -p /home/classlens/.cache /tmp/classlens_uploads \
    && chown -R classlens:classlens /app /home/classlens /tmp/classlens_uploads
USER classlens

EXPOSE 10000
CMD ["python", "server.py"]
