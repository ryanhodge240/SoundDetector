FROM python:3.11-slim-bookworm

ARG BUILD_VERSION

ENV VISUAL_DETECTOR_VERSION="${BUILD_VERSION}"

LABEL \
    io.hass.version="${BUILD_VERSION}" \
    io.hass.type="app" \
    io.hass.arch="aarch64"

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    'numpy<2' \
    'opencv-python-headless==4.10.0.84' \
    'tflite-runtime==2.14.0'

WORKDIR /app
COPY visual_detector.py /app/visual_detector.py

CMD ["python", "/app/visual_detector.py", "--config", "/data/options.json"]
