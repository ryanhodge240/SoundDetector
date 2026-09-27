FROM python:3.11-slim

ARG BUILD_VERSION

ENV SOUND_DETECTOR_VERSION="${BUILD_VERSION}"

LABEL \
    io.hass.version="${BUILD_VERSION}" \
    io.hass.type="app" \
    io.hass.arch="aarch64"

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates libasound2 libasound2-plugins libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    'numpy<2' \
    'sounddevice==0.5.1' \
    'tflite-runtime==2.14.0'

WORKDIR /app
COPY sound_detector.py /app/sound_detector.py

CMD ["python", "/app/sound_detector.py", "--config", "/data/options.json"]
