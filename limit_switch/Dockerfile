FROM python:3.11-slim-bookworm

ARG BUILD_VERSION

ENV LIMIT_SWITCH_VERSION="${BUILD_VERSION}"

LABEL \
    io.hass.version="${BUILD_VERSION}" \
    io.hass.type="app" \
    io.hass.arch="aarch64"

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates libgpiod2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir 'gpiod==2.2.1'

WORKDIR /app
COPY limit_switch.py /app/limit_switch.py

CMD ["python", "/app/limit_switch.py", "--config", "/data/options.json"]
