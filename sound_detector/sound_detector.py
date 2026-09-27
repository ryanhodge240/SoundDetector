#!/usr/bin/env python3

import argparse
import csv
import json
import logging
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
from tflite_runtime.interpreter import Interpreter


MODEL_URL = 'https://huggingface.co/thelou1s/yamnet/resolve/main/lite-model_yamnet_tflite_1.tflite'
LABELS_URL = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 15600
HOP_SAMPLES = WINDOW_SAMPLES // 2
MODEL_FILENAME = 'yamnet.tflite'
LABELS_FILENAME = 'yamnet_labels.csv'

TARGET_LABELS = {
    'baby cry, infant cry': 'baby_crying',
    'crying, sobbing': 'baby_crying',
    'doorbell': 'doorbell',
    'ding-dong': 'doorbell',
    'knock': 'door_knock',
    'fire alarm': 'fire_alarm',
    'smoke detector, smoke alarm': 'fire_alarm',
    'bark': 'dog_barking',
    'dog': 'dog_barking',
    'telephone bell ringing': 'phone_ringing',
    'ringtone': 'phone_ringing',
}


def read_config(path: str) -> dict[str, Any]:
    with open(path, encoding='utf-8') as config_file:
        return json.load(config_file)


def download_if_missing(path: Path, url: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f'{path.suffix}.download')
    logging.info('Downloading %s', path.name)
    urllib.request.urlretrieve(url, temporary_path)
    temporary_path.replace(path)


def load_labels(path: Path) -> list[str]:
    with path.open(encoding='utf-8', newline='') as labels_file:
        rows = list(csv.reader(labels_file))

    if not rows:
        raise RuntimeError('The YAMNet label map is empty.')

    label_column = 2 if len(rows[0]) >= 3 and rows[0][2] == 'display_name' else 0
    return [row[label_column] for row in rows[1:] if len(row) > label_column]


def normalize_device(value: Any) -> Any:
    if value in (None, ''):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def event_payload(event_name: str, confidence: float) -> dict[str, Any]:
    return {
        'event': event_name,
        'confidence': round(float(confidence), 4),
        'timestamp': int(time.time() * 1000),
    }


def send_event(server_url: str, server_token: str, payload: dict[str, Any]) -> None:
    endpoint = server_url.rstrip('/')
    if endpoint.endswith('/api/triggers'):
        endpoint = f'{endpoint}/{payload["event"]}'

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            **({'Authorization': f'Bearer {server_token}'} if server_token else {}),
        },
        method='POST',
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f'server returned HTTP {response.status}')
            logging.info('Sent %s event to the Beacon server', payload['event'])
    except urllib.error.URLError as error:
        logging.warning('Unable to send sound event: %s', error)


def classify_window(interpreter: Interpreter, labels: list[str], samples: np.ndarray) -> list[tuple[str, float]]:
    input_details = interpreter.get_input_details()[0]
    input_shape = input_details['shape']
    input_length = int(input_shape[-1])
    window = samples[:input_length]
    if len(window) < input_length:
        window = np.pad(window, (0, input_length - len(window)))

    input_data = window.astype(np.float32)
    if input_details['dtype'] != np.float32:
        input_data = input_data.astype(input_details['dtype'])
    if len(input_shape) > 1:
        input_data = input_data.reshape(tuple(input_shape))
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    scores = np.asarray(output).reshape(-1, len(labels)).mean(axis=0)
    results = []
    for index, label in enumerate(labels):
        event_name = TARGET_LABELS.get(label.strip().lower())
        if event_name:
            results.append((event_name, float(scores[index])))
    return results


def run(config: dict[str, Any]) -> None:
    data_directory = Path('/data')
    model_path = data_directory / MODEL_FILENAME
    labels_path = data_directory / LABELS_FILENAME
    download_if_missing(model_path, MODEL_URL)
    download_if_missing(labels_path, LABELS_URL)

    labels = load_labels(labels_path)
    interpreter = Interpreter(model_path=str(model_path))
    input_details = interpreter.get_input_details()[0]
    if input_details.get('shape_signature', input_details['shape'])[-1] == -1:
        interpreter.resize_tensor_input(input_details['index'], [WINDOW_SAMPLES])
    interpreter.allocate_tensors()

    threshold = float(config.get('confidence_threshold', 0.35))
    cooldown_seconds = int(config.get('cooldown_seconds', 10))
    server_url = str(config.get('server_url') or '').strip()
    server_token = str(config.get('server_token') or '').strip()
    dry_run = bool(config.get('dry_run', True))
    device = normalize_device(config.get('audio_input_device'))
    version = os.environ.get('SOUND_DETECTOR_VERSION') or 'unknown'
    log_level = str(config.get('log_level') or 'info').upper()
    logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))
    last_events: dict[str, float] = {}

    logging.info(
        'Configuration: sample_rate=%s window_ms=%.0f hop_ms=%.0f threshold=%.2f cooldown_seconds=%s '
        'dry_run=%s server_configured=%s',
        SAMPLE_RATE,
        WINDOW_SAMPLES / SAMPLE_RATE * 1000,
        HOP_SAMPLES / SAMPLE_RATE * 1000,
        threshold,
        cooldown_seconds,
        dry_run,
        bool(server_url),
    )

    logging.info('Available audio devices:')
    for index, device_info in enumerate(sd.query_devices()):
        if device_info['max_input_channels'] > 0:
            logging.info('  %s: %s', index, device_info['name'])

    logging.info('Starting microphone capture at %s Hz with 50%% overlapping windows', SAMPLE_RATE)
    audio_buffer = np.empty(0, dtype=np.float32)
    window_count = 0
    next_summary = time.monotonic() + 10
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        blocksize=HOP_SAMPLES,
        device=device,
    ) as stream:
        stream_device = stream.device
        if isinstance(stream_device, (tuple, list)):
            stream_device = stream_device[0]
        stream_device_info = sd.query_devices(stream_device)
        logging.info('Sound Detector version=%s', version)
        logging.info('Using audio input device %s: %s', stream_device, stream_device_info['name'])

        while True:
            audio, status = stream.read(HOP_SAMPLES)
            if status:
                logging.warning('Audio input status: %s', status)

            audio_buffer = np.concatenate((audio_buffer, np.asarray(audio[:, 0], dtype=np.float32)))
            while len(audio_buffer) >= WINDOW_SAMPLES:
                samples = audio_buffer[:WINDOW_SAMPLES]
                audio_buffer = audio_buffer[HOP_SAMPLES:]
                window_count += 1
                rms = math.sqrt(float(np.mean(np.square(samples))))
                if rms < 0.001:
                    logging.debug('Window %s skipped: rms=%.5f below gate', window_count, rms)
                    continue

                inference_started = time.monotonic()
                scores = classify_window(interpreter, labels, samples)
                inference_ms = (time.monotonic() - inference_started) * 1000
                score_by_event: dict[str, float] = {}
                for event_name, confidence in scores:
                    score_by_event[event_name] = max(score_by_event.get(event_name, 0), confidence)
                logging.debug(
                    'Window %s: rms=%.5f inference_ms=%.1f scores=%s',
                    window_count,
                    rms,
                    inference_ms,
                    ', '.join(f'{name}={score:.3f}' for name, score in score_by_event.items()),
                )

                for event_name, confidence in score_by_event.items():
                    if confidence < threshold:
                        continue
                    now = time.monotonic()
                    if now - last_events.get(event_name, 0) < cooldown_seconds:
                        logging.debug('Suppressed %s: cooldown is active', event_name)
                        continue

                    last_events[event_name] = now
                    payload = event_payload(event_name, confidence)
                    logging.info('Detected %s (confidence %.2f, rms %.5f)', event_name, confidence, rms)
                    if server_url and not dry_run:
                        send_event(server_url, server_token, payload)

            if time.monotonic() >= next_summary:
                logging.info('Audio summary: analyzed_windows=%s buffered_samples=%s', window_count, len(audio_buffer))
                next_summary = time.monotonic() + 10


def main() -> None:
    parser = argparse.ArgumentParser(description='Detect household sounds with YAMNet.')
    parser.add_argument('--config', default=os.environ.get('SOUND_DETECTOR_CONFIG', '/data/options.json'))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run(read_config(arguments.config))


if __name__ == '__main__':
    main()
