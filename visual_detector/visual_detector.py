#!/usr/bin/env python3

import argparse
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter


MODEL_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/models/tflite/"
    "task_library/object_detection/android/"
    "lite-model_ssd_mobilenet_v1_1_metadata_2.tflite"
)
MODEL_FILENAME = "ssd_mobilenet_v1.tflite"
PERSON_CLASS = 1
VEHICLE_CLASSES = {3, 4, 6, 8}  # car, motorcycle, bus, truck


def read_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as config_file:
        return json.load(config_file)


def download_if_missing(path: Path, url: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.download")
    logging.info("Downloading object detection model")
    urllib.request.urlretrieve(url, temporary_path)
    temporary_path.replace(path)


def normalize_camera_device(value: Any) -> Any:
    if value in (None, ""):
        return "/dev/video0"
    value = str(value)
    try:
        return int(value)
    except ValueError:
        return value


def event_payload(object_name: str, state: str, confidence: float | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event": "visual_detection",
        "object": object_name,
        "state": state,
        "timestamp": int(time.time() * 1000),
    }
    if confidence is not None:
        payload["confidence"] = round(float(confidence), 4)
    return payload


def send_event(server_url: str, server_token: str, payload: dict[str, Any]) -> None:
    request = urllib.request.Request(
        server_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {server_token}"} if server_token else {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f"server returned HTTP {response.status}")
            logging.info("Sent %s %s event", payload["object"], payload["state"])
    except (urllib.error.URLError, OSError) as error:
        logging.warning("Unable to send visual event: %s", error)


def dequantize(values: np.ndarray, details: dict[str, Any]) -> np.ndarray:
    scale, zero_point = details.get("quantization", (0.0, 0))
    if scale:
        return (values.astype(np.float32) - zero_point) * scale
    return values.astype(np.float32)


def tensor_input(interpreter: Interpreter, frame: np.ndarray) -> None:
    details = interpreter.get_input_details()[0]
    height = int(details["shape"][1])
    width = int(details["shape"][2])
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)

    if details["dtype"] == np.float32:
        image /= 255.0
    elif np.issubdtype(details["dtype"], np.integer):
        scale, zero_point = details.get("quantization", (0.0, 0))
        if scale:
            image = image / scale + zero_point
        limits = np.iinfo(details["dtype"])
        image = np.clip(image, limits.min, limits.max)

    interpreter.set_tensor(details["index"], np.expand_dims(image.astype(details["dtype"]), axis=0))


def detection_outputs(
    interpreter: Interpreter,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, float]]]:
    vectors: list[np.ndarray] = []
    boxes: np.ndarray | None = None
    for details in interpreter.get_output_details():
        values = np.squeeze(dequantize(interpreter.get_tensor(details["index"]), details))
        if values.ndim == 2 and values.shape[-1] == 4:
            boxes = values
        elif values.size > 1:
            vectors.append(values.reshape(-1))

    if boxes is None or len(vectors) < 2:
        raise RuntimeError("Object detection model outputs are not in SSD format")

    # Standard SSD exports contain scores, class IDs, and a scalar count.
    # Ignore the scalar count and identify scores by their [0, 1] range.
    vectors = [vector for vector in vectors if len(vector) > 1]
    score_candidates = [
        vector for vector in vectors
        if float(np.min(vector)) >= -0.01 and float(np.max(vector)) <= 1.01
    ]
    if not score_candidates:
        raise RuntimeError("Could not identify score output from object model")
    scores = max(score_candidates, key=len)
    class_candidates = [vector for vector in vectors if vector is not scores]
    classes = min(class_candidates, key=lambda vector: float(np.mean(np.abs(vector - np.round(vector)))))
    count = min(len(boxes), len(classes), len(scores))
    raw_detections = sorted(
        (
            int(round(float(class_id))),
            float(score),
        )
        for class_id, score in zip(classes[:count], scores[:count])
    )
    raw_detections = sorted(raw_detections, key=lambda item: item[1], reverse=True)[:10]
    return boxes[:count], classes[:count], scores[:count], raw_detections


def detect(
    interpreter: Interpreter,
    frame: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float], list[tuple[int, float]]]:
    tensor_input(interpreter, frame)
    interpreter.invoke()
    _, classes, scores, raw_detections = detection_outputs(interpreter)
    detected: dict[str, float] = {}
    for class_id, score in zip(classes, scores):
        if score < threshold:
            continue
        class_id = int(round(float(class_id)))
        if class_id == PERSON_CLASS:
            detected["person"] = max(detected.get("person", 0.0), float(score))
        elif class_id in VEHICLE_CLASSES:
            detected["vehicle"] = max(detected.get("vehicle", 0.0), float(score))
    return detected, raw_detections


def log_model_outputs(interpreter: Interpreter) -> None:
    for details in interpreter.get_output_details():
        logging.debug(
            "Model output: index=%s name=%s shape=%s dtype=%s quantization=%s",
            details["index"],
            details.get("name"),
            details["shape"],
            details["dtype"].__name__,
            details.get("quantization"),
        )


def report_state(
    object_name: str,
    state: str,
    confidence: float | None,
    server_url: str,
    server_token: str,
    dry_run: bool,
) -> None:
    payload = event_payload(object_name, state, confidence)
    confidence_text = f" confidence={confidence:.2f}" if confidence is not None else ""
    logging.info("%s state=%s%s", object_name, state, confidence_text)
    if server_url and not dry_run:
        send_event(server_url, server_token, payload)


def run(config: dict[str, Any]) -> None:
    model_path = Path("/data") / MODEL_FILENAME
    download_if_missing(model_path, MODEL_URL)

    camera_device = normalize_camera_device(config.get("camera_device"))
    camera_width = int(config.get("camera_width", 640))
    camera_height = int(config.get("camera_height", 480))
    detection_interval = float(config.get("detection_interval_seconds", 2))
    threshold = float(config.get("confidence_threshold", 0.65))
    confirm_frames = int(config.get("confirm_frames", 2))
    clear_frames = int(config.get("clear_frames", 5))
    server_url = str(config.get("server_url") or "").strip()
    server_token = str(config.get("server_token") or "").strip()
    dry_run = bool(config.get("dry_run", True))
    save_debug_frame = bool(config.get("save_debug_frame", False))
    version = os.environ.get("VISUAL_DETECTOR_VERSION") or "unknown"

    interpreter = Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    input_shape = input_details["shape"]
    log_model_outputs(interpreter)
    logging.info(
        "Visual Detector version=%s camera=%s requested_resolution=%sx%s "
        "model_input=%sx%s interval_seconds=%s threshold=%.2f dry_run=%s "
        "server_configured=%s save_debug_frame=%s",
        version, camera_device, camera_width, camera_height,
        input_shape[2], input_shape[1], detection_interval, threshold,
        dry_run, bool(server_url), save_debug_frame,
    )
    logging.debug(
        "Model input: index=%s name=%s shape=%s dtype=%s quantization=%s",
        input_details["index"], input_details.get("name"), input_details["shape"],
        input_details["dtype"].__name__, input_details.get("quantization"),
    )

    camera = cv2.VideoCapture(camera_device)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    if not camera.isOpened():
        raise RuntimeError(f"Unable to open camera {camera_device}")
    logging.info(
        "Using camera %s at %sx%s",
        camera_device,
        int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    objects = ("person", "vehicle")
    present = {object_name: False for object_name in objects}
    positive_counts = {object_name: 0 for object_name in objects}
    negative_counts = {object_name: 0 for object_name in objects}
    last_confidence: dict[str, float | None] = {object_name: None for object_name in objects}
    for object_name in objects:
        report_state(object_name, "cleared", None, server_url, server_token, dry_run)

    next_detection = 0.0
    inference_count = 0
    total_inference_seconds = 0.0
    debug_frame_saved = False
    try:
        while True:
            now = time.monotonic()
            if now < next_detection:
                time.sleep(min(next_detection - now, 0.1))
                continue
            next_detection = now + detection_interval
            success, frame = camera.read()
            if not success:
                logging.warning("Unable to read a frame from camera")
                time.sleep(1)
                continue

            if save_debug_frame and not debug_frame_saved:
                debug_path = "/data/debug_frame.jpg"
                if cv2.imwrite(debug_path, frame):
                    logging.info("Saved debug camera frame to %s", debug_path)
                    debug_frame_saved = True
                else:
                    logging.warning("Unable to save debug camera frame to %s", debug_path)

            logging.debug(
                "Frame %s: shape=%s dtype=%s min=%.1f max=%.1f mean=%.1f",
                inference_count + 1,
                frame.shape,
                frame.dtype,
                float(np.min(frame)),
                float(np.max(frame)),
                float(np.mean(frame)),
            )

            started = time.monotonic()
            detected, raw_detections = detect(interpreter, frame, threshold)
            inference_seconds = time.monotonic() - started
            inference_count += 1
            total_inference_seconds += inference_seconds
            logging.debug(
                "Frame %s inference_ms=%.1f raw_top=%s filtered=%s",
                inference_count,
                inference_seconds * 1000,
                raw_detections,
                detected,
            )

            for object_name in objects:
                if object_name in detected:
                    positive_counts[object_name] += 1
                    negative_counts[object_name] = 0
                    last_confidence[object_name] = detected[object_name]
                    if not present[object_name] and positive_counts[object_name] >= confirm_frames:
                        present[object_name] = True
                        report_state(object_name, "detected", last_confidence[object_name], server_url, server_token, dry_run)
                else:
                    negative_counts[object_name] += 1
                    positive_counts[object_name] = 0
                    if present[object_name] and negative_counts[object_name] >= clear_frames:
                        present[object_name] = False
                        report_state(object_name, "cleared", None, server_url, server_token, dry_run)

            if inference_count == 1 or inference_count % 30 == 0:
                logging.info(
                    "Detection summary: frames=%s average_inference_ms=%.1f person=%s vehicle=%s",
                    inference_count,
                    total_inference_seconds / inference_count * 1000,
                    present["person"], present["vehicle"],
                )
    finally:
        camera.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect people and vehicles from a webcam.")
    parser.add_argument("--config", default=os.environ.get("VISUAL_DETECTOR_CONFIG", "/data/options.json"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = read_config(arguments.config)
    log_level = str(config.get("log_level") or "info").upper()
    logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))
    run(config)


if __name__ == "__main__":
    main()
