#!/usr/bin/env python3

import argparse
import json
import logging
import os
import time
import urllib.error
import urllib.request
from datetime import timedelta
from typing import Any

import gpiod
from gpiod.line import Bias, Direction, Edge, Value


def read_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as config_file:
        return json.load(config_file)


def event_payload(state: int, gpio_line: int) -> dict[str, Any]:
    return {
        "event": "limit_switch",
        "state": state,
        "gpio": gpio_line,
        "timestamp": int(time.time() * 1000),
    }


def send_event(server_url: str, server_token: str, payload: dict[str, Any]) -> None:
    request = urllib.request.Request(
        server_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **(
                {"Authorization": f"Bearer {server_token}"}
                if server_token
                else {}
            ),
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f"server returned HTTP {response.status}")
            logging.info("Sent limit switch state=%s", payload["state"])
    except (urllib.error.URLError, OSError) as error:
        logging.warning("Unable to send limit switch event: %s", error)


def line_state(request: gpiod.LineRequest, gpio_line: int) -> int:
    # With the pull-up enabled, an open switch is electrically high (0) and
    # a closed switch pulls the line low (1).
    return int(request.get_value(gpio_line) == Value.INACTIVE)


def report_state(
    state: int,
    gpio_line: int,
    server_url: str,
    server_token: str,
    dry_run: bool,
) -> None:
    payload = event_payload(state, gpio_line)
    logging.info("Limit switch state=%s", state)
    if server_url and not dry_run:
        send_event(server_url, server_token, payload)


def run(config: dict[str, Any]) -> None:
    chip = str(config.get("gpio_chip") or "/dev/gpiochip0")
    gpio_line = int(config.get("gpio_line", 17))
    debounce_ms = int(config.get("debounce_ms", 50))
    server_url = str(config.get("server_url") or "").strip()
    server_token = str(config.get("server_token") or "").strip()
    dry_run = bool(config.get("dry_run", True))
    version = os.environ.get("LIMIT_SWITCH_VERSION") or "unknown"

    logging.info(
        "Limit Switch Detector version=%s chip=%s line=%s debounce_ms=%s "
        "dry_run=%s server_configured=%s",
        version,
        chip,
        gpio_line,
        debounce_ms,
        dry_run,
        bool(server_url),
    )

    settings = gpiod.LineSettings(
        direction=Direction.INPUT,
        edge_detection=Edge.BOTH,
        bias=Bias.PULL_UP,
        debounce_period=timedelta(milliseconds=debounce_ms),
    )

    with gpiod.request_lines(
        chip,
        consumer="homeassistant-limit-switch",
        config={gpio_line: settings},
    ) as request:
        state = line_state(request, gpio_line)
        report_state(state, gpio_line, server_url, server_token, dry_run)

        while True:
            if not request.wait_edge_events(timeout=timedelta(seconds=60)):
                logging.debug("No limit switch change in the last 60 seconds")
                continue

            events = request.read_edge_events()
            for event in events:
                new_state = line_state(request, gpio_line)
                if new_state == state:
                    continue

                state = new_state
                logging.debug("GPIO edge=%s", event.event_type.name)
                report_state(state, gpio_line, server_url, server_token, dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor a Raspberry Pi limit switch.")
    parser.add_argument(
        "--config",
        default=os.environ.get("LIMIT_SWITCH_CONFIG", "/data/options.json"),
    )
    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = read_config(arguments.config)
    log_level = str(config.get("log_level") or "info").upper()
    logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))
    run(config)


if __name__ == "__main__":
    main()
