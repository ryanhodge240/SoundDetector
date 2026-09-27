# Sound Detector Home Assistant App

This Home Assistant OS app listens to a USB microphone, classifies short audio
windows with YAMNet, and optionally sends detected sound events to a server over
the Raspberry Pi's Tailscale connection.

The app sends event metadata only. It does not upload or store raw audio.

## Supported events

- `baby_cry`
- `doorbell`
- `knock`
- `fire_alarm`
- `smoke_detector`

## Install

This directory is an app repository. Make it available from a Git repository,
then add that repository under **Settings -> Apps -> App store -> Repositories**.
Install **Sound Detector**, configure it, and start it.

The app currently supports `aarch64`, which is the architecture used by a
64-bit Home Assistant OS installation on a supported Raspberry Pi.

## Configuration

- `server_url`: Optional HTTP endpoint that accepts event JSON. Leave empty for
  local dry-run logging.
- `server_token`: Optional bearer token sent to `server_url`.
- `confidence_threshold`: Minimum model confidence, from `0` to `1`.
- `cooldown_seconds`: Minimum time between events of the same type.
- `dry_run`: Log detections without sending them, even if `server_url` is set.
- `audio_input_device`: Optional PortAudio device name or index. Leave empty to
  use the default USB microphone.
- `log_level`: Set to `debug` to log model scores for every analyzed window.

The server URL should use the server's Tailscale IP or MagicDNS hostname, for
example `http://100.100.100.100:3001/api/audio/events`. Tailscale must already
be connected on both devices.

The YAMNet model and label map are downloaded on the first start and stored in
the app's persistent `/data` directory. The Pi therefore needs internet access
for the initial model download.

## Troubleshooting

Check the app logs for the available audio devices. If the default device is
not the USB microphone, set `audio_input_device` to the displayed device name.

The model is intentionally downloaded at runtime rather than bundled into the
repository because it is much larger than the app source.

The detector analyzes overlapping windows. At `debug` level, logs include the
RMS level, model scores for the target sounds, inference duration, and audio
input overflow warnings.

## Limit Switch app

This repository also contains a **Limit Switch Detector** app for a normally-open
switch connected between Raspberry Pi **BCM GPIO17** (physical pin 11) and GND.
The app enables the GPIO's internal pull-up, so an open switch reports state `0`
and a pressed switch reports state `1`. It reports the initial state at startup
and then reports only state changes, with configurable debounce filtering.

The limit-switch app uses the same `server_url`, `server_token`, and `dry_run`
settings as the sound detector. Its event payload has this form:

```json
{
  "event": "limit_switch",
  "state": 1,
  "gpio": 17,
  "timestamp": 1720000000000
}
```
