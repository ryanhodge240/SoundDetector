# Sound Detector Home Assistant App

This Home Assistant OS app listens to a USB microphone, classifies short audio
windows with YAMNet, and optionally sends detected sound events to a server over
the Raspberry Pi's Tailscale connection.

The app sends event metadata only. It does not upload or store raw audio.

## Supported events

- `baby_crying`
- `doorbell`
- `door_knock`
- `fire_alarm`
- Smoke detector sounds are sent as `fire_alarm`.

## Install

This directory is an app repository. Make it available from a Git repository,
then add that repository under **Settings -> Apps -> App store -> Repositories**.
Install **Sound Detector**, configure it, and start it.

The app currently supports `aarch64`, which is the architecture used by a
64-bit Home Assistant OS installation on a supported Raspberry Pi.

## Configuration

- `server_url`: Optional HTTP endpoint that accepts event JSON. Leave empty for
  local dry-run logging.
- `server_token`: Optional bearer token sent to `server_url`. Set this to the
  OwlHacks `TRIGGER_API_KEY`.
- `confidence_threshold`: Minimum model confidence, from `0` to `1`.
- `cooldown_seconds`: Minimum time between events of the same type.
- `dry_run`: Log detections without sending them, even if `server_url` is set.
- `audio_input_device`: Optional PortAudio device name or index. Leave empty to
  use the default USB microphone.
- `log_level`: Set to `debug` to log model scores for every analyzed window.

The server URL should use the server's Tailscale IP or MagicDNS hostname and
end in `/api/triggers`, for example
`http://100.100.100.100:3001/api/triggers`. The detector app appends the
detected trigger name to that path. Tailscale must already be connected on both
devices.

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

## Visual Detector app

The **Visual Detector** app uses a USB webcam and a lightweight TensorFlow Lite
object-detection model. It detects people and vehicles locally without storing
or uploading camera images. It processes one frame every two seconds by default
and requires repeated detections before reporting a state change.

The default camera is `/dev/video0`. The camera should be fixed in position and
have a clear, reasonably lit view of the area being monitored. A Raspberry Pi 4
or newer is recommended. The app uses the same `server_url`, `server_token`,
and `dry_run` settings as the other apps.

Configuration options include:

- `camera_device`: USB camera device, normally `/dev/video0`.
- `camera_width` and `camera_height`: requested capture resolution.
- `detection_interval_seconds`: delay between inferences.
- `confidence_threshold`: minimum model confidence.
- `confirm_frames`: detections required before reporting arrival.
- `clear_frames`: missed detections required before reporting departure.
- `save_debug_frame`: optionally save the first captured frame as
  `/data/debug_frame.jpg` for camera troubleshooting.

The app reports an initial `cleared` state for both object types, then reports
only subsequent state changes. Example event:

```json
{
  "event": "visual_detection",
  "object": "person",
  "state": "detected",
  "confidence": 0.87,
  "timestamp": 1720000000000
}
```

For troubleshooting, set `log_level` to `debug`. The logs then include camera
frame statistics, model input/output tensor details, the top raw class IDs and
scores, filtered detections, and inference timing. The optional debug frame is
stored locally in the app's persistent `/data` directory and is never sent to
the server.
