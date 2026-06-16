# Marmot API Reference

The Marmot server is a small Flask application (default port `5000`). You can override the port with the `MARMOT_PORT` environment variable.

## POST /connect

The primary endpoint used by the client for both voice and text interactions.

**Input**

- `multipart/form-data` with `file=@recording.wav` → audio path (16 kHz mono WAV recommended)
- `application/json` body: `{ "text": "your question" }` → text-only path

**Response**

```json
{
  "transcription": "what the user said (from audio or the text you sent)",
  "text": "Here is the answer...",
  "audio": "UklGRiQ...base64 wav..."   // or null
}
```

- `transcription`: The recognized (or provided) user input.
- `text`: Marmot's final reply.
- `audio`: Base64-encoded WAV file of the spoken reply (when TTS is configured), otherwise `null`.

## Other endpoints

- `GET /` — Simple self-contained status dashboard. Shows live health data, services, context/memory stats, cron jobs, and the mascot image. Auto-refreshes every 5 seconds. Includes a one-click "Reset Context" button.
- `GET /health` — Returns server status, current context size, last message time, pending proactive count, detection server config, plus service models/voices. The dashboard consumes this endpoint.
- `POST /reset` — Clears the rolling conversation history (also extracts persistent memory before clearing).
- `GET /poll` — Internal endpoint used by the interactive client. Supports optional long-poll via `?wait=` (capped at 10s by the server).  
  Returns `{"action":"initiate","message":{...}}` when the server has a queued proactive message, or `{"action":"noop"}`.
- `POST /inject` — Manually queue a proactive (server-initiated) message.  
  Body example: `{ "text": "Hey, the build finished.", "speak": true }`  
  Useful for testing or from external scripts / future background logic.
- `POST /detect` — Accepts an image and returns a list of detected object labels (e.g. `{"objects": ["person", "laptop"]}`). Used by the client for camera-based human presence checks before speaking proactive messages. Requires an external YOLO detection server (configured via `DETECTION_BASE_URL`).

## POST /detect

Accepts an image (via multipart upload) and returns the object labels detected by a configured external YOLO-based image recognition server.

**Input**

- `multipart/form-data` with `image=@snapshot.jpg` **or** `file=@snapshot.jpg`

**Response**

```json
{
  "objects": ["person", "laptop", "chair"]
}
```

- Returns a flat list of class names (strings) as reported by the YOLO model (e.g. COCO classes: "person", "cat", "bottle", etc.).
- The interactive client calls this (after capturing a webcam frame) to decide whether to speak a proactive message. It considers a human present if `"person"` or `"human"` appears in the list.
- The Marmot server itself does not run YOLO — it proxies the image to the external detection server specified by `DETECTION_BASE_URL` (set during first-run setup or in `server/code/config.json`). If no detection server is configured, `/detect` returns an error.

### Server-initiated (proactive) messages

- When the interactive client is idle (not recording, not already speaking a response, not mid-request), its background poller will pick up queued messages via `/poll`.
- The proactive text is appended to the conversation context on the server (so follow-up hotkey responses continue the thread naturally).
- Audio is auto-played but **never overlaps** previous audio (playback is serialized on the client).
- If the client is busy (recording, in the middle of a response, or audio still playing) when a proactive arrives from the server, it is buffered in a small local queue (max 4) on the client and played automatically as soon as the client becomes unblocked. The server has already committed these messages to conversation context at delivery time.
- **Human presence gating**: Before speaking any proactive message (fresh or from the local buffer), the client captures a frame from the webcam and calls `POST /detect`. The message is only spoken if the result contains `"person"` or `"human"`. This prevents the agent from talking to an empty room.
- Client-side backoff: After 5 minutes with no "user interaction" (pressing the record hotkey or a successful human detection via camera), the client backs off its `/poll` and camera checks to approximately once per minute.
- On the client, proactives are printed with a `(proactive)` label, copied to the clipboard, and spoken (when audio is present). Camera access (and `opencv-python`) is required on the client machine for the human-presence feature.

## Testing with curl

Here are handy `curl` commands for testing the server directly (especially useful during development or when the Python client isn't available).

**Health check**
```bash
curl -s http://localhost:5000/health | jq
```

**Send a text query** (recommended for quick tests)
```bash
curl -s -X POST http://localhost:5000/connect \
  -H "Content-Type: application/json" \
  -d '{"text": "what is the current hostname and kernel version?"}' | jq
```

**Send text and print only the response** (clean output)
```bash
curl -s -X POST http://localhost:5000/connect \
  -H "Content-Type: application/json" \
  -d '{"text": "list the top 5 processes by memory usage"}' | jq -r '.text'
```

**Send text and save the spoken audio reply**
```bash
curl -s -X POST http://localhost:5000/connect \
  -H "Content-Type: application/json" \
  -d '{"text": "tell me a short joke about marmots"}' \
  | jq -r '.audio' | base64 -d > /tmp/marmot_reply.wav \
  && echo "Saved audio to /tmp/marmot_reply.wav"
```

**Send an audio file** (multipart upload)
```bash
curl -s -X POST http://localhost:5000/connect \
  -F "file=@/path/to/your/recording.wav" | jq -r '.text'
```

**Reset conversation context**
```bash
curl -s -X POST http://localhost:5000/reset | jq
```

**Queue a proactive message (server initiates)**
```bash
curl -s -X POST http://localhost:5000/inject \
  -H "Content-Type: application/json" \
  -d '{"text": "The long-running job you started earlier just completed successfully.", "speak": true}' | jq
```
The next time the interactive client is idle it will receive it via its `/poll` background loop, print it with a `(proactive)` label, copy to clipboard, and speak the audio (if TTS is enabled). The message is also recorded in the rolling conversation context.

**Check for humans/objects in an image (via the /detect endpoint)**
```bash
curl -s -X POST http://localhost:5000/detect \
  -F "image=@/tmp/webcam.jpg" | jq
```

Returns something like `{"objects": ["person", "keyboard", "cup"]}`. The interactive client uses this internally (after grabbing a webcam frame) to decide whether to speak proactive messages. The server forwards the image to the external YOLO detection server configured at startup (`DETECTION_BASE_URL`).

> **Tip**: Replace `localhost:5000` with your server's address if it's running elsewhere.  
> `jq` is recommended for readable JSON (install with `sudo apt install jq` or equivalent).  
> Useful fields: `.transcription` (what the user said), `.text` (Marmot's reply), `.audio` (base64 wav or null).

Example to show both sides:
```bash
curl -s -X POST http://localhost:5000/connect \
  -H "Content-Type: application/json" \
  -d '{"text": "list files in ~"}' | jq '{transcription, text}'
```

## Cron / scheduled prompts

The server can load scheduled jobs from `server/code/cron.json` (copy `cron.json.example` to start).

- JSON array of objects: `{ "schedule": "...", "prompt": "...", "enabled": true, "comment": "optional note (ignored)" }`. Only "schedule" + "prompt" are required. `"enabled"` defaults to true.
- Standard 5-field cron (supports `*`, lists, ranges, steps like `*/15`).
- The prompt is executed internally against the LLM (tools + full context work). Result is sent through `queue_proactive_message()`.
- In-memory last execution time per job prevents duplicate firing for the same time slot.
- See `GET /health` (`"cron_jobs"` and `"cron"` array) for status.
- Useful for hourly briefings, morning check-ins, background monitoring alerts, etc.
