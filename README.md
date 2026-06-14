# marmot-harness
![screenshot](/images/marmot-harness.jpg "A marmot in a climbing harness")

A local voice-first AI agent with tool use.  
Push-to-talk (or text) → STT (whisper.cpp) → LLM with tools (ReAct/multi-turn) → TTS spoken response.

## Features

- Hold **Right Option/Alt** to record (cross-platform, same as spark-dictate)
- Audio sent to local Flask server → whisper.cpp for transcription
- Full conversation context (rolling, ~150k token default)
- LLM (any OpenAI-compatible endpoint) with tool calling support
- Built-in `run_terminal` tool (bash commands executed on host, results fed back to LLM)
- Multiple tool turns per utterance (agent can explore, run commands, iterate before final answer)
- Final response spoken via local TTS (Kokoro-style `/audio/speech`) and returned
- Text response printed + copied to clipboard
- `-m "text here"` client flag for quick text-only queries (no mic, exits after one response)
- Simple Flask server with `/connect`, `/health`, `/reset`, `/poll`, `/inject`
- Server can initiate conversations (client polls `/poll` when idle; proactives play audio without overlapping prior speech)
- Auto-clears conversation context after 10 hours of inactivity (configurable)

## Architecture

```
Client (mic hotkey or -m)          Marmot Server (Flask)          External Services
─────────────────────────          ────────────────────          ─────────────────
record (16kHz + gain + pad)  ──▶   /connect (audio or text)
                                   │
                                   ├──▶ whisper.cpp /v1/audio/transcriptions
                                   │         (STT)
                                   │
                                   ├──▶ LLM /v1/chat/completions
                                   │      (tools + full context, multi-turn)
                                   │         run_terminal tool
                                   │
                                   └──▶ TTS /v1/audio/speech
                                         (spoken reply)
                                   │
◀── JSON {transcription, text, audio:base64} ◀──── response
```

Client plays audio, prints text, copies to clipboard.

## Requirements

**Client machine:**
- Python 3.10+
- Microphone (for hotkey mode)
- Same clipboard tools as spark-dictate (`wl-clipboard` or `xclip` on Linux)

**Servers (designed for NVIDIA CUDA):**
- whisper.cpp server (CUDA) on port 8025 (or your choice)
- OpenAI-compatible LLM server (vLLM, llama.cpp server, Ollama OpenAI compat, etc.)
- Kokoro FastAPI or other TTS exposing `/v1/audio/speech` (optional but recommended)

**LLM Endpoint:**
- designed for local LLM endpoint using vLLM
- openai compatible
- tested with MiniMax M2.7 on a 2x DGX Spark Cluster

## Quick Setup

### 1. Whisper + LLM + TTS (external)

Install whisper.cpp and start the server:

```bash
./server/whisper.cpp/start_whisper_cuda.sh
```

#### Optional: Text-to-Speech (Kokoro)

To enable spoken responses, run the Kokoro FastAPI TTS server (GPU):

```bash
docker run -d --gpus all \
  -p 8880:8880 \
  --name kokoro-tts \
  ghcr.io/remsky/kokoro-fastapi-gpu:latest
```

During marmot server first-run setup you will be prompted for the whisper.cpp URL (e.g. `192.168.1.45:8025`), LLM base URL + model, and optional TTS base URL (e.g. `http://192.168.1.45:8880/v1`). Supported voices include `af_heart` (default), `am_adam`, etc. (see Kokoro docs).

### 2. Marmot Server

```bash
cd server
./start_server.sh
```

First run will interactively ask for:
- whisper.cpp URL
- LLM base URL + model
- TTS base URL + voice (optional)

Settings saved to `server/code/config.json`.

Edit `SYSTEM_PROMPT` or `MAX_CONTEXT_TOKENS` in the json if desired.

### 3. Client

```bash
cd client
./start_client.sh
```

First run prompts for Marmot server address (e.g. `localhost:5000` or remote IP).

## Usage

### Voice Mode (default)

1. Run client
2. Hold **Right Option (⌥)** / **Right Alt**
3. Speak your request (can be complex, involve tools)
4. Release → server transcribes, thinks (with tools if needed), speaks final answer
5. Client shows what it heard (transcription), then Marmot's reply, plays audio, and copies the reply to clipboard

Examples you can say:
- "what processes are using the most memory?"
- "list the files in my home directory and summarize the largest ones"
- "run a quick disk space check and tell me if anything is over 80% full"
- "read my todo list and suggest the next three priorities"

### Text / Testing Mode

```bash
./start_client.sh -m "show me the top 5 processes by cpu"
./start_client.sh -m "what is the current hostname and kernel?"
```

The client will show:

```
🐹 Sending message: ...
🗣️  You: <your message>
🐹 Marmot: <reply>
```

(then play audio + copy reply to clipboard)

Useful for debugging without touching the mic, and for scripting.

### API

`POST /connect`

- `multipart/form-data` with `file=@recording.wav` → audio path
- `application/json` `{ "text": "your question" }` → text path

Returns:

```json
{
  "transcription": "what the user said (from audio or the text you sent)",
  "text": "Here is the answer...",
  "audio": "UklGRiQ...base64 wav..."   // or null
}
```

Other endpoints:
- `GET /health`
- `POST /reset` (clears conversation history)
- `GET /poll` — client background poll (supports `?wait=2.5` for long-poll style). Returns `{"action":"initiate","message":{text,audio,id}}` when the server has something queued, or `{"action":"noop"}`.
- `POST /inject` — queue a proactive message for delivery on the next client poll (for testing or future schedulers/background logic). Body: `{"text": "Hey, the build finished.", "speak": true}`

Server-initiated (proactive) messages:
- When the interactive client is idle (not recording, not already speaking a response, not mid-request), its background poller will pick up queued messages.
- The proactive text is appended to conversation context on the server (so follow-up hotkey responses continue the thread naturally).
- Audio is auto-played but **never overlaps** previous audio (playback is serialized).
- If the client is busy (recording, in the middle of a response, or audio still playing) when a proactive arrives from the server, it is buffered in a small local queue (max 4) on the client and played automatically as soon as the client becomes unblocked. The server already committed these messages to conversation context at delivery time.
- Text is printed with a `(proactive)` label, copied to clipboard, and spoken if TTS audio was provided.

### Testing with curl

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

> **Tip**: Replace `localhost:5000` with your server's address if it's running elsewhere.  
> `jq` is recommended for readable JSON (install with `sudo apt install jq` or equivalent).  
> Useful fields: `.transcription` (what the user said), `.text` (Marmot's reply), `.audio` (base64 wav or null).

Example to show both sides:
```bash
curl -s -X POST http://localhost:5000/connect \
  -H "Content-Type: application/json" \
  -d '{"text": "list files in ~"}' | jq '{transcription, text}'
```

## Tool Use (ReAct style)

The server implements a loop similar to the example in the project prompt:

1. Send current context + new user turn + tools schema to LLM
2. If LLM returns `tool_calls`, execute them locally (`run_terminal` runs the bash via subprocess with timeout)
3. Feed results back as `role: tool` messages
4. Repeat until LLM produces a plain `content` response (no tool_calls)
5. That final text becomes the spoken + returned answer

Only the original user utterance and the final assistant text are appended to the persistent rolling context (internal tool scratchpad is discarded per turn to control token usage).

## Configuration

All server settings live in `server/code/config.json` (created on first run).

Key fields:
- `MAX_CONTEXT_TOKENS`: 150000 (approx char/3 estimate; trims oldest turns)
- `SYSTEM_PROMPT`: customize agent personality
- `TOOL_TIMEOUT`: seconds per `run_terminal` call (default 30)
- `MAX_TOOL_TURNS`: safety cap on ReAct iterations (default 8)
- `CONTEXT_TIMEOUT_HOURS`: hours of inactivity before automatically clearing conversation context (default: 10)

Client: `client/code/client_config.json` (only server address + gain).

## Security Note

The `run_terminal` tool gives the LLM real shell access to your machine. Only run this against trusted local models and review what it does. Consider running the whole stack in a container or VM for experiments.


## License

Unlicense (public domain) — same as spark-dictate.
