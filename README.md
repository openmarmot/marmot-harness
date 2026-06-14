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
- Simple Flask server with `/connect`, `/health`, `/reset`, `/poll`, `/inject` (see [docs/API.md](docs/API.md))
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

## API

See [docs/API.md](docs/API.md) for the complete API reference, including:

- `POST /connect` (primary endpoint)
- `GET /health`, `POST /reset`
- `GET /poll` (client proactive polling, with long-poll support)
- `POST /inject` (manually queue server-initiated messages)
- Full details on server-initiated (proactive) conversations
- Many `curl` examples for testing the server directly

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
