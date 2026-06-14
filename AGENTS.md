# AGENTS.md

Minimal guidance for AI coding agents working on marmot-harness.

## Project Overview
Local voice-first AI agent with tool use.
- Client records audio (hotkey) or takes text (`-m`), sends to server.
- Server: STT (whisper.cpp) → LLM (OpenAI-compatible + ReAct tools) → optional TTS.
- All conversation state, persistent memory, and context management lives on the **server**.
- Client is intentionally thin (audio + UI + background polling).

## Key Locations
- `client/code/client.py` — Main client logic (recording, hotkey via pynput, background proactive poller, playback).
- `server/code/server.py` — Almost all the interesting code (Flask routes, LLM loop, tools, context trimming, proactive queue, memory extraction).
- `docs/API.md` — API reference (endpoints, curl examples, proactive behavior).
- `server/code/config.json` and `client/code/client_config.json`.
- `agent-data/` — Tool working directory + persistent memory.txt.

## Running
```bash
# Server (first run prompts for whisper/LLM/TTS URLs)
cd server && ./start_server.sh

# Client (interactive hotkey mode)
cd client && ./start_client.sh

# One-shot text
cd client && ./start_client.sh -m "your question here"
```

Requires external services running:
- whisper.cpp (STT)
- OpenAI-compatible LLM server
- Optional: Kokoro-style TTS at `/v1/audio/speech`

## Architecture Notes (Important)
- **Proactive / server-initiated conversations**: Client polls `GET /poll` (with optional `?wait=`) when idle. Server can queue messages via `queue_proactive_message()` or the `POST /inject` endpoint.
- Proactive messages are appended to server `conversation_history` the moment they are delivered over `/poll`.
- Client maintains a small local buffer (`pending_proactive_queue`) for messages that arrive while busy (recording / sending / playing audio). They play automatically when the client becomes free. Audio never overlaps thanks to `playback_lock`.
- `/poll` request logs are deliberately suppressed on the server (see `QuietPollRequestHandler`) to keep the console clean.
- The `run_terminal` tool has real shell access on the host — be careful.
- Context lives only on the server. The client does not maintain history.

## Development Tips
- Most new features belong on the server.
- Use `POST /inject` heavily when testing proactive behavior.
- Important `print()` statements (user turns, Marmot replies, proactive queuing/delivery, tool calls) are the primary way to observe behavior.
- Keep changes minimal and focused — this is a small, simple codebase.

See `README.md` for user-facing documentation and `docs/API.md` for endpoint details.