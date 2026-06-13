#!/usr/bin/env python3
"""
Marmot Agent Server

Flask orchestrator:
  audio/text input -> whisper.cpp STT (if audio) -> LLM (OpenAI-comp. w/ tools, multi-turn ReAct)
  -> final text response -> TTS (Kokoro-style) -> return transcription + text + audio (base64)

Rolling conversation context with:
  - configurable max tokens
  - auto-clear after N hours of inactivity (default 10h)
"""

import os
import json
import tempfile
import subprocess
import base64
import datetime
from flask import Flask, request, jsonify
import requests

# ========================= CONFIG =========================
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

def _fix_url(u):
    u = (u or "").strip()
    if u and not u.startswith(("http://", "https://")):
        u = "http://" + u
    return u.rstrip("/")

def load_config():
    cfg = {
        "WHISPER_BASE_URL": None,
        "WHISPER_MODEL": "whisper-large-v3",
        "LLM_BASE_URL": None,
        "LLM_MODEL": "your-model-name",
        "TTS_BASE_URL": None,
        "TTS_MODEL": "kokoro",
        "TTS_VOICE": "af_heart",
        "MAX_CONTEXT_TOKENS": 150000,
        "SYSTEM_PROMPT": "You are Marmot, a helpful local AI agent running on the user's machine. You have tools to inspect and control the Linux system. Use tools when needed to answer accurately. Be concise in final answers. Always think step-by-step before calling tools.",
        "TOOLS_ENABLED": True,
        "TOOL_TIMEOUT": 30,
        "MAX_TOOL_TURNS": 8,
        "CONTEXT_TIMEOUT_HOURS": 10,
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                loaded = json.load(f)
            cfg.update({k: v for k, v in loaded.items() if k in cfg})
        except Exception as e:
            print("Warning: could not load config:", e)

    needs_save = False
    for key in ("WHISPER_BASE_URL", "LLM_BASE_URL", "TTS_BASE_URL"):
        if cfg.get(key):
            fixed = _fix_url(cfg[key])
            if fixed != cfg[key]:
                cfg[key] = fixed
                needs_save = True

    # Interactive first-run setup (mirrors spark-dictate style)
    if not cfg.get("WHISPER_BASE_URL"):
        val = input("\nEnter whisper.cpp server (e.g. 192.168.1.45:8025 or http://localhost:8025) [default: http://localhost:8025]: ").strip()
        if not val:
            val = "http://localhost:8025"
        cfg["WHISPER_BASE_URL"] = _fix_url(val)
        needs_save = True

    if not cfg.get("LLM_BASE_URL"):
        val = input("\nEnter LLM base URL (OpenAI-compatible, e.g. http://10.12.0.50:8000/v1) [default: http://localhost:8000/v1]: ").strip()
        if not val:
            val = "http://localhost:8000/v1"
        cfg["LLM_BASE_URL"] = _fix_url(val)
        needs_save = True

    if not cfg.get("LLM_MODEL") or cfg.get("LLM_MODEL") == "your-model-name":
        val = input("Enter LLM model name [required, e.g. Qwen/Qwen2.5-7B-Instruct]: ").strip()
        if val:
            cfg["LLM_MODEL"] = val
            needs_save = True

    if not cfg.get("TTS_BASE_URL"):
        val = input("\nEnter TTS base URL (OpenAI-comp /audio/speech e.g. http://192.168.1.45:8880/v1) [Enter to skip TTS]: ").strip()
        if val:
            cfg["TTS_BASE_URL"] = _fix_url(val)
            needs_save = True

    if cfg.get("TTS_BASE_URL"):
        if not cfg.get("TTS_MODEL"):
            val = input("TTS model name [default: kokoro]: ").strip() or "kokoro"
            cfg["TTS_MODEL"] = val
            needs_save = True
        if not cfg.get("TTS_VOICE"):
            val = input("TTS voice [default: af_heart]: ").strip() or "af_heart"
            cfg["TTS_VOICE"] = val
            needs_save = True

    if needs_save:
        try:
            keys = ["WHISPER_BASE_URL", "WHISPER_MODEL", "LLM_BASE_URL", "LLM_MODEL",
                    "TTS_BASE_URL", "TTS_MODEL", "TTS_VOICE", "MAX_CONTEXT_TOKENS",
                    "SYSTEM_PROMPT", "TOOLS_ENABLED", "TOOL_TIMEOUT", "MAX_TOOL_TURNS",
                    "CONTEXT_TIMEOUT_HOURS"]
            with open(CONFIG_PATH, "w") as f:
                json.dump({k: cfg[k] for k in keys if k in cfg}, f, indent=2)
            print(f"✅ Saved config to {CONFIG_PATH}")
        except Exception as e:
            print("⚠️  Could not save config:", e)
    return cfg

config = load_config()

WHISPER_BASE_URL = config["WHISPER_BASE_URL"]
WHISPER_MODEL = config.get("WHISPER_MODEL", "whisper-large-v3")
LLM_BASE_URL = config["LLM_BASE_URL"]
LLM_MODEL = config["LLM_MODEL"]
TTS_BASE_URL = config.get("TTS_BASE_URL")
TTS_MODEL = config.get("TTS_MODEL", "kokoro")
TTS_VOICE = config.get("TTS_VOICE", "af_heart")
MAX_CONTEXT_TOKENS = int(config.get("MAX_CONTEXT_TOKENS", 150000))
SYSTEM_PROMPT = config.get("SYSTEM_PROMPT", "You are a helpful agent.")
TOOLS_ENABLED = bool(config.get("TOOLS_ENABLED", True))
TOOL_TIMEOUT = int(config.get("TOOL_TIMEOUT", 30))
MAX_TOOL_TURNS = int(config.get("MAX_TOOL_TURNS", 8))
CONTEXT_TIMEOUT_HOURS = int(config.get("CONTEXT_TIMEOUT_HOURS", 10))

last_message_time = None  # Used for auto-clearing context after long inactivity

print("🐹 Marmot Agent Server ready")
print(f"   Whisper: {WHISPER_BASE_URL}  model={WHISPER_MODEL}")
print(f"   LLM:     {LLM_MODEL} @ {LLM_BASE_URL}")
print(f"   TTS:     {TTS_MODEL}/{TTS_VOICE} @ {TTS_BASE_URL or '(disabled)'}")
print(f"   Context: ~{MAX_CONTEXT_TOKENS} tokens max (rolling)")
print(f"   Tools:   {'on' if TOOLS_ENABLED else 'off'}   tool-timeout={TOOL_TIMEOUT}s")
print(f"   Inactivity timeout: {CONTEXT_TIMEOUT_HOURS}h → auto-clear context")
print()

# ====================== TOOLS ======================
AGENT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "agent-data"))
os.makedirs(AGENT_DATA_DIR, exist_ok=True)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_terminal",
        "description": "Execute a Linux bash command. Returns exit code + stdout + stderr. Use to explore files, run commands, check processes, edit via echo/cat etc. Prefer non-destructive commands when possible.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run, e.g. 'ls -la', 'ps aux | head', 'cat README.md'"}
            },
            "required": ["command"]
        }
    }
}]

def execute_run_terminal(command: str) -> str:
    if not command or not command.strip():
        return "Error: empty command"
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=TOOL_TIMEOUT,
            cwd=AGENT_DATA_DIR,
            env={**os.environ}
        )
        parts = [f"Exit code: {result.returncode}"]
        if result.stdout:
            out = result.stdout
            if len(out) > 7000:
                out = out[:7000] + "\n[truncated]"
            parts.append("STDOUT:\n" + out)
        if result.stderr:
            err = result.stderr
            if len(err) > 4000:
                err = err[:4000] + "\n[truncated]"
            parts.append("STDERR:\n" + err)
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"Error: timed out after {TOOL_TIMEOUT}s"
    except Exception as e:
        return f"Error: {str(e)}"

def execute_tool(tool_call: dict) -> str:
    fn = tool_call.get("function", {})
    name = fn.get("name", "")
    try:
        args = json.loads(fn.get("arguments", "{}"))
    except Exception:
        args = {}
    if name == "run_terminal":
        return execute_run_terminal(args.get("command", ""))
    return f"Error: unknown tool {name}"

# ====================== ROLLING CONTEXT ======================
conversation_history = []  # user + assistant messages (tool internals kept only per-turn)

def estimate_tokens(x) -> int:
    if x is None:
        return 0
    try:
        s = json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x)
    except Exception:
        s = str(x)
    return max(1, len(s) // 3)  # conservative ~3 chars/token for headroom

def trim_conversation_history():
    global conversation_history
    if not conversation_history:
        return
    sys = {"role": "system", "content": SYSTEM_PROMPT}
    cur = [sys] + conversation_history
    while len(cur) > 1 and estimate_tokens(cur) > MAX_CONTEXT_TOKENS:
        cur.pop(1)  # drop oldest after system
    conversation_history = cur[1:]

# ====================== LLM + MULTI-TURN TOOLS ======================
def process_with_llm(user_text: str) -> str:
    """Core agent loop. Adds user turn, runs LLM allowing tool_calls until final message, returns text. Persists only user+final assistant."""
    global conversation_history
    trim_conversation_history()
    conversation_history.append({"role": "user", "content": user_text})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history
    turn = 0
    final_text = ""

    while turn < MAX_TOOL_TURNS:
        turn += 1
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.6,
        }
        if TOOLS_ENABLED and TOOLS:
            payload["tools"] = TOOLS
            payload["tool_choice"] = "auto"

        try:
            r = requests.post(f"{LLM_BASE_URL}/chat/completions", json=payload, timeout=300)
            if r.status_code != 200:
                print(f"LLM HTTP {r.status_code}: {r.text[:250]}")
                final_text = f"(LLM error {r.status_code})"
                break
            data = r.json()
            msg = data.get("choices", [{}])[0].get("message", {})
            messages.append(msg)

            if msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    print(f"  🔧 {tc.get('function', {}).get('name', 'tool')}")
                    out = execute_tool(tc)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": out
                    })
                continue
            else:
                final_text = (msg.get("content") or "").strip()
                conversation_history.append({"role": "assistant", "content": final_text})
                break
        except Exception as e:
            print("LLM exception:", e)
            final_text = f"(LLM failure: {e})"
            break

    trim_conversation_history()
    return final_text

# ====================== TTS ======================
def generate_tts_audio(text: str) -> bytes:
    if not text or not TTS_BASE_URL:
        return b""
    try:
        payload = {
            "model": TTS_MODEL,
            "input": text.strip(),
            "voice": TTS_VOICE,
            "response_format": "wav"
        }
        print("🔊 TTS synthesis...")
        r = requests.post(f"{TTS_BASE_URL}/audio/speech", json=payload, timeout=180)
        if r.status_code == 200 and r.content:
            return r.content
        print(f"TTS {r.status_code}: {r.text[:150] if r.text else ''}")
    except Exception as e:
        print("TTS error:", e)
    return b""

# ====================== STT (whisper.cpp) ======================
def transcribe_audio(audio_file) -> str:
    """FileStorage -> text via whisper.cpp server"""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    try:
        audio_file.save(tmp_path)
        files = {"file": open(tmp_path, "rb")}
        data = {
            "model": WHISPER_MODEL,
            "language": "en",
            "temperature": "0.0",
            "response_format": "json"
        }
        print("📤 Transcribing via whisper.cpp...")
        r = requests.post(f"{WHISPER_BASE_URL}/v1/audio/transcriptions", files=files, data=data, timeout=120)
        if r.status_code == 200:
            txt = r.json().get("text", "").strip()
            print(f"🗣️  Heard: {txt[:120]}{'...' if len(txt) > 120 else ''}")
            return txt
        print(f"Whisper {r.status_code}: {r.text[:150]}")
    except Exception as e:
        print("STT error:", e)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return ""

# ====================== FLASK ======================
app = Flask(__name__)

@app.route("/connect", methods=["POST"])
def connect():
    user_text = None
    if request.files and "file" in request.files:
        f = request.files["file"]
        if f and f.filename:
            user_text = transcribe_audio(f)

    if not user_text:
        if request.is_json:
            user_text = (request.json or {}).get("text", "")
        else:
            user_text = request.form.get("text", "")
        user_text = (user_text or "").strip()

    if not user_text:
        return jsonify({"error": "Send audio file or text"}), 400

    # === Inactivity timeout check: clear context if > CONTEXT_TIMEOUT_HOURS since last message ===
    global last_message_time
    now = datetime.datetime.now()
    if last_message_time is not None:
        delta = now - last_message_time
        if delta.total_seconds() > (CONTEXT_TIMEOUT_HOURS * 3600):
            print(f"⏰ No messages for >{CONTEXT_TIMEOUT_HOURS} hours — clearing conversation context")
            conversation_history.clear()
    last_message_time = now

    print(f"\n👤 User: {user_text}")
    final = process_with_llm(user_text)
    print(f"🐹 Marmot: {final[:160]}{'...' if len(final) > 160 else ''}")

    audio_b = generate_tts_audio(final)
    audio_b64 = base64.b64encode(audio_b).decode("ascii") if audio_b else None

    return jsonify({
        "transcription": user_text,
        "text": final,
        "audio": audio_b64
    })

@app.route("/health", methods=["GET"])
def health():
    now = datetime.datetime.now()
    seconds_since_last = None
    if last_message_time is not None:
        seconds_since_last = int((now - last_message_time).total_seconds())

    return jsonify({
        "ok": True,
        "whisper": WHISPER_BASE_URL,
        "llm": LLM_MODEL,
        "tts": bool(TTS_BASE_URL),
        "turns": len([m for m in conversation_history if m["role"] in ("user", "assistant")]),
        "context_timeout_hours": CONTEXT_TIMEOUT_HOURS,
        "last_message_at": last_message_time.isoformat() if last_message_time else None,
        "seconds_since_last_message": seconds_since_last
    })

@app.route("/reset", methods=["POST"])
def reset():
    global conversation_history, last_message_time
    conversation_history = []
    last_message_time = None
    return jsonify({"ok": True, "msg": "context cleared"})

if __name__ == "__main__":
    port = int(os.environ.get("MARMOT_PORT", 5000))
    print(f"🌐 http://0.0.0.0:{port}   /connect  /health  /reset")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
