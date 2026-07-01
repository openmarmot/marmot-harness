#!/usr/bin/env python3
"""
Marmot Agent Server

Flask orchestrator:
  audio/text input -> whisper.cpp STT (if audio) -> LLM (OpenAI-comp. w/ tools, multi-turn ReAct)
  -> final text response -> TTS (Kokoro-style) -> return transcription + text + audio (base64)

Rolling conversation context with:
  - configurable max tokens
  - auto-clear after N hours of inactivity (default 10h)
  - persistent memory (≤~100 lines) extracted by asking the LLM before each full clear
  - LLM compaction: oldest turns are summarized into compact notes when nearing token limit
    (simple oldest-turn dropping is kept only as emergency fallback)
"""

import os
import io
import json
import wave
import tempfile
import subprocess
import base64
import datetime
import threading
import time
import uuid
from collections import deque
from flask import Flask, request, jsonify
import requests
from werkzeug.serving import WSGIRequestHandler

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
        "SYSTEM_PROMPT": "You are Marmot, a helpful local AI agent running on the user's machine. You have tools to inspect and control the Linux system and to search the web for current events or facts not on this machine. Use tools when needed to answer accurately. Be concise in final answers. Always think step-by-step before calling tools.",
        "TOOLS_ENABLED": True,
        "TOOL_TIMEOUT": 30,
        "MAX_TOOL_TURNS": 15,
        "CONTEXT_TIMEOUT_HOURS": 10,
        "DETECTION_BASE_URL": None,
        "WEB_SEARCH_ENABLED": True,
        "BRAVE_SEARCH_API_KEY": None,
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                loaded = json.load(f)
            cfg.update({k: v for k, v in loaded.items() if k in cfg})
        except Exception as e:
            print("Warning: could not load config:", e)

    needs_save = False
    for key in ("WHISPER_BASE_URL", "LLM_BASE_URL", "TTS_BASE_URL", "DETECTION_BASE_URL"):
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

    if not cfg.get("DETECTION_BASE_URL"):
        val = input("\nEnter YOLO detection server base URL (e.g. http://localhost:8007) [Enter to skip]: ").strip()
        if val:
            cfg["DETECTION_BASE_URL"] = _fix_url(val)
            needs_save = True

    if needs_save:
        try:
            keys = ["WHISPER_BASE_URL", "WHISPER_MODEL", "LLM_BASE_URL", "LLM_MODEL",
                    "TTS_BASE_URL", "TTS_MODEL", "TTS_VOICE", "MAX_CONTEXT_TOKENS",
                    "SYSTEM_PROMPT", "TOOLS_ENABLED", "TOOL_TIMEOUT", "MAX_TOOL_TURNS",
                    "CONTEXT_TIMEOUT_HOURS", "DETECTION_BASE_URL",
                    "WEB_SEARCH_ENABLED", "BRAVE_SEARCH_API_KEY"]
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
DETECTION_BASE_URL = config.get("DETECTION_BASE_URL")
if DETECTION_BASE_URL:
    DETECTION_BASE_URL = _fix_url(DETECTION_BASE_URL)
MAX_CONTEXT_TOKENS = int(config.get("MAX_CONTEXT_TOKENS", 150000))
SYSTEM_PROMPT = config.get("SYSTEM_PROMPT", "You are a helpful agent.")
TOOLS_ENABLED = bool(config.get("TOOLS_ENABLED", True))
TOOL_TIMEOUT = int(config.get("TOOL_TIMEOUT", 30))
MAX_TOOL_TURNS = int(config.get("MAX_TOOL_TURNS", 8))
CONTEXT_TIMEOUT_HOURS = int(config.get("CONTEXT_TIMEOUT_HOURS", 10))
WEB_SEARCH_ENABLED = bool(config.get("WEB_SEARCH_ENABLED", True))
BRAVE_SEARCH_API_KEY = (config.get("BRAVE_SEARCH_API_KEY") or "").strip() or None

last_message_time = None  # Used for auto-clearing context after long inactivity
persistent_memory = ""  # durable notes persisted across conversation clears (bounded ~100 lines)

# ====================== SIMPLE CRON JOBS ======================
# Cron jobs are loaded once at startup from server/code/cron.json (optional; copy cron.json.example to get started).
# Format (JSON array of simple objects). Only "schedule" + "prompt" are required.
# Supported fields per job: "schedule", "prompt", optional "id", "enabled" (bool, defaults true), "comment" (ignored).
# Extra/unknown fields are ignored.
# [
#   {
#     "schedule": "0 * * * *",
#     "prompt": "Give a short hourly status note.",
#     "enabled": true,
#     "comment": "This runs every hour on the hour. Feel free to change the text."
#   }
# ]
# Standard 5-field cron (min hour dom month dow). Supports *, ranges, lists, and steps (e.g. */5, 1-10/2).
# Each job's prompt is sent (internally) to the LLM with full tool access (ReAct). The final response text
# is queued via queue_proactive_message(). Last execution time per job is persisted in cron_state.json and
# used to avoid duplicate runs for the same time slot (deduped at minute granularity).

CRON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cron.json")
CRON_STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cron_state.json")
cron_jobs = []  # list of {"id": str, "schedule": str, "prompt": str, "enabled": bool, "last_run": datetime|None}

def _load_cron_state() -> dict:
    """Return {job_id: isoformat str} from disk."""
    if not os.path.exists(CRON_STATE_PATH):
        return {}
    try:
        with open(CRON_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print("Warning: could not load cron_state.json:", e)
        return {}

def _save_cron_last_run(job_id: str, when: datetime.datetime) -> None:
    state = _load_cron_state()
    state[job_id] = when.isoformat()
    try:
        with open(CRON_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.write("\n")
    except Exception as e:
        print("Warning: could not save cron_state.json:", e)

def _cron_field_values(field: str, min_val: int, max_val: int) -> set:
    """Expand cron field like '*', '5', '1,3', '*/15', '9-17', '1-10/2' into set of ints."""
    values = set()
    if not field or field == "*":
        return set(range(min_val, max_val + 1))
    for part in field.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if "/" in part:
            base, st = part.split("/", 1)
            try:
                step = max(1, int(st))
            except Exception:
                step = 1
            part = base
        if part == "*":
            start, end = min_val, max_val
        elif "-" in part:
            try:
                a, b = part.split("-", 1)
                start, end = int(a), int(b)
            except Exception:
                continue
        else:
            try:
                start = end = int(part)
            except Exception:
                continue
        for v in range(start, end + 1, step):
            if min_val <= v <= max_val:
                values.add(v)
    return values

def cron_due(schedule: str, dt: datetime.datetime) -> bool:
    """True if 5-field cron schedule matches dt (uses local time, minute resolution).

    Day matching follows classic cron "OR" rule: when both dom and dow are restricted (not *),
    the job runs if *either* the day-of-month *or* the day-of-week matches.
    """
    try:
        parts = [p.strip() for p in (schedule or "").split()]
        if len(parts) != 5:
            return False
        minute_f, hour_f, dom_f, month_f, dow_f = parts

        if dt.minute not in _cron_field_values(minute_f, 0, 59):
            return False
        if dt.hour not in _cron_field_values(hour_f, 0, 23):
            return False
        if dt.month not in _cron_field_values(month_f, 1, 12):
            return False

        doms = _cron_field_values(dom_f, 1, 31)
        dom_match = dt.day in doms
        dom_restricted = (dom_f != "*")

        # DOW: cron 0/7=Sun, 1=Mon..6=Sat; datetime.weekday Mon=0..Sun=6
        dows_raw = _cron_field_values(dow_f, 0, 7)
        dows = {0 if d == 7 else d for d in dows_raw}
        py_wd = dt.weekday()
        cron_wd = (py_wd + 1) % 7
        dow_match = (cron_wd in dows) if dows else True
        dow_restricted = (dow_f != "*")

        # Classic cron: when *both* dom and dow are restricted (neither is "*"), match if either matches (OR).
        # Otherwise require the (effective) matches (unrestricted sides always match because their set is full range).
        if dom_restricted and dow_restricted:
            day_ok = dom_match or dow_match
        else:
            day_ok = dom_match and dow_match
        if not day_ok:
            return False
        return True
    except Exception:
        return False

def load_cron_jobs():
    global cron_jobs
    cron_jobs = []
    if not os.path.exists(CRON_PATH):
        if os.path.exists(CRON_PATH + ".example"):
            print("   (Cron enabled: copy cron.json.example -> cron.json to schedule prompt jobs)")
        return
    try:
        with open(CRON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print("Warning: cron.json must be a JSON array of {schedule, prompt} objects")
            return
        saved_runs = _load_cron_state()
        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                continue
            # "comment" (and any other extra keys) are allowed for human notes and are deliberately ignored.
            comment = entry.get("comment")  # optional human-readable note only

            # "enabled" defaults to true so old configs keep working. Accepts bool or common string forms.
            enabled = entry.get("enabled", True)
            if isinstance(enabled, str):
                enabled = enabled.lower() not in ("0", "false", "no", "off", "disabled")
            enabled = bool(enabled)

            sched = str(entry.get("schedule", "")).strip()
            prompt = str(entry.get("prompt", "")).strip()
            if not sched or not prompt:
                continue
            jid = str(entry.get("id") or f"{sched}:{i}")
            last_run = None
            saved = saved_runs.get(jid)
            if saved:
                try:
                    last_run = datetime.datetime.fromisoformat(saved)
                except Exception:
                    pass
            cron_jobs.append({
                "id": jid,
                "schedule": sched,
                "prompt": prompt,
                "enabled": enabled,
                "last_run": last_run
            })
        if cron_jobs:
            enabled_jobs = [j for j in cron_jobs if j.get("enabled", True)]
            schedules = ", ".join(j["schedule"] for j in enabled_jobs)
            total = len(cron_jobs)
            if len(enabled_jobs) < total:
                print(f"⏰ Loaded {len(enabled_jobs)}/{total} cron job(s) ({total - len(enabled_jobs)} disabled): {schedules}")
            else:
                print(f"⏰ Loaded {total} cron job(s): {schedules}")
    except Exception as e:
        print("Warning: could not load cron.json:", e)

# ====================== PROACTIVE INITIATION (server -> client) ======================
# Client polls /poll when idle. Server can queue messages it wants to deliver unprompted.
# Items are dicts: {"id": str, "text": str, "audio": base64 or None, "created_at": iso}
pending_initiations = deque()
pending_lock = threading.Lock()
initiation_ready = threading.Condition(pending_lock)  # allows efficient long-poll wakeups
MAX_PENDING_INITIATIONS = 5
MAX_INITIATION_AGE_SECONDS = 3600  # 1 hour

# Forward stubs (real implementations defined after ROLLING CONTEXT)
def _get_memory_messages() -> list:
    return []

# ====================== TOOLS ======================
AGENT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "agent-data"))
os.makedirs(AGENT_DATA_DIR, exist_ok=True)

# Tool calls get their own working directory so files the agent creates (via run_terminal etc.)
# are separated from Marmot's own data like memory.txt.
TOOL_CALLS_DIR = os.path.join(AGENT_DATA_DIR, "tool-calls")
os.makedirs(TOOL_CALLS_DIR, exist_ok=True)

MEMORY_PATH = os.path.join(AGENT_DATA_DIR, "memory.txt")

_RUN_TERMINAL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_terminal",
        "description": "Execute a Linux bash command (cwd is the dedicated tool-calls workspace under agent-data/tool-calls/). Returns exit code + stdout + stderr. Use to explore files, run commands, check processes, edit via echo/cat etc. Prefer non-destructive commands when possible. Created files stay isolated from Marmot's own data (e.g. memory.txt).",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run, e.g. 'ls -la', 'ps aux | head', 'cat README.md'"}
            },
            "required": ["command"]
        }
    }
}

_WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web via Brave Search for current events, news, documentation, or facts not available on this machine. Returns titles, snippets, and URLs. Summarize findings in your final answer; do not read URLs aloud.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query, e.g. 'Python 3.13 release date' or 'weather San Francisco'"},
                "max_results": {"type": "integer", "description": "Number of results to return (1-10, default 5)"}
            },
            "required": ["query"]
        }
    }
}

TOOLS = [_RUN_TERMINAL_TOOL]
if WEB_SEARCH_ENABLED and BRAVE_SEARCH_API_KEY:
    TOOLS.append(_WEB_SEARCH_TOOL)

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
            cwd=TOOL_CALLS_DIR,
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

def execute_web_search(query: str, max_results: int = 5) -> str:
    if not BRAVE_SEARCH_API_KEY:
        return "Error: web search not configured (set BRAVE_SEARCH_API_KEY in config.json)"
    q = (query or "").strip()
    if not q:
        return "Error: empty query"
    try:
        n = int(max_results)
    except (TypeError, ValueError):
        n = 5
    n = max(1, min(n, 10))
    try:
        r = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": BRAVE_SEARCH_API_KEY,
            },
            params={"q": q, "count": n},
            timeout=TOOL_TIMEOUT,
        )
        if r.status_code != 200:
            detail = (r.text or "")[:500]
            return f"Error: Brave Search HTTP {r.status_code}" + (f" — {detail}" if detail else "")
        data = r.json()
        results = (data.get("web") or {}).get("results") or []
        if not results:
            return f"No results for: {q}"
        parts = [f"Query: {q}", f"Results ({len(results)}):"]
        for i, item in enumerate(results, 1):
            title = (item.get("title") or "(no title)").strip()
            url = (item.get("url") or "").strip()
            desc = (item.get("description") or "").strip()
            block = f"{i}. {title}"
            if desc:
                block += f"\n   {desc}"
            if url:
                block += f"\n   {url}"
            parts.append(block)
        out = "\n\n".join(parts)
        if len(out) > 7000:
            out = out[:7000] + "\n[truncated]"
        return out
    except requests.Timeout:
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
    if name == "web_search":
        return execute_web_search(args.get("query", ""), args.get("max_results", 5))
    return f"Error: unknown tool {name}"

# ====================== ROLLING CONTEXT ======================
# conversation_history holds only the current session's user + final assistant turns.
# It is managed by trim_conversation_history which *prefers* LLM-generated compaction
# summaries over raw deletion when we approach the token limit.
conversation_history = []  # user + assistant messages (tool internals ephemeral per turn)

def estimate_tokens(x) -> int:
    if x is None:
        return 0
    try:
        s = json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x)
    except Exception:
        s = str(x)
    return max(1, len(s) // 3)  # conservative ~3 chars/token for headroom

def trim_conversation_history():
    """Ensure conversation_history (+ protected memory messages) stays under MAX_CONTEXT_TOKENS.

    Preferred path: LLM compaction of oldest turns into a single dense summary message that
    is inserted at the front of the remaining history. This preserves session coherence far
    better than raw deletion.

    Dumb per-turn popping is retained only as an emergency fallback when:
    - We've already performed the allowed number of LLM compactions in this call, or
    - The summarizer returns "nothing significant", or
    - There aren't enough turns to justify a summary.

    The system prompt + persistent memory messages are always protected (never compacted).
    """
    global conversation_history
    if not conversation_history:
        return

    prefix = [{"role": "system", "content": SYSTEM_PROMPT}] + _get_memory_messages()
    pfx = len(prefix)
    max_compactions = 2  # limit expensive LLM calls per trim invocation
    compactions = 0

    while True:
        cur = prefix + conversation_history
        if len(cur) <= pfx or estimate_tokens(cur) <= MAX_CONTEXT_TOKENS:
            break

        # Preferred: try to compact a chunk of the oldest raw turns via LLM
        if compactions < max_compactions:
            total = len(conversation_history)
            # Compact a worthwhile chunk: at least 3 turns, at most ~10 or 1/3 of history
            chunk = min(10, max(3, total // 3))
            if total >= 3:
                to_compact = conversation_history[:chunk]
                summary = summarize_for_compaction(to_compact)
                # Drop the raw prefix we just summarized
                conversation_history = conversation_history[chunk:]
                low = (summary or "").lower()
                if summary and "no significant earlier context" not in low:
                    compacted_msg = {
                        "role": "assistant",
                        "content": "[Compacted summary of earlier turns in this conversation]\n" + summary.strip()
                    }
                    conversation_history.insert(0, compacted_msg)
                    print(f"🗜️  Compacted {chunk} older turns into a summary note")
                    compactions += 1
                    continue  # check budget again

        # Emergency dumb fallback: bluntly drop the single oldest conversation turn.
        # When the front is a freshly created compaction summary we just paid an LLM call for,
        # prefer to drop an older raw turn behind it instead (protect the value of the compaction).
        if conversation_history:
            if "Compacted summary" in conversation_history[0].get("content", "") and len(conversation_history) > 1:
                del conversation_history[1]
            else:
                conversation_history.pop(0)

# ====================== PERSISTENT MEMORY ======================
# Small durable memory (~100 lines max) extracted from conversation before it is cleared.
# Injected as an extra system message at the start of new conversations.

def _load_persistent_memory():
    global persistent_memory
    if not os.path.exists(MEMORY_PATH):
        persistent_memory = ""
        return
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            persistent_memory = f.read()
    except Exception as e:
        print("Warning: could not load memory:", e)
        persistent_memory = ""

def _save_persistent_memory():
    try:
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            f.write(persistent_memory)
    except Exception as e:
        print("Warning: could not save memory:", e)

def _get_memory_messages() -> list:
    mem = (persistent_memory or "").strip()
    if not mem:
        return []
    return [{
        "role": "system",
        "content": "Key facts and context remembered from previous conversations (carry these forward):\n" + mem
    }]

def _append_memory(new_text: str):
    """Append a new memory entry (with date) and enforce ~100 line cap."""
    global persistent_memory
    txt = (new_text or "").strip()
    if not txt:
        return
    low = txt.lower()
    if "nothing significant" in low or "nothing to remember" in low or low in ("", "none", "n/a"):
        return
    ts = datetime.datetime.now().strftime("%Y-%m-%d")
    entry = f"[{ts}] {txt}"
    combined = (persistent_memory + "\n\n" + entry).strip() if persistent_memory else entry
    lines = combined.splitlines()
    if len(lines) > 100:
        lines = lines[-100:]
    persistent_memory = "\n".join(lines)
    _save_persistent_memory()

def _call_llm_simple(messages: list, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """Minimal non-tool LLM call for memory extraction and similar."""
    try:
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        r = requests.post(f"{LLM_BASE_URL}/chat/completions", json=payload, timeout=120)
        if r.status_code == 200:
            return (r.json().get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        print(f"LLM (simple) HTTP {r.status_code}")
    except Exception as e:
        print("LLM (simple) error:", e)
    return ""


def summarize_for_compaction(older_turns: list) -> str:
    """Ask the LLM for a compact summary of a prefix of older turns.
    This is for within-session coherence when we need to reduce the rolling history
    (different goal from the durable persistent memory extracted on full clears).
    """
    if not older_turns:
        return ""
    # Instruction scoped to "still useful right now in this conversation".
    instruction = {
        "role": "user",
        "content": (
            "The turns above are older parts of the *current ongoing conversation* and need to be compacted.\n"
            "Create an extremely concise summary (bullets or 1-3 short paragraphs) of the user goals, key facts, decisions, important discoveries or tool outcomes, and context that the assistant must remember to remain coherent and effective for the rest of *this* session.\n"
            "Ignore transient one-off details. If there is little still relevant, reply exactly with: No significant earlier context."
        )
    }
    # Reuse main SYSTEM_PROMPT so the summarizer stays in the agent's character.
    msgs = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + older_turns
        + [instruction]
    )
    return _call_llm_simple(msgs, max_tokens=400, temperature=0.1)


def extract_memory_from_history() -> str:
    """Ask the LLM what (if anything) should be remembered before clearing the conversation."""
    global conversation_history
    if not conversation_history:
        return ""
    # Use the actual dialog turns + a targeted instruction.
    # Include the main SYSTEM_PROMPT so the model stays in character for "what *I* should remember".
    instruction = {
        "role": "user",
        "content": (
            "The conversation above is about to be cleared (inactivity or explicit reset).\n"
            "Before it is cleared, tell your future self the most important durable things to remember:\n"
            "- User preferences, name, style, or recurring requests\n"
            "- Key projects, tasks, files, or goals in progress\n"
            "- Important facts, decisions, or context that will help in future conversations\n\n"
            "Be extremely concise (a few bullets or short paragraphs at most).\n"
            "If there is truly nothing worth carrying forward, reply with exactly: Nothing significant to remember."
        )
    }
    msgs = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + conversation_history
        + [instruction]
    )
    return _call_llm_simple(msgs, max_tokens=450, temperature=0.15)

def commit_memory_before_clear():
    """Extract memory from the about-to-be-cleared history and append if useful."""
    try:
        mem = extract_memory_from_history()
        if mem:
            _append_memory(mem)
            # Keep a brief trace
            lines = [l for l in mem.splitlines() if l.strip()]
            print(f"🧠 Extracted memory ({len(lines)} lines) before clearing context")
    except Exception as e:
        print("Memory extraction failed (continuing):", e)

# ====================== LLM + MULTI-TURN TOOLS ======================
def process_with_llm(user_text: str, internal: bool = False) -> str:
    """Core agent loop. Adds user turn (unless internal), runs LLM allowing tool_calls until final message, returns text.
    trim_conversation_history (with LLM compaction) is called before adding the user turn and after the response.
    Persists only user + final assistant messages (plus occasional compaction summaries).

    When internal=True (cron jobs, future internal triggers), the provided user_text is used to drive the LLM
    and tool loop but is *not* appended to conversation_history, nor is the resulting assistant message.
    The caller is responsible for what to do with the returned text (e.g. queue_proactive_message)."""
    global conversation_history
    trim_conversation_history()
    if not internal:
        conversation_history.append({"role": "user", "content": user_text})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + _get_memory_messages() + conversation_history
    if internal:
        # Drive the agent with an internal prompt/directive without recording the trigger in visible history.
        messages = messages + [{"role": "user", "content": user_text}]

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
                    fn = tc.get("function", {})
                    name = fn.get("name", "tool")
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except Exception:
                        args = {}
                    if name == "web_search":
                        q = args.get("query", "")
                        print(f"  🔧 web_search: {q}" if q else "  🔧 web_search")
                    elif name == "run_terminal":
                        cmd = args.get("command", "")
                        print(f"  🔧 run_terminal: {cmd}" if cmd else "  🔧 run_terminal")
                    else:
                        print(f"  🔧 {name}")
                    out = execute_tool(tc)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": out
                    })
                continue
            else:
                final_text = (msg.get("content") or "").strip()
                if not internal:
                    conversation_history.append({"role": "assistant", "content": final_text})
                break
        except Exception as e:
            print("LLM exception:", e)
            final_text = f"(LLM failure: {e})"
            break

    if not final_text:
        # All tool turns exhausted without a final response — let the user know.
        fallback = "(Response unavailable — the model used all available tool turns without producing a final answer.)"
        if not internal:
            conversation_history.append({"role": "assistant", "content": fallback})
        final_text = fallback
    trim_conversation_history()
    return final_text

# ====================== TTS ======================
def generate_tts_audio(text: str, quiet: bool = False) -> bytes:
    if not text or not TTS_BASE_URL:
        return b""
    try:
        payload = {
            "model": TTS_MODEL,
            "input": text.strip(),
            "voice": TTS_VOICE,
            "response_format": "wav"
        }
        if not quiet:
            print("🔊 TTS synthesis...")
        r = requests.post(f"{TTS_BASE_URL}/audio/speech", json=payload, timeout=180)
        if r.status_code == 200 and r.content:
            return r.content
        if r.status_code == 200:
            # 200 but no bytes: the TTS server accepted the request but generated no audio data.
            # Common causes: specific voice unavailable in this container/image (af_heart is flaky),
            # or the Kokoro service itself has stopped synthesizing (stale container, GPU issue, etc).
            print(f"TTS {r.status_code} but 0-byte body (no audio generated). voice={TTS_VOICE}")
            print("   Suggestion: restart the Kokoro TTS container, or try a different TTS_VOICE in config.json (am_adam / af_bella often more reliable).")
        else:
            print(f"TTS {r.status_code}: {r.text[:150] if r.text else ''}")
    except Exception as e:
        print("TTS error:", e)
    return b""

TTS_PROBE_TEXT = "Hi."
TTS_PROBE_MIN_BYTES = 1000  # WAV header + real audio (0-byte 200s mean Kokoro GPU is broken)
TTS_PROBE_CACHE_SECONDS = 60
_tts_probe_cache = {"ok": None, "bytes": 0, "error": None, "checked_at": None}
_tts_probe_lock = threading.Lock()

def probe_tts_synthesis(force: bool = False) -> dict:
    """Check whether Kokoro is actually returning audio (not just HTTP 200 + empty body).
    Results are cached for TTS_PROBE_CACHE_SECONDS; pass force=True to bypass."""
    if not TTS_BASE_URL:
        return {"ok": None, "bytes": 0, "error": None, "checked_at": None}

    with _tts_probe_lock:
        cached_at = _tts_probe_cache.get("checked_at")
        if not force and cached_at and (time.time() - cached_at) < TTS_PROBE_CACHE_SECONDS:
            return dict(_tts_probe_cache)

    audio = generate_tts_audio(TTS_PROBE_TEXT, quiet=True)
    if len(audio) >= TTS_PROBE_MIN_BYTES:
        result = {"ok": True, "bytes": len(audio), "error": None, "checked_at": time.time()}
    else:
        result = {
            "ok": False,
            "bytes": len(audio),
            "error": "no audio — restart kokoro-tts container",
            "checked_at": time.time(),
        }

    with _tts_probe_lock:
        _tts_probe_cache.update(result)
    return dict(result)

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

WHISPER_PROBE_CACHE_SECONDS = 60
_whisper_probe_cache = {"ok": None, "error": None, "checked_at": None}
_whisper_probe_lock = threading.Lock()

def _make_stt_probe_wav(duration_sec: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Minimal 16 kHz mono WAV for whisper.cpp health checks (silence is fine)."""
    n_samples = max(1, int(sample_rate * duration_sec))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()

def probe_whisper_stt(force: bool = False) -> dict:
    """Check whether whisper.cpp accepts audio and returns a valid transcription response."""
    if not WHISPER_BASE_URL:
        return {"ok": None, "error": None, "checked_at": None}

    with _whisper_probe_lock:
        cached_at = _whisper_probe_cache.get("checked_at")
        if not force and cached_at and (time.time() - cached_at) < WHISPER_PROBE_CACHE_SECONDS:
            return dict(_whisper_probe_cache)

    try:
        wav = _make_stt_probe_wav()
        files = {"file": ("probe.wav", wav, "audio/wav")}
        data = {
            "model": WHISPER_MODEL,
            "language": "en",
            "temperature": "0.0",
            "response_format": "json",
        }
        r = requests.post(
            f"{WHISPER_BASE_URL}/v1/audio/transcriptions",
            files=files,
            data=data,
            timeout=30,
        )
        if r.status_code == 200:
            r.json()  # validate JSON body
            result = {"ok": True, "error": None, "checked_at": time.time()}
        else:
            result = {
                "ok": False,
                "error": f"HTTP {r.status_code}" + (f": {r.text[:80]}" if r.text else ""),
                "checked_at": time.time(),
            }
    except Exception as e:
        result = {"ok": False, "error": str(e), "checked_at": time.time()}

    with _whisper_probe_lock:
        _whisper_probe_cache.update(result)
    return dict(result)


# ====================== IMAGE DETECTION (YOLO external server) ======================
def detect_objects(image_file) -> list:
    """Accept FileStorage (from request.files 'image' or 'file'). Forward to YOLO /upload.
    Return list of detected object label strings (e.g. ['person', 'cat']).
    """
    if not DETECTION_BASE_URL:
        return []
    try:
        image_bytes = image_file.read()
        files = {"image": ("image.jpg", image_bytes)}
        print("🖼️  Detecting objects via YOLO server...")
        r = requests.post(f"{DETECTION_BASE_URL}/upload", files=files, timeout=120)
        if r.status_code == 200:
            data = r.json()
            dets = data.get("detections", [])
            labels = [d.get("name") for d in dets if d.get("name")]
            print(f"   Detected: {labels}")
            return labels
        print(f"Detection HTTP {r.status_code}: {r.text[:200] if r.text else ''}")
    except Exception as e:
        print("Detection error:", e)
    return []


# ====================== PROACTIVE QUEUE HELPER ======================
def queue_proactive_message(text: str, speak: bool = True) -> dict:
    """Queue a message for the client to receive on its next /poll when idle.
    If speak and TTS is configured, pre-generates the audio at enqueue time.
    Returns the queued item dict. Thread-safe. Enforces size + age limits.
    """
    global pending_initiations
    if not text or not text.strip():
        return {}

    audio_b64 = None
    if speak and TTS_BASE_URL:
        try:
            audio_bytes = generate_tts_audio(text)
            if audio_bytes:
                audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        except Exception as e:
            print("Proactive TTS generation failed:", e)

    item = {
        "id": str(uuid.uuid4()),
        "text": text.strip(),
        "audio": audio_b64,
        "created_at": datetime.datetime.now().isoformat()
    }

    with pending_lock:
        # Drop anything too old
        now = datetime.datetime.now()
        while pending_initiations:
            oldest = pending_initiations[0]
            try:
                created = datetime.datetime.fromisoformat(oldest["created_at"])
                if (now - created).total_seconds() > MAX_INITIATION_AGE_SECONDS:
                    pending_initiations.popleft()
                    print("🗑️  Dropped stale proactive message (age)")
                    continue
            except Exception:
                pending_initiations.popleft()
                continue
            break

        # Enforce max depth (drop oldest if full)
        while len(pending_initiations) >= MAX_PENDING_INITIATIONS:
            dropped = pending_initiations.popleft()
            print(f"🗑️  Dropped oldest proactive (queue full): {dropped['text'][:60]}...")

        pending_initiations.append(item)
        # Wake any long-poll waiters
        initiation_ready.notify_all()

    print(f"📣 Queued proactive message (queue size={len(pending_initiations)}): {text[:80]}{'...' if len(text) > 80 else ''}")
    return item


# ====================== CRON SCHEDULER ======================
def start_cron_scheduler():
    """Start a daemon thread that periodically checks cron_jobs and fires any that are due.
    Due jobs run their prompt through the LLM (internal=True so history stays clean) and
    the resulting text is queued as a proactive message (which will be spoken + added to
    conversation on delivery, exactly like other proactive messages)."""
    enabled_jobs = [j for j in cron_jobs if j.get("enabled", True)]
    if not enabled_jobs:
        return

    def _cron_loop():
        while True:
            try:
                time.sleep(30)  # minute-granularity crons are well served by 30s checks
                now = datetime.datetime.now()
                for job in cron_jobs:
                    if not job.get("enabled", True):
                        continue
                    sched = job.get("schedule", "")
                    prompt = job.get("prompt", "")
                    if not sched or not prompt:
                        continue
                    if not cron_due(sched, now):
                        continue
                    # Use minute slot for "already executed this occurrence?"
                    slot = now.replace(second=0, microsecond=0)
                    lr = job.get("last_run")
                    if lr is not None:
                        if lr.replace(second=0, microsecond=0) == slot:
                            continue
                    job["last_run"] = now
                    _save_cron_last_run(job["id"], now)
                    print(f"\n⏰ Cron fired [{sched}]: {prompt[:90]}{'...' if len(prompt) > 90 else ''}")
                    try:
                        result = process_with_llm(prompt, internal=True)
                        if result and result.strip():
                            print(f"🐹 Cron result: {result[:140]}{'...' if len(result) > 140 else ''}")
                            queue_proactive_message(result, speak=True)
                    except Exception as ex:
                        print("Cron job failed:", ex)
            except Exception as e:
                print("Cron scheduler error (retrying):", e)
                time.sleep(10)

    t = threading.Thread(target=_cron_loop, daemon=True, name="marmot-cron")
    t.start()
    print(f"⏰ Cron scheduler started for {len(enabled_jobs)} job(s)")


# ====================== FLASK ======================
# Load memory (after all helper defs are registered) and emit startup banner
_load_persistent_memory()
_mem_lines = len([ln for ln in (persistent_memory or "").splitlines() if ln.strip()])

load_cron_jobs()

print("🐹 Marmot Agent Server ready")
print(f"   Whisper: {WHISPER_BASE_URL}  model={WHISPER_MODEL}")
print(f"   LLM:     {LLM_MODEL} @ {LLM_BASE_URL}")
print(f"   TTS:     {TTS_MODEL}/{TTS_VOICE} @ {TTS_BASE_URL or '(disabled)'}")
print(f"   Detection: {DETECTION_BASE_URL or '(disabled)'}")
print(f"   Context: ~{MAX_CONTEXT_TOKENS} tokens max (rolling + LLM compaction of old turns)")
_tool_names = ", ".join(t["function"]["name"] for t in TOOLS) if TOOLS else "(none)"
print(f"   Tools:   {'on' if TOOLS_ENABLED else 'off'}   [{_tool_names}]   tool-timeout={TOOL_TIMEOUT}s")
if WEB_SEARCH_ENABLED and not BRAVE_SEARCH_API_KEY:
    print("   Web search: disabled (set BRAVE_SEARCH_API_KEY in config.json)")
print(f"   Inactivity timeout: {CONTEXT_TIMEOUT_HOURS}h → auto-clear context")
print(f"   Memory:   {_mem_lines} lines persisted (≤100, extracted before clears)")
if cron_jobs:
    en = sum(1 for j in cron_jobs if j.get("enabled", True))
    dis = len(cron_jobs) - en
    extra = f" ({dis} disabled)" if dis else ""
    print(f"   Cron:     {en} job(s) from cron.json{extra}")
print()

try:
    wprobe = probe_whisper_stt(force=True)
    if wprobe.get("ok"):
        print("   Whisper probe: OK")
    else:
        print(f"⚠️  Whisper probe failed: {wprobe.get('error') or 'unknown error'}")
except Exception as _e:
    print("⚠️  Whisper probe error (non-fatal):", _e)

# Quick TTS probe so users immediately see if the configured voice is producing audio.
if TTS_BASE_URL:
    try:
        probe = probe_tts_synthesis(force=True)
        if probe.get("ok"):
            print(f"   TTS probe: OK ({probe['bytes']} bytes)")
        else:
            print(f"⚠️  TTS probe failed: {probe.get('error') or 'no audio'}")
    except Exception as _e:
        print("⚠️  TTS probe error (non-fatal):", _e)

app = Flask(__name__)

# ====================== SIMPLE STATUS DASHBOARD ======================
# Served at GET /  — a lightweight, auto-refreshing page showing /health data.
# The mascot is embedded as a data URL so the page is completely self-contained.
MARMOT_B64 = ""
try:
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "images", "marmot-harness.jpg")
    with open(img_path, "rb") as f:
        MARMOT_B64 = base64.b64encode(f.read()).decode("ascii")
except Exception as e:
    print("Warning: could not load mascot image for dashboard:", e)

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Marmot • Status</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&amp;family=Space+Grotesk:wght@500;600&amp;display=swap');
    :root { font-family: Inter, system_ui, sans-serif; }
    .font-display { font-family: "Space Grotesk", Inter, system_ui, sans-serif; }
    .status-dot { width: 10px; height: 10px; border-radius: 9999px; }
    .card { transition: transform 0.1s ease, box-shadow 0.1s ease; }
    .card:hover { transform: translateY(-1px); box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    .section-title { font-size: 11px; letter-spacing: 0.5px; text-transform: uppercase; font-weight: 600; color: #64748b; }
    .health-value { font-size: 15px; font-weight: 600; line-height: 1.1; }
    .chat-messages { scrollbar-width: thin; scrollbar-color: #334155 transparent; }
    .chat-messages::-webkit-scrollbar { width: 6px; }
    .chat-messages::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    .chat-bubble { max-width: 85%; word-break: break-word; white-space: pre-wrap; }
    .chat-bubble-user { margin-left: auto; background: #1e3a5f; border: 1px solid #1e40af; }
    .chat-bubble-assistant { margin-right: auto; background: #0f172a; border: 1px solid #334155; }
    .chat-typing span { animation: chat-blink 1.4s infinite both; }
    .chat-typing span:nth-child(2) { animation-delay: 0.2s; }
    .chat-typing span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes chat-blink { 0%, 80%, 100% { opacity: 0.2; } 40% { opacity: 1; } }
  </style>
</head>
<body class="bg-slate-950 text-slate-200">
  <div class="max-w-4xl mx-auto px-6 py-8">
    <!-- Header -->
    <div class="flex items-start justify-between gap-4">
      <div class="flex items-center gap-4">
        <img id="mascot" 
             src="data:image/jpeg;base64,MARMOT_B64_HERE" 
             alt="Marmot mascot" 
             class="w-20 h-20 rounded-3xl object-cover ring-1 ring-white/10 shadow-xl shadow-black/50">
        <div>
          <div class="flex items-center gap-2">
            <h1 class="text-4xl font-semibold tracking-tighter font-display">Marmot</h1>
            <div class="px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 text-xs font-medium flex items-center gap-1">
              <div class="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse"></div>
              LIVE
            </div>
          </div>
          <p class="text-slate-400 text-sm mt-0.5">Local voice-first AI agent • Status dashboard &amp; web chat</p>
        </div>
      </div>

      <div class="text-right text-xs text-slate-500 pt-1">
        <div>Port <span class="text-slate-400 font-medium" id="port">5000</span></div>
        <div id="updated" class="text-[10px] mt-0.5">—</div>
      </div>
    </div>

    <!-- Toolbar -->
    <div class="mt-6 flex flex-wrap items-center gap-2">
      <button onclick="refreshHealth()" 
              class="inline-flex items-center gap-2 rounded-2xl bg-slate-900 hover:bg-slate-800 active:bg-slate-950 border border-slate-700 px-4 py-2 text-sm font-medium transition-colors">
        <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.058 11H1M12 3v2m0 16v2m9-9H15m-6 0a8 8 0 01-.938-1.5M12 21a8 8 0 01-1.5-.938" />
        </svg>
        Refresh
      </button>

      <button onclick="resetContext()" 
              class="inline-flex items-center gap-2 rounded-2xl bg-red-950/60 hover:bg-red-950 border border-red-900/70 text-red-300 px-4 py-2 text-sm font-medium transition-colors">
        <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.595 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.595-1.858L5 7m5 4v6m4-6v6m1-10V9a1 1 0 00-1-1H9a1 1 0 00-1 1v1m-1 4h10" />
        </svg>
        Reset Context
      </button>

      <div class="flex-1"></div>

      <div class="text-xs px-3 py-1.5 rounded-2xl bg-slate-900 border border-slate-800 text-slate-400 flex items-center gap-2">
        <span>Auto-refresh</span> 
        <span class="font-mono text-emerald-400">5s</span>
      </div>
    </div>

    <!-- Chat -->
    <div class="mt-6">
      <div class="section-title mb-2 px-1">Chat</div>
      <div class="card bg-slate-900 border border-slate-800 rounded-3xl overflow-hidden flex flex-col h-[28rem]">
        <div id="chat-messages" class="chat-messages flex-1 overflow-y-auto p-4 space-y-3">
          <div id="chat-empty" class="h-full flex flex-col items-center justify-center text-center text-slate-500 text-sm px-6">
            <div class="text-2xl mb-2 opacity-60">💬</div>
            <div>Send a message to Marmot. Responses are text-only from this interface.</div>
          </div>
        </div>
        <div class="border-t border-slate-800 p-3 flex gap-2 items-end">
          <textarea id="chat-input" rows="1" placeholder="Type a message…"
                    class="flex-1 resize-none rounded-2xl bg-slate-950 border border-slate-700 px-4 py-2.5 text-sm text-slate-200 placeholder:text-slate-500 focus:outline-none focus:ring-1 focus:ring-emerald-500/50 focus:border-emerald-500/50"></textarea>
          <button id="chat-send" onclick="sendChat()"
                  class="inline-flex items-center justify-center rounded-2xl bg-emerald-600 hover:bg-emerald-500 active:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed px-4 py-2.5 text-sm font-medium transition-colors shrink-0">
            <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          </button>
        </div>
      </div>
    </div>

    <!-- Services -->
    <div class="mt-6">
      <div class="section-title mb-2 px-1">Services</div>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3" id="services">
        <!-- JS populated cards -->
      </div>
    </div>

    <!-- Context & Memory -->
    <div class="mt-6 grid grid-cols-1 md:grid-cols-2 gap-3">
      <div class="card bg-slate-900 border border-slate-800 rounded-3xl p-5">
        <div class="section-title mb-3">Conversation</div>
        <div class="space-y-3">
          <div class="flex justify-between items-baseline">
            <div class="text-slate-400 text-sm">Active turns</div>
            <div id="turns" class="health-value text-3xl tabular-nums font-semibold text-white">—</div>
          </div>
          <div>
            <div class="flex justify-between text-sm mb-1">
              <span class="text-slate-400">Last message</span>
              <span id="last-seen" class="font-medium text-slate-300 tabular-nums">—</span>
            </div>
            <div id="last-at" class="text-[10px] text-slate-500 mono">—</div>
          </div>
        </div>
      </div>

      <div class="card bg-slate-900 border border-slate-800 rounded-3xl p-5">
        <div class="section-title mb-3">Memory &amp; Proactive</div>
        <div class="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
          <div>
            <div class="text-slate-400">Persistent memory</div>
            <div id="memory-lines" class="health-value text-2xl font-semibold tabular-nums">—</div>
            <div class="text-[10px] text-slate-500">lines saved</div>
          </div>
          <div>
            <div class="text-slate-400">Pending proactive</div>
            <div id="pending" class="health-value text-2xl font-semibold tabular-nums">—</div>
            <div class="text-[10px] text-slate-500">queued for delivery</div>
          </div>
          <div class="col-span-2 pt-1 border-t border-slate-800 text-xs">
            Context auto-clears after <span id="ctx-timeout" class="font-medium text-slate-300 tabular-nums">—</span> hours of inactivity
          </div>
        </div>
      </div>
    </div>

    <!-- Cron -->
    <div class="mt-6">
      <div class="flex items-center justify-between mb-2 px-1">
        <div class="section-title">Cron Jobs</div>
        <div id="cron-count" class="text-xs text-slate-400"></div>
      </div>
      <div id="cron-list" class="flex flex-wrap gap-2 text-sm">
        <!-- JS populated schedule pills -->
      </div>
      <div id="cron-empty" class="text-xs text-slate-500 px-1 hidden">No cron jobs configured (copy cron.json.example to get started).</div>
    </div>

    <!-- Footer -->
    <div class="mt-8 pt-5 border-t border-slate-800 text-xs text-slate-500 flex items-center justify-between">
      <div>
        Data from <a href="/health" class="text-slate-400 hover:text-slate-300 underline">/health</a>. 
        All state lives on the server.
      </div>
      <div class="text-[10px]">Marmot harness</div>
    </div>
  </div>

<script>
let lastSeconds = null;
let lastUpdateTs = Date.now();
let chatBusy = false;

function tailwindInit() {
  // Tailwind script already loaded via CDN; any custom config could go here.
}

function formatAgo(seconds) {
  if (seconds == null) return "never";
  if (seconds < 5) return "just now";
  if (seconds < 60) return seconds + "s ago";
  const m = Math.floor(seconds / 60);
  if (m < 60) return m + "m ago";
  const h = Math.floor(m / 60);
  return h + "h " + (m % 60) + "m ago";
}

function updateRelativeTimes() {
  if (lastSeconds == null) return;
  // Decrement locally for smooth "live" feel between full fetches
  const elapsed = Math.floor((Date.now() - lastUpdateTs) / 1000);
  const current = Math.max(0, lastSeconds + elapsed);
  const el = document.getElementById("last-seen");
  if (el) el.textContent = formatAgo(current);
}

function renderServices(data) {
  const container = document.getElementById("services");
  if (!container) return;
  container.innerHTML = "";

  const services = [
    (() => {
      const stt = data.whisper_stt || {};
      let whisperOn = !!data.whisper;
      let whisperError = false;
      let whisperSub = data.whisper_model || "";
      if (data.whisper) {
        if (stt.ok === true) {
          whisperSub = (data.whisper_model || "whisper") + " · transcription OK";
        } else if (stt.ok === false) {
          whisperError = true;
          whisperSub = stt.error || "transcription failed";
        } else if (!whisperSub) {
          whisperSub = "not probed yet";
        }
      }
      return { key: "whisper", label: "Whisper", value: data.whisper || "—", sub: whisperSub, on: whisperOn, error: whisperError };
    })(),
    { key: "llm", label: "LLM", value: data.llm || "—", sub: data.llm_base_url || "", on: true },
    (() => {
      const synth = data.tts_synthesis || {};
      let ttsOn = !!data.tts;
      let ttsError = false;
      let ttsSub = "";
      if (data.tts) {
        if (synth.ok === true) {
          ttsSub = "synthesis OK (" + Math.round((synth.bytes || 0) / 1024) + " KB)";
        } else if (synth.ok === false) {
          ttsError = true;
          ttsSub = synth.error || "synthesis failed";
        } else {
          ttsSub = "not probed yet";
        }
      }
      return { key: "tts", label: "TTS", value: data.tts ? ((data.tts_model || "tts") + " / " + (data.tts_voice || "")) : "disabled", sub: ttsSub, on: ttsOn, error: ttsError };
    })(),
    { key: "detect", label: "Detection", value: data.detection || "disabled", sub: "", on: !!data.detection }
  ];

  services.forEach(s => {
    const div = document.createElement("div");
    div.className = "bg-slate-900 border border-slate-800 rounded-2xl p-3.5 flex flex-col";
    const dotColor = s.error ? "bg-red-400" : (s.on ? "bg-emerald-400" : "bg-slate-600");
    const valueText = (s.on && !s.error) ? s.value : (s.error ? s.value : `<span class="text-slate-400">${s.value}</span>`);

    div.innerHTML = `
      <div class="flex items-center gap-2">
        <div class="status-dot ${dotColor} ${s.on ? '' : 'opacity-60'} ring-1 ring-inset ring-white/10"></div>
        <div class="text-xs font-medium tracking-wider text-slate-400">${s.label}</div>
      </div>
      <div class="mt-1.5">
        <div class="health-value break-all">${valueText}</div>
        ${s.sub ? `<div class="text-[10px] ${s.error ? 'text-red-400' : 'text-slate-500'} mono mt-0.5 truncate" title="${s.sub}">${s.sub}</div>` : ''}
      </div>
    `;
    container.appendChild(div);
  });
}

function renderCron(data) {
  const list = document.getElementById("cron-list");
  const empty = document.getElementById("cron-empty");
  const countEl = document.getElementById("cron-count");
  if (!list || !empty || !countEl) return;

  list.innerHTML = "";
  empty.classList.add("hidden");

  const crons = data.cron || [];
  const total = data.cron_jobs || 0;
  const enabled = crons.filter(c => c.enabled).length;

  countEl.textContent = total ? `${enabled}/${total} enabled` : "";

  if (crons.length === 0) {
    empty.classList.remove("hidden");
    return;
  }

  crons.forEach(c => {
    const pill = document.createElement("div");
    const isOn = c.enabled !== false;
    pill.className = `px-3 py-1 rounded-2xl text-xs border flex items-center gap-1.5 ${isOn ? 'bg-emerald-900/30 border-emerald-800 text-emerald-300' : 'bg-slate-800 border-slate-700 text-slate-400'}`;

    let last = "never";
    if (c.last_run) {
      const d = new Date(c.last_run);
      const sameDay = d.toDateString() === new Date().toDateString();
      const time = d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
      last = sameDay ? time : d.toLocaleDateString([], {month:'short', day:'numeric'}) + " " + time;
    }
    pill.innerHTML = `
      <span class="font-medium">${c.schedule}</span>
      ${!isOn ? '<span class="text-[10px] opacity-60">(off)</span>' : ''}
      <span class="opacity-50 text-[10px]">• last ${last}</span>
    `;
    list.appendChild(pill);
  });
}

async function fetchHealth(forceServiceProbe) {
  try {
    const url = forceServiceProbe ? "/health?probe=all" : "/health";
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error("bad status " + res.status);
    const data = await res.json();

    // Services
    renderServices(data);

    // Conversation stats
    document.getElementById("turns").textContent = data.turns ?? "—";
    document.getElementById("memory-lines").textContent = data.memory_lines ?? "—";
    document.getElementById("pending").textContent = data.pending_initiations ?? "—";
    document.getElementById("ctx-timeout").textContent = data.context_timeout_hours ?? "—";

    // Last message
    const lastSeenEl = document.getElementById("last-seen");
    const lastAtEl = document.getElementById("last-at");

    if (data.seconds_since_last_message != null) {
      lastSeconds = data.seconds_since_last_message;
      lastUpdateTs = Date.now();
      lastSeenEl.textContent = formatAgo(lastSeconds);
      lastAtEl.textContent = data.last_message_at ? new Date(data.last_message_at).toLocaleString() : "—";
    } else {
      lastSeenEl.textContent = "never";
      lastAtEl.textContent = "—";
      lastSeconds = null;
    }

    // Cron
    renderCron(data);

    // Updated time
    const upd = document.getElementById("updated");
    upd.textContent = "updated " + new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});

    // Also expose port if we can (static for now)
    const portEl = document.getElementById("port");
    if (portEl && location.port) portEl.textContent = location.port;

  } catch (e) {
    console.warn("Health fetch failed", e);
    const container = document.getElementById("services");
    if (container) {
      container.innerHTML = `<div class="col-span-full text-sm text-red-400 bg-red-950/40 border border-red-900 rounded-2xl p-4">Unable to reach /health — is the Marmot server running?</div>`;
    }
  }
}

async function resetContext() {
  if (!confirm("Clear the current conversation context and extract memory?")) return;
  try {
    const res = await fetch("/reset", { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    // brief flash
    const btns = document.querySelectorAll("button");
    btns.forEach(b => b.disabled = true);
    await fetchHealth();
    setTimeout(() => {
      btns.forEach(b => b.disabled = false);
    }, 600);
  } catch (e) {
    alert("Reset failed: " + e);
  }
}

function refreshHealth() {
  fetchHealth();
}

function setChatBusy(busy) {
  chatBusy = busy;
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("chat-send");
  if (input) input.disabled = busy;
  if (sendBtn) sendBtn.disabled = busy;
}

function appendChatBubble(role, text) {
  const container = document.getElementById("chat-messages");
  const empty = document.getElementById("chat-empty");
  if (!container) return;
  if (empty) empty.remove();

  const wrap = document.createElement("div");
  wrap.className = "flex " + (role === "user" ? "justify-end" : "justify-start");

  const bubble = document.createElement("div");
  bubble.className = "chat-bubble rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed " +
    (role === "user" ? "chat-bubble-user text-slate-100" : "chat-bubble-assistant text-slate-200");
  bubble.textContent = text;
  wrap.appendChild(bubble);
  container.appendChild(wrap);
  container.scrollTop = container.scrollHeight;
}

function appendChatTyping() {
  const container = document.getElementById("chat-messages");
  if (!container) return null;
  const wrap = document.createElement("div");
  wrap.id = "chat-typing";
  wrap.className = "flex justify-start";
  wrap.innerHTML = `
    <div class="chat-bubble chat-bubble-assistant rounded-2xl px-3.5 py-2.5 text-sm text-slate-400 chat-typing">
      Marmot is thinking<span>.</span><span>.</span><span>.</span>
    </div>
  `;
  container.appendChild(wrap);
  container.scrollTop = container.scrollHeight;
  return wrap;
}

function removeChatTyping() {
  const el = document.getElementById("chat-typing");
  if (el) el.remove();
}

async function sendChat() {
  if (chatBusy) return;
  const input = document.getElementById("chat-input");
  if (!input) return;
  const text = input.value.trim();
  if (!text) return;

  appendChatBubble("user", text);
  input.value = "";
  input.style.height = "auto";
  setChatBusy(true);
  appendChatTyping();

  try {
    const res = await fetch("/connect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text, text_only: true })
    });
    removeChatTyping();
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || ("HTTP " + res.status));
    }
    const data = await res.json();
    appendChatBubble("assistant", data.text || "(empty response)");
    fetchHealth();
  } catch (e) {
    removeChatTyping();
    appendChatBubble("assistant", "Error: " + e.message);
  } finally {
    setChatBusy(false);
    input.focus();
  }
}

function initChatInput() {
  const input = document.getElementById("chat-input");
  if (!input) return;
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  });
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 120) + "px";
  });
}

function startAutoRefresh() {
  // Initial fetch includes live Whisper + TTS probes
  fetchHealth(true);
  // Full refresh every 5 seconds (service probes cached server-side ~60s)
  setInterval(() => fetchHealth(false), 5000);
  // Force fresh Whisper + TTS checks every 60 seconds
  setInterval(() => fetchHealth(true), 60000);
  // Smooth relative time updates every 1s
  setInterval(updateRelativeTimes, 1000);
}

window.onload = function() {
  tailwindInit();
  initChatInput();
  startAutoRefresh();
};
</script>
</body>
</html>
"""
INDEX_HTML = INDEX_HTML.replace("MARMOT_B64_HERE", MARMOT_B64)

@app.route("/", methods=["GET"])
def index():
    """Simple self-contained status dashboard. Refreshes automatically."""
    return INDEX_HTML


@app.route("/connect", methods=["POST"])
def connect():
    user_text = None
    if request.files and "file" in request.files:
        f = request.files["file"]
        if f and f.filename:
            user_text = transcribe_audio(f)

    text_only = False
    if not user_text:
        if request.is_json:
            body = request.json or {}
            user_text = body.get("text", "")
            text_only = bool(body.get("text_only"))
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
            commit_memory_before_clear()
            conversation_history.clear()
    last_message_time = now

    print(f"\n👤 User: {user_text}")
    final = process_with_llm(user_text)
    print(f"🐹 Marmot: {final[:160]}{'...' if len(final) > 160 else ''}")

    audio_b64 = None
    if not text_only:
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

    with pending_lock:
        pending_count = len(pending_initiations)

    cron_summary = [
        {
            "schedule": j["schedule"],
            "enabled": j.get("enabled", True),
            "last_run": j["last_run"].isoformat() if j.get("last_run") else None
        }
        for j in cron_jobs
    ]

    probe_arg = (request.args.get("probe") or "").strip().lower()
    force_all = probe_arg in ("1", "true", "all", "yes")
    force_whisper = force_all or probe_arg == "whisper"
    force_tts = force_all or probe_arg == "tts"

    whisper_stt = probe_whisper_stt(force=force_whisper)
    tts_synthesis = probe_tts_synthesis(force=force_tts) if TTS_BASE_URL else None
    ok = True
    if whisper_stt.get("ok") is False:
        ok = False
    if tts_synthesis and tts_synthesis.get("ok") is False:
        ok = False

    return jsonify({
        "ok": ok,
        "whisper": WHISPER_BASE_URL,
        "whisper_model": WHISPER_MODEL,
        "whisper_stt": whisper_stt,
        "llm": LLM_MODEL,
        "llm_base_url": LLM_BASE_URL,
        "tts": bool(TTS_BASE_URL),
        "tts_model": TTS_MODEL if TTS_BASE_URL else None,
        "tts_voice": TTS_VOICE if TTS_BASE_URL else None,
        "tts_synthesis": tts_synthesis,
        "detection": DETECTION_BASE_URL,
        "turns": len([m for m in conversation_history if m["role"] in ("user", "assistant")]),
        "context_timeout_hours": CONTEXT_TIMEOUT_HOURS,
        "last_message_at": last_message_time.isoformat() if last_message_time else None,
        "seconds_since_last_message": seconds_since_last,
        "memory_lines": len([ln for ln in (persistent_memory or "").splitlines() if ln.strip()]),
        "pending_initiations": pending_count,
        "cron_jobs": len(cron_jobs),
        "cron": cron_summary
    })

@app.route("/reset", methods=["POST"])
def reset():
    global conversation_history, last_message_time
    commit_memory_before_clear()
    conversation_history = []
    last_message_time = None
    with pending_lock:
        pending_initiations.clear()
    return jsonify({"ok": True, "msg": "context cleared"})

@app.route("/poll", methods=["GET"])
def poll():
    """Client idle poll. Returns a proactive initiation if one is queued (and commits it to conversation history).
    Supports optional long-poll via ?wait=seconds (capped at 10).
    """
    global last_message_time, conversation_history

    wait = 0.0
    try:
        wait = float(request.args.get("wait", "0") or "0")
    except Exception:
        wait = 0.0
    wait = max(0.0, min(wait, 10.0))

    deadline = time.time() + wait

    while True:
        with initiation_ready:
            # Prune stale inside the lock
            now = datetime.datetime.now()
            while pending_initiations:
                try:
                    oldest = pending_initiations[0]
                    created = datetime.datetime.fromisoformat(oldest["created_at"])
                    if (now - created).total_seconds() > MAX_INITIATION_AGE_SECONDS:
                        pending_initiations.popleft()
                        continue
                except Exception:
                    pending_initiations.popleft()
                    continue
                break

            if pending_initiations:
                item = pending_initiations.popleft()
                # Commit this as an assistant turn so the conversation continues naturally
                conversation_history.append({"role": "assistant", "content": item["text"]})
                last_message_time = datetime.datetime.now()
                # Trim opportunistically (cheap if not near limit)
                try:
                    trim_conversation_history()
                except Exception:
                    pass
                print(f"📤 Delivering proactive via /poll: {item['text'][:100]}{'...' if len(item['text']) > 100 else ''}")
                return jsonify({"action": "initiate", "message": item})

            remaining = deadline - time.time()
            if remaining <= 0:
                return jsonify({"action": "noop"})

            # Efficiently wait for a new enqueue or timeout slice
            initiation_ready.wait(timeout=min(remaining, 1.0))

@app.route("/inject", methods=["POST"])
def inject():
    """Manual/test hook to queue a proactive message from outside (e.g. scripts, future schedulers).
    Body: {"text": "message here", "speak": true}
    """
    if not request.is_json:
        return jsonify({"error": "expected application/json"}), 400
    data = request.json or {}
    text = (data.get("text") or "").strip()
    speak = data.get("speak", True)
    if not isinstance(speak, bool):
        speak = str(speak).lower() in ("1", "true", "yes", "on")
    if not text:
        return jsonify({"error": "text is required"}), 400

    item = queue_proactive_message(text, speak=speak)
    return jsonify({"ok": True, "queued": bool(item), "message": item})


@app.route("/detect", methods=["POST"])
def detect():
    """Detect objects in an uploaded image using the external YOLO server.
    Accepts multipart form with 'image' or 'file'.
    Returns {"objects": ["label", "label", ...]} (just the class names).
    """
    if not DETECTION_BASE_URL:
        return jsonify({"error": "Detection server not configured"}), 503

    image_file = None
    if request.files:
        image_file = request.files.get("file") or request.files.get("image")
    if not image_file or not getattr(image_file, "filename", None):
        return jsonify({"error": "Send image file as 'image' or 'file' form field"}), 400

    labels = detect_objects(image_file)
    return jsonify({"objects": labels})


class QuietPollRequestHandler(WSGIRequestHandler):
    """Custom request handler that suppresses log spam from frequent endpoints
    (/poll for the proactive client, and GET /health for the status dashboard auto-refresh).
    All other endpoints continue to log normally.
    """
    def log_request(self, code='-', size='-'):
        if self.path:
            path = self.path.split('?', 1)[0]
            if path.startswith('/poll') or path == '/health':
                return  # keep console clean for high-frequency polling/status endpoints
        super().log_request(code, size)


if __name__ == "__main__":
    port = int(os.environ.get("MARMOT_PORT", 5000))
    start_cron_scheduler()
    print(f"🌐 Dashboard: http://0.0.0.0:{port}/")
    print(f"   API:      /connect  /health  /reset  /poll  /inject  /detect")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True,
            request_handler=QuietPollRequestHandler)
