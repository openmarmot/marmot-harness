#!/usr/bin/env python3
"""
Marmot Agent Client

- Hold Right Option/Alt to record -> send audio to local Marmot server /connect
- Server does STT + LLM (tools) + TTS
- Client receives transcription + AI response + audio
- Prints "You:" (transcription) then "Marmot:" reply, plays audio, copies reply to clipboard
- -m "text" flag: send text directly, play/print/copy response, exit (for testing)
"""

import os
import sys
import tempfile
import time
import threading
import signal
import argparse
import sounddevice as sd
import requests
import subprocess
import numpy as np
from pynput import keyboard
import wave
import platform
import json
import base64

# ========================= CONFIG =========================
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "client_config.json")
HOTKEY = keyboard.Key.alt_r  # Right Option (⌥) on macOS / Right Alt on Win/Linux

def _fix_url(u):
    u = (u or "").strip()
    if u and not u.startswith(("http://", "https://")):
        u = "http://" + u
    return u.rstrip("/")

def load_client_config():
    cfg = {
        "GAIN": 4.0,
        "MARMOT_SERVER": None,
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                loaded = json.load(f)
            cfg.update({k: v for k, v in loaded.items() if k in cfg})
        except Exception:
            pass

    needs_save = False
    if not cfg.get("MARMOT_SERVER"):
        srv = input("\nEnter Marmot server address (host:port) [default: localhost:5000]: ").strip()
        if not srv:
            srv = "localhost:5000"
        cfg["MARMOT_SERVER"] = srv
        needs_save = True

    if needs_save:
        try:
            with open(CONFIG_PATH, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"✅ Saved config to {CONFIG_PATH}")
        except Exception as e:
            print("⚠️  Could not save config:", e)
    return cfg

config = load_client_config()
GAIN = float(config.get("GAIN", 4.0))
MARMOT_SERVER = _fix_url(config.get("MARMOT_SERVER", "localhost:5000"))
if not MARMOT_SERVER.startswith("http"):
    MARMOT_BASE = f"http://{MARMOT_SERVER}"
else:
    MARMOT_BASE = MARMOT_SERVER

print(f"🐹 Marmot Agent client")
print(f"   Server: {MARMOT_BASE}/connect")
print(f"   Gain:   {GAIN}x")
print()

# ====================== AUDIO PLAYBACK ======================
def play_wav(path):
    """Play WAV using sounddevice (cross-platform, reuses deps).
    All playback goes through playback_lock so proactive and normal messages never overlap audio.
    """
    with playback_lock:
        try:
            with wave.open(path, 'rb') as wf:
                sr = wf.getframerate()
                nch = wf.getnchannels()
                sw = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())
                if sw == 2:
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                elif sw == 1:
                    audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128.0
                elif sw == 4:
                    audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    audio = np.frombuffer(frames, dtype=np.float32)
                if nch > 1:
                    audio = audio.reshape(-1, nch)
                sd.play(audio, samplerate=sr)
                sd.wait()
            print("🔊 Playback done")
        except Exception as e:
            print("Playback error:", e)

# ====================== CLIPBOARD ======================
SYSTEM = platform.system()

def copy_to_clipboard(text):
    if not text:
        return
    try:
        if SYSTEM == "Darwin":
            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        elif SYSTEM == "Windows":
            subprocess.run(["clip"], input=text.encode("utf-8"), check=True)
        elif SYSTEM == "Linux":
            try:
                subprocess.run(["wl-copy"], input=text.encode("utf-8"), check=True)
            except FileNotFoundError:
                subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode("utf-8"), check=True)
        print("📋 Copied to clipboard")
    except Exception as e:
        print(f"Clipboard failed ({SYSTEM}): {e}")


def is_audio_playing():
    """Non-blocking check: returns True if another message is currently being spoken."""
    acquired = playback_lock.acquire(blocking=False)
    if acquired:
        playback_lock.release()
        return False
    return True


def _is_currently_sending():
    with sending_lock:
        return is_sending


def _enqueue_proactive(text, audio_b64):
    """Buffer a proactive that we received from the server but couldn't present yet
    because the client was busy. These will be drained later when idle.
    """
    with pending_proactive_lock:
        if len(pending_proactive_queue) >= MAX_LOCAL_PROACTIVE_QUEUE:
            dropped = pending_proactive_queue.pop(0)
            print(f"🗑️  Dropped oldest buffered proactive (local queue full): {dropped['text'][:50]}...")
        pending_proactive_queue.append({"text": text, "audio": audio_b64})
        print(f"📥 Buffered proactive (client busy). Local queue size={len(pending_proactive_queue)}")


def _try_drain_proactive():
    """If the client is sufficiently unblocked, play the next buffered proactive (if any).
    Returns True if we presented one. Playback will naturally wait for any current audio
    via playback_lock inside play_wav.
    """
    item = None
    # Check recording first (don't interrupt the mic)
    if recording:
        return False

    with pending_proactive_lock:
        if not pending_proactive_queue:
            return False
        # One more recording check after acquiring the queue lock (best effort)
        if recording:
            return False
        item = pending_proactive_queue.pop(0)

    if item:
        # Final safety: if recording started in the last moment, put it back
        if recording:
            with pending_proactive_lock:
                pending_proactive_queue.insert(0, item)
            return False

        print(f"📤 Playing buffered proactive: {item['text'][:80]}{'...' if len(item['text']) > 80 else ''}")
        handle_response("", item["text"], item.get("audio"), proactive=True)
        # Natural pause after a spoken proactive before we consider the next thing
        time.sleep(0.75)
        return True
    return False


# ====================== SEND TO SERVER ======================
def send_to_marmot(audio_path=None, text=None):
    url = f"{MARMOT_BASE}/connect"
    try:
        if audio_path and os.path.exists(audio_path):
            print("📤 Sending audio to Marmot server...")
            with open(audio_path, "rb") as f:
                files = {"file": f}
                resp = requests.post(url, files=files, timeout=300)
        else:
            print(f"📤 Sending text: {text[:80]}{'...' if text and len(text)>80 else ''}")
            resp = requests.post(url, json={"text": text or ""}, timeout=300)

        if resp.status_code != 200:
            print(f"Server error {resp.status_code}: {resp.text[:200]}")
            return None, None, None

        data = resp.json()
        transcription = data.get("transcription", "")
        resp_text = data.get("text", "")
        audio_b64 = data.get("audio")
        return transcription, resp_text, audio_b64
    except Exception as e:
        print("Send failed:", e)
        return None, None, None

def handle_response(transcription, resp_text, audio_b64, proactive=False):
    if resp_text is None:
        return

    if transcription:
        print(f"🗣️  You: {transcription}")

    prefix = "🐹 Marmot (proactive): " if proactive else "🐹 Marmot: "
    print(f"{prefix}{resp_text}\n")
    copy_to_clipboard(resp_text)
    if audio_b64:
        try:
            audio_bytes = base64.b64decode(audio_b64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                play_wav(tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        except Exception as e:
            print("Audio decode/play error:", e)
    else:
        print("(no audio returned)")

# ====================== RECORDING (borrowed from spark-dictate) ======================
recording = False
audio_data = []
stream = None
lock = threading.Lock()

# Playback serialization: prevent overlapping audio (some TTS responses are long)
playback_lock = threading.Lock()
# Track when we're in the middle of a normal user-initiated send/response cycle
sending_lock = threading.Lock()
is_sending = False

# Small client-side queue for proactives that arrived via /poll while the client
# was busy (recording / playing audio / sending). They are played as soon as the
# client becomes unblocked. The server has already committed them to conversation
# history at delivery time.
MAX_LOCAL_PROACTIVE_QUEUE = 4
pending_proactive_queue = []
pending_proactive_lock = threading.Lock()

def callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    with lock:
        audio_data.append(indata.copy())

def start_recording():
    global stream, audio_data, recording
    with lock:
        audio_data = []
        recording = True
    print("🎤 Recording... (hold Right ⌥ / Alt)")
    try:
        stream = sd.InputStream(samplerate=16000, channels=1, dtype="float32", callback=callback)
        stream.start()
    except Exception as e:
        print("Mic start failed:", e)
        recording = False

def stop_recording():
    global stream, recording
    print("⏹️  Stopping...")
    with lock:
        recording = False
    if stream:
        stream.stop()
        stream.close()
        stream = None
    process_and_send()

def process_and_send():
    global audio_data, is_sending
    if not audio_data:
        print("No audio captured")
        return

    arr = np.concatenate(audio_data, axis=0).flatten()
    peak = np.max(np.abs(arr))
    print(f"🔊 Peak: {peak:.4f}")

    boosted = (arr * GAIN).clip(-1.0, 1.0)
    # 0.5s silence pad front+back like spark
    silence = np.zeros(int(16000 * 0.5), dtype=np.int16)
    pcm = (boosted * 32767).astype(np.int16)
    padded = np.concatenate([silence, pcm, silence])

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(padded.tobytes())

    with sending_lock:
        is_sending = True
    try:
        transcription, resp_text, audio_b64 = send_to_marmot(audio_path=tmp_path)
        handle_response(transcription, resp_text, audio_b64)
    finally:
        with sending_lock:
            is_sending = False
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        # Opportunistically play any proactives that were buffered while we were busy
        if not recording:
            _try_drain_proactive()

# ====================== HOTKEY ======================
def on_press(key):
    global recording
    if key == HOTKEY and not recording:
        threading.Thread(target=start_recording, daemon=True).start()

def on_release(key):
    global recording
    if key == HOTKEY and recording:
        threading.Thread(target=stop_recording, daemon=True).start()

def signal_handler(sig, frame):
    print("\n👋 Shutting down...")
    if stream:
        stream.stop()
        stream.close()
    os._exit(0)

# ====================== TEXT MESSAGE MODE (-m) ======================
def send_message_mode(message: str):
    global is_sending
    print(f"🐹 Sending message: {message}")
    with sending_lock:
        is_sending = True
    try:
        transcription, resp_text, audio_b64 = send_to_marmot(text=message)
        handle_response(transcription, resp_text, audio_b64)
    finally:
        with sending_lock:
            is_sending = False
        print("Done.")
        # Opportunistically play any proactives that were buffered while we were busy
        if not recording:
            _try_drain_proactive()


# ====================== PROACTIVE POLLER (server can initiate via /poll) ======================
def proactive_poller():
    """Background thread.

    Strategy:
    - We maintain a small local queue (pending_proactive_queue) for proactives that the
      server delivered (and already committed to conversation history) while we were busy.
    - Whenever we are in an idle window (not recording, not sending), we first try to
      drain one buffered proactive. The actual audio playback is still serialized by
      playback_lock, so it will wait for any in-progress speech to finish.
    - Only when the client is fully unblocked do we poll the server for *new* proactives.
    - This gives the "small queue for proactives that get blocked" behavior.
    """
    print("   (proactive poller active — server can initiate conversations when idle)")
    base_poll_wait = 1.2  # seconds for the long-poll wait param

    while True:
        try:
            # === Drain any proactives we previously buffered while busy ===
            # We do this opportunistically whenever recording and sending are clear.
            # (is_audio_playing() is allowed — the drain will block on playback_lock if needed.)
            if not recording and not _is_currently_sending():
                _try_drain_proactive()

            # === Conservative checks before polling the *server* for fresh messages ===
            # We avoid asking for new ones while busy so we don't accumulate too many
            # server-side, and we give the user space.
            if recording:
                time.sleep(0.35)
                continue
            if is_audio_playing():
                time.sleep(0.3)
                continue
            if _is_currently_sending():
                time.sleep(0.3)
                continue

            # Poll the server (with moderate wait for decent latency)
            try:
                resp = requests.get(
                    f"{MARMOT_BASE}/poll",
                    params={"wait": base_poll_wait},
                    timeout=base_poll_wait + 2.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("action") == "initiate":
                        msg = data.get("message") or {}
                        text = (msg.get("text") or "").strip()
                        audio_b64 = msg.get("audio")
                        if text:
                            # Final safety checks right before presenting a fresh one
                            if not recording and not is_audio_playing() and not _is_currently_sending():
                                handle_response("", text, audio_b64, proactive=True)
                                time.sleep(0.9)
                            else:
                                # Client became busy between poll and now, or during the wait.
                                # Buffer it locally so it plays as soon as we're unblocked.
                                _enqueue_proactive(text, audio_b64)
            except requests.exceptions.RequestException:
                # Server unreachable or slow — back off
                time.sleep(2.5)
                continue
            except Exception as e:
                print("Poller response error:", e)
                time.sleep(2.0)
                continue

            # Small natural idle between checks when using wait (keeps CPU low)
            time.sleep(0.25)

        except Exception as e:
            print("Poller outer error:", e)
            time.sleep(3.0)


# ====================== MAIN ======================
def main():
    parser = argparse.ArgumentParser(description="Marmot Agent Client")
    parser.add_argument("-m", "--message", type=str, help="Send text message, play/print response, then exit")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    if args.message:
        send_message_mode(args.message)
        return

    print("   Hold Right Option (⌥) / Right Alt to speak → release for AI response")
    print("   Use -m \"your text\" for quick text queries (no recording)")
    print()

    # Start background poller only for interactive (hotkey) mode.
    # It will respect recording/speaking/sending so it never interrupts the user.
    poller = threading.Thread(target=proactive_poller, daemon=True)
    poller.start()

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            signal_handler(None, None)

if __name__ == "__main__":
    main()
