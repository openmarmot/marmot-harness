#!/usr/bin/env python3
"""
Marmot Agent Client

- Hold Right Option/Alt to record -> send audio to local Marmot server /connect
- Server does STT + LLM (tools) + TTS
- Client receives transcription + AI response + audio
- Prints "You:" (transcription) then "Marmot:" reply, plays audio, copies reply to clipboard
- -m "text" flag: send text directly, play/print/copy response, exit (for testing)
- Proactive messages from server are now gated by detect_human() (camera snapshot sent to
  server's /detect; only spoken if a "person"/"human" object is reported).
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
    Before playing any proactive, detect_human() is called (camera snapshot + server /detect)
    so we only speak when a person is actually present to hear it.
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

        # === Human presence gate: only speak proactive messages if someone is actually there ===
        if not detect_human():
            print("👤 No human visible — deferring buffered proactive (will retry when present).")
            with pending_proactive_lock:
                pending_proactive_queue.insert(0, item)
            defer_sleep = BACKOFF_INTERVAL if _is_in_backoff_mode() else 1.2
            time.sleep(defer_sleep)  # back off to ~1/min when no recent user activity
            return False

        print(f"📤 Playing buffered proactive: {item['text'][:80]}{'...' if len(item['text']) > 80 else ''}")
        handle_response("", item["text"], item.get("audio"), proactive=True)
        # Natural pause after a spoken proactive before we consider the next thing
        time.sleep(0.75)
        return True
    return False


def _mark_user_interaction():
    """Record that the user is present (either via hotkey or successful camera human detection)."""
    global last_user_interaction
    last_user_interaction = time.time()


def _is_in_backoff_mode() -> bool:
    """True if we have not seen a user interaction (record button or human on camera) recently."""
    return (time.time() - last_user_interaction) > USER_INTERACTION_TIMEOUT


# ====================== CAMERA + HUMAN PRESENCE (for gating proactive speech) ======================
def capture_camera_image(timeout: float = 6.0) -> bytes:
    """Capture one frame from the default webcam. Returns JPEG bytes or b'' on failure.

    Uses OpenCV (opencv-python) which is now listed in requirements.txt.
    The imagesnap CLI fallback is kept only for macOS as a last resort.
    """
    # Primary path: OpenCV
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            print("👁️  OpenCV could not open camera (index 0). Check permissions or camera in use.")
            # fall through to imagesnap fallback on macOS
        else:
            # Let auto-exposure / white balance settle
            for _ in range(5):
                cap.read()
                time.sleep(0.07)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    return buf.tobytes()
            print("👁️  OpenCV read a frame but it was empty.")
    except ImportError:
        print("👁️  opencv-python not installed. Run: pip install opencv-python")
    except Exception as e:
        print("👁️  OpenCV camera error:", e)

    # Last-resort macOS fallback (imagesnap). Users should prefer the opencv-python path.
    if platform.system() == "Darwin":
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            cmd = ["imagesnap", "-q", tmp_path]
            res = subprocess.run(cmd, capture_output=True, timeout=timeout)
            if res.returncode == 0 and tmp_path and os.path.exists(tmp_path):
                with open(tmp_path, "rb") as f:
                    data = f.read()
                return data if data else b""
        except FileNotFoundError:
            pass  # imagesnap not installed
        except Exception:
            pass
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    return b""


def detect_human() -> bool:
    """Capture an image from the Mac's camera and ask the Marmot server /detect endpoint
    whether a human/person is visible. Returns True only if the server reports a relevant label.
    """
    img = capture_camera_image()
    if not img:
        print("👁️  Camera capture failed — treating as no human present (safe default).")
        return False

    try:
        url = f"{MARMOT_BASE}/detect"
        files = {"image": ("cam.jpg", img, "image/jpeg")}
        resp = requests.post(url, files=files, timeout=35)
        if resp.status_code != 200:
            print(f"👁️  /detect failed ({resp.status_code}) — skipping proactive.")
            return False

        data = resp.json() or {}
        objects = [str(x).lower() for x in (data.get("objects") or [])]
        # YOLO COCO typically reports "person"; support "human" too for flexibility
        human = any(label in ("person", "human") for label in objects)
        if human:
            _mark_user_interaction()  # Successful human detection counts as user interaction/presence
        print(f"👁️  Camera saw: {data.get('objects')} → human_present={human}")
        return human
    except Exception as e:
        print("👁️  Human detection request error:", e)
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

# Last "user presence" marker for backoff logic.
# Updated on: user pressing the record hotkey, or successful detect_human() (human seen by camera).
last_user_interaction = time.time()
USER_INTERACTION_TIMEOUT = 5 * 60   # 5 minutes with no interaction → enter backoff
NORMAL_POLL_WAIT = 1.2              # seconds for normal /poll long-wait
BACKOFF_POLL_WAIT = 10.0            # server caps long-poll at 10s
BACKOFF_INTERVAL = 60.0             # target check rate (poll + camera) when backed off

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
    _mark_user_interaction()  # User is actively present (hit the hotkey)
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
    - All proactive playback (fresh or buffered) is gated by detect_human() — we only
      speak server-initiated messages when the camera sees a person ("human"/"person" label
      from the YOLO server via /detect). If no one is there we defer and retry later.
    - After 5 minutes with no "user interaction" (record hotkey press or successful
      detect_human()), the poller and camera checks back off to approximately once per minute
      to avoid unnecessary work / camera use when the user is away.
    """
    print("   (proactive poller active — server can initiate conversations when idle)")
    print("   (proactive speech is gated by camera human detection via opencv + server /detect)")
    print("   (polling + camera checks back off to ~1/min after 5 min with no user interaction)")

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

            # Choose poll aggressiveness based on recent user activity (record button or camera seeing a human)
            if _is_in_backoff_mode():
                poll_wait = BACKOFF_POLL_WAIT      # use server's max long-poll (10s)
                post_poll_sleep = BACKOFF_INTERVAL - poll_wait  # ~50s to reach ~1/min total
            else:
                poll_wait = NORMAL_POLL_WAIT
                post_poll_sleep = 0.25

            # Poll the server (long-poll when possible)
            try:
                resp = requests.get(
                    f"{MARMOT_BASE}/poll",
                    params={"wait": poll_wait},
                    timeout=poll_wait + 2.0
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
                                if detect_human():
                                    handle_response("", text, audio_b64, proactive=True)
                                    time.sleep(0.9)
                                else:
                                    # No one in front of the camera — buffer locally.
                                    # The drain path will re-check presence before speaking.
                                    _enqueue_proactive(text, audio_b64)
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

            # Natural idle / backoff sleep (keeps CPU + camera low when user is away)
            if post_poll_sleep > 0:
                time.sleep(post_poll_sleep)

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
