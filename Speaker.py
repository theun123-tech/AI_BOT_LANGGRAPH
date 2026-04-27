"""
Speaker.py — Cartesia TTS + Recall.ai audio delivery

Supports two modes:
  - REST (_synthesise): Returns complete MP3. Used for fallback + pre-baked audio.
  - WebSocket streaming (_stream_tts): Yields PCM chunks. Used for Output Media.
"""

import os
import base64
import asyncio
import json
import httpx
import io
import hashlib
import platform
import re

if platform.system() == "Windows":
    os.environ.setdefault("FFMPEG_BINARY",  r"C:\Users\user\Downloads\ffmpeg-8.1-full_build\bin\ffmpeg.exe")
    os.environ.setdefault("FFPROBE_BINARY", r"C:\Users\user\Downloads\ffmpeg-8.1-full_build\bin\ffprobe.exe")

from pydub import AudioSegment

NOISE_FILE   = "freesound_community-office-ambience-24734 (1).mp3"
NOISE_SLICES = 20

# ── Number to words for TTS ──────────────────────────────────────────────────
_DIGIT_WORDS = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
}

def _prep_for_tts(text: str) -> str:
    """Convert numbers to spoken form for TTS clarity."""
    # SCRUM-15 → "SCRUM fifteen"
    def _ticket_repl(m):
        prefix = m.group(1)
        num = m.group(2)
        spoken = " ".join(_DIGIT_WORDS.get(d, d) for d in num)
        return f"{prefix} {spoken}"

    text = re.sub(r'\b([A-Z]+-?)(\d+)\b', _ticket_repl, text)

    # Standalone numbers: 123 → "one two three"
    def _num_repl(m):
        return " ".join(_DIGIT_WORDS.get(d, d) for d in m.group(0))

    text = re.sub(r'(?<![A-Za-z])\b(\d{2,})\b(?![A-Za-z])', _num_repl, text)
    return text


def _mix_noise(voice_bytes: bytes, noise_slices: list, text: str) -> tuple[bytes, int]:
    try:
        voice       = AudioSegment.from_file(io.BytesIO(voice_bytes)).fade_in(80)
        duration_ms = len(voice)
        hash_val    = int(hashlib.md5(text.encode()).hexdigest(), 16)
        slice_idx   = hash_val % len(noise_slices)
        noise_seg   = noise_slices[slice_idx]
        loops       = (duration_ms // len(noise_seg)) + 2
        noise       = (noise_seg * loops)[:duration_ms]
        noise       = noise + 3
        noise       = noise.low_pass_filter(4000)
        combined    = voice.overlay(noise, gain_during_overlay=-3)
        output      = io.BytesIO()
        combined.export(output, format="mp3", bitrate="64k")
        return output.getvalue(), duration_ms
    except Exception as e:
        print(f"[Speaker] Noise failed: {e}")
        return voice_bytes, get_duration_ms(voice_bytes)


def get_duration_ms(audio_bytes: bytes) -> int:
    try:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        return len(seg)
    except Exception:
        return int((len(audio_bytes) * 8) / (48 * 1000) * 1000)


RECALL_REGION   = os.environ.get("RECALLAI_REGION", "ap-northeast-1")
RECALL_API_BASE = f"https://{RECALL_REGION}.recall.ai/api/v1"

CARTESIA_VOICE_ID = "79a125e8-cd45-4c13-8a67-188112f4dd22"
CARTESIA_MODEL    = "sonic-turbo"
CARTESIA_WS_URL   = "wss://api.cartesia.ai/tts/websocket"


class CartesiaSpeaker:
    def __init__(self, bot_id: str = None):
        import Speaker as _self_module
        print(f"[Speaker] Loaded from: {_self_module.__file__}")
        self.recall_key = os.environ["RECALLAI_API_KEY"]
        self.bot_id     = bot_id

        base_dir = os.path.dirname(os.path.abspath(__file__))
        noise_path = os.path.join(base_dir, NOISE_FILE)
        self._noise_slices = []
        try:
            full_noise = AudioSegment.from_file(noise_path)
            slice_len = len(full_noise) // NOISE_SLICES
            self._noise_slices = [full_noise[i * slice_len:(i + 1) * slice_len] for i in range(NOISE_SLICES)]
        except Exception as e:
            print(f"[Speaker] Noise load failed (not critical): {e}")

        self._base_noise = self._noise_slices if self._noise_slices else None
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)

        # Multi-key Cartesia setup
        self._cartesia_keys = []
        for key_name in ["CARTESIA_API_KEY", "CARTESIA_API_KEY_2", "CARTESIA_API_KEY_3", "CARTESIA_API_KEY_4", "CARTESIA_API_KEY_5"]:
            val = os.environ.get(key_name, "").strip()
            if val:
                self._cartesia_keys.append(val)

        if not self._cartesia_keys:
            raise ValueError("No CARTESIA_API_KEY found in environment")

        print(f"[Speaker] {len(self._cartesia_keys)} Cartesia key(s) loaded")
        self._key_index = 0
        self._failed_keys = set()  # Keys that failed at runtime (blacklisted for session)

        self._cartesia_client = httpx.AsyncClient(timeout=30, limits=limits)
        self._recall_client   = httpx.AsyncClient(timeout=30, limits=limits)
        self._recall_headers  = {
            "Authorization": f"Token {self.recall_key}",
            "Content-Type":  "application/json",
            "accept":        "application/json",
        }

        # WebSocket connection (persistent, reused across TTS calls)
        self._cartesia_ws = None
        self._ws_lock = asyncio.Lock()
        self._context_counter = 0

    async def warmup(self):
        valid_keys = []
        for i, key in enumerate(self._cartesia_keys):
            try:
                headers = {"Authorization": f"Bearer {key}", "Cartesia-Version": "2025-04-16", "Content-Type": "application/json"}
                response = await self._cartesia_client.post(
                    "https://api.cartesia.ai/tts/bytes", headers=headers,
                    json={
                        "model_id": CARTESIA_MODEL, "transcript": "hi",
                        "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
                        "language": "en",
                        "output_format": {"container": "mp3", "sample_rate": 44100, "bit_rate": 192000},
                    },
                )
                if response.status_code in (200, 201):
                    valid_keys.append(key)
                    print(f"[Speaker] ✅ Cartesia key #{i+1} valid")
                else:
                    print(f"[Speaker] ❌ Cartesia key #{i+1} invalid ({response.status_code})")
            except Exception as e:
                print(f"[Speaker] ❌ Cartesia key #{i+1} failed: {e}")

        if valid_keys:
            self._cartesia_keys = valid_keys
            print(f"[Speaker] ✅ {len(valid_keys)} valid key(s), Cartesia warmed up")
        else:
            print(f"[Speaker] ⚠️  No valid Cartesia keys!")

    def _next_key(self) -> str:
        """Get next Cartesia key, skipping any that have failed at runtime.
        Falls back to any key if all are blacklisted (maybe they've recovered)."""
        attempts = 0
        while attempts < len(self._cartesia_keys):
            key = self._cartesia_keys[self._key_index % len(self._cartesia_keys)]
            self._key_index += 1
            attempts += 1
            if key not in self._failed_keys:
                self._current_key = key
                return key
        # All keys blacklisted — fall back to any (maybe quota reset, etc)
        key = self._cartesia_keys[self._key_index % len(self._cartesia_keys)]
        self._key_index += 1
        self._current_key = key
        print(f"[Speaker] ⚠️  All keys blacklisted — retrying anyway")
        return key

    def _blacklist_current_key(self, reason: str = ""):
        """Mark the currently-used key as failed for the rest of the session."""
        key = getattr(self, '_current_key', None)
        if key and key not in self._failed_keys:
            self._failed_keys.add(key)
            # Find which key number this was for logging
            key_num = self._cartesia_keys.index(key) + 1 if key in self._cartesia_keys else "?"
            active = len(self._cartesia_keys) - len(self._failed_keys)
            print(f"[Speaker] 🚫 Blacklisted key #{key_num} ({reason}) — {active} key(s) remaining")

    def _next_cartesia_headers(self) -> dict:
        key = self._next_key()
        return {"Authorization": f"Bearer {key}", "Cartesia-Version": "2025-04-16", "Content-Type": "application/json"}

    # ══════════════════════════════════════════════════════════════════════════
    # REST TTS — returns complete MP3 (used for fallback + pre-baked audio)
    # ══════════════════════════════════════════════════════════════════════════

    async def _synthesise(self, text: str) -> bytes:
        text = _prep_for_tts(text)
        headers = self._next_cartesia_headers()
        key_num = (self._key_index - 1) % len(self._cartesia_keys) + 1
        print(f"[Speaker] TTS via Cartesia (key #{key_num})...")
        response = await self._cartesia_client.post(
            "https://api.cartesia.ai/tts/bytes", headers=headers,
            json={
                "model_id": CARTESIA_MODEL, "transcript": text,
                "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
                "language": "en",
                "output_format": {"container": "mp3", "sample_rate": 44100, "bit_rate": 192000},
            },
        )
        response.raise_for_status()
        return response.content

    # ══════════════════════════════════════════════════════════════════════════
    # WebSocket TTS — yields PCM chunks (used for Output Media streaming)
    # ══════════════════════════════════════════════════════════════════════════

    async def _ensure_ws_connected(self):
        """Connect to Cartesia WebSocket if not already connected."""
        if self._cartesia_ws is not None:
            try:
                # Check if still open
                await self._cartesia_ws.ping()
                return
            except Exception:
                self._cartesia_ws = None

        import websockets
        key = self._next_key()
        url = f"{CARTESIA_WS_URL}?api_key={key}&cartesia_version=2025-04-16"
        self._cartesia_ws = await websockets.connect(url, ping_interval=20, ping_timeout=10)
        print("[Speaker] ✅ Cartesia WebSocket connected")

    async def _close_ws(self):
        """Close WebSocket so next call reconnects (with next key on retry)."""
        if self._cartesia_ws:
            try:
                await self._cartesia_ws.close()
            except Exception:
                pass
        self._cartesia_ws = None

    async def _stream_tts(self, text: str):
        """Stream TTS as PCM s16le chunks via Cartesia WebSocket.

        Rotates API keys on failure (WS disconnect, error message, 0-byte response).
        Cannot retry mid-stream — once the first chunk is yielded, any subsequent
        error is raised to the caller since audio has already been committed.

        Yields: bytes — raw PCM s16le chunks at 48kHz mono.
        """
        text = _prep_for_tts(text)
        self._context_counter += 1
        context_id = f"ctx-{self._context_counter}"

        max_retries = max(1, len(self._cartesia_keys))
        last_error = None
        first_chunk_received = False

        for attempt in range(max_retries):
            # Connect (or reconnect with next key after prior failure)
            try:
                async with self._ws_lock:
                    await self._ensure_ws_connected()
            except Exception as e:
                last_error = e
                print(f"[Speaker] ⚠️  Cartesia WS connect failed (attempt {attempt+1}/{max_retries}): {e}")
                continue

            # Send TTS request
            request = {
                "model_id": CARTESIA_MODEL,
                "transcript": text,
                "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
                "language": "en",
                "context_id": context_id,
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": 48000,
                },
                "continue": False,
            }

            total_bytes = 0
            stream_error = None

            try:
                await self._cartesia_ws.send(json.dumps(request))

                while True:
                    raw = await asyncio.wait_for(self._cartesia_ws.recv(), timeout=10)
                    msg = json.loads(raw)

                    if msg.get("context_id") != context_id:
                        continue

                    msg_type = msg.get("type", "")

                    # Explicit error from Cartesia (bad key, quota exhausted, etc)
                    if msg_type == "error":
                        err_msg = msg.get("error") or msg.get("message") or str(msg)[:200]
                        print(f"[Speaker] ❌ Cartesia error (attempt {attempt+1}/{max_retries}): {err_msg}")
                        stream_error = Exception(f"Cartesia error: {err_msg}")
                        break

                    # Audio chunk — yield to caller
                    if msg_type == "chunk" and msg.get("data"):
                        pcm_bytes = base64.b64decode(msg["data"])
                        total_bytes += len(pcm_bytes)
                        if not first_chunk_received:
                            print(f"[Speaker] 🔊 First PCM chunk ({len(pcm_bytes)} bytes)")
                            first_chunk_received = True
                        yield pcm_bytes
                    # Unexpected message type — log for debugging
                    elif msg_type and msg_type != "chunk" and not msg.get("done"):
                        print(f"[Speaker] ℹ️  Cartesia msg type={msg_type}: {str(msg)[:200]}")

                    # Stream finished
                    if msg.get("done"):
                        break

            except Exception as e:
                stream_error = e
                print(f"[Speaker] ⚠️  Stream TTS error (attempt {attempt+1}/{max_retries}): {e}")

            # Evaluate outcome
            if total_bytes > 0 and stream_error is None:
                # Success
                duration_ms = (total_bytes / 2 / 48000) * 1000
                print(f"[Speaker] ✅ Streamed {total_bytes} bytes ({duration_ms:.0f}ms audio)")
                return

            # Failure path — close WS so next attempt reconnects with new key
            await self._close_ws()
            last_error = stream_error or Exception("Cartesia returned 0 bytes")

            if first_chunk_received:
                # Already yielded audio — cannot retry cleanly
                print(f"[Speaker] ⚠️  Mid-stream failure after {total_bytes} bytes — cannot retry")
                raise last_error

            # Pre-first-chunk failure — blacklist this key and rotate
            reason = str(last_error)[:60] if last_error else "0 bytes"
            self._blacklist_current_key(reason)

            if attempt < max_retries - 1:
                print(f"[Speaker] 🔁 Rotating to next Cartesia key...")

        # All keys exhausted
        raise Exception(f"All {max_retries} Cartesia TTS attempt(s) failed. Last error: {last_error}")

    # ══════════════════════════════════════════════════════════════════════════
    # Recall.ai audio injection (fallback mode)
    # ══════════════════════════════════════════════════════════════════════════

    async def _inject_into_meeting(self, b64_audio: str):
        if not self.bot_id:
            return
        if os.environ.get("DEBUG_SAVE_AUDIO", "").lower() in ("1", "true", "yes"):
            try:
                raw = base64.b64decode(b64_audio)
                self._debug_audio_counter = getattr(self, '_debug_audio_counter', 0) + 1
                fname = f"debug_inject_{self._debug_audio_counter:03d}.mp3"
                with open(fname, "wb") as f:
                    f.write(raw)
            except Exception:
                pass

        response = await self._recall_client.post(
            f"{RECALL_API_BASE}/bot/{self.bot_id}/output_audio/",
            headers=self._recall_headers,
            json={"kind": "mp3", "b64_data": b64_audio},
        )
        if response.status_code not in (200, 201):
            print(f"[Speaker] Inject error {response.status_code}: {response.text}")
        else:
            print("[Speaker] Audio injected")

    async def stop_audio(self):
        if not self.bot_id:
            return
        try:
            response = await self._recall_client.delete(
                f"{RECALL_API_BASE}/bot/{self.bot_id}/output_audio/",
                headers=self._recall_headers,
            )
            if response.status_code == 204:
                print("[Speaker] ⏹️  Audio stopped")
        except Exception as e:
            print(f"[Speaker] Stop audio error: {e}")

    async def close(self):
        if self._cartesia_ws:
            try:
                await self._cartesia_ws.close()
            except Exception:
                pass
        await asyncio.gather(
            self._cartesia_client.aclose(),
            self._recall_client.aclose(),
        )