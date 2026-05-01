"""
key_rotator.py — Shared multi-key rotator for all external API services.

Universal pattern:
  Every paid API key is stored in ONE comma-separated env var per service:
      SERVICE_API_KEYS = "key1,key2,key3,...,keyN"

  Backward compatibility: also reads the singular form (SERVICE_API_KEY).

Two rotation modes:
  Mode B — per-request round-robin:    key_for_request("CARTESIA")
  Mode A — sticky per session/bot:     key_for_session("DEEPGRAM", session_id)

Failure handling:
  When a key returns 401/429, call mark_key_failed(...) to put it in a
  short cooldown. Rotator skips cooldowned keys until the cooldown expires.

Add a new service later:
  No code change to this file. Just set NEW_API_KEYS env var and call
  key_for_request("NEW") or key_for_session("NEW", session_id).

Logging:
  Verbose-by-default during rollout. Turn down by setting
  KEY_ROTATOR_VERBOSE=0 in the environment.
"""

import os
import time
import threading
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Module-level state — protected by a single lock for thread-safety
# ─────────────────────────────────────────────────────────────────────────────
_lock = threading.Lock()

# Cached key lists per service: {"CARTESIA": ["key1", "key2", ...], ...}
_keys_cache: Dict[str, List[str]] = {}

# Round-robin counters for Mode B: {"CARTESIA": 7, "GROQ": 142, ...}
_request_counters: Dict[str, int] = {}

# Sticky session→key bindings for Mode A: {"DEEPGRAM": {"sess_abc": "key3"}}
_session_bindings: Dict[str, Dict[str, str]] = {}

# Counter for picking the next key when a new session arrives (Mode A).
# Separate from request counters so per-request and per-session rotations
# don't interfere with each other.
_session_counters: Dict[str, int] = {}

# Failed-key cooldowns: {"CARTESIA": {"key3": expiry_timestamp, ...}}
_failed_keys: Dict[str, Dict[str, float]] = {}

# Default cooldown after a key fails (seconds). Auto-recovers after this.
DEFAULT_COOLDOWN_SECONDS = 60.0


def _verbose() -> bool:
    """Whether to print rotation events. Default ON, off via env var."""
    return os.environ.get("KEY_ROTATOR_VERBOSE", "1") != "0"


def _log(msg: str) -> None:
    if _verbose():
        print(f"[KeyRotator] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Public: key loading
# ─────────────────────────────────────────────────────────────────────────────
def load_keys(service_name: str, force_reload: bool = False) -> List[str]:
    """Load all keys for a service from env vars.

    Reads BOTH:
      - SERVICE_API_KEYS  (plural — comma-separated, preferred)
      - SERVICE_API_KEY   (singular — backward compatibility)

    Special-cases SERPAPI which historically uses SERPAPI_KEYS / SERPAPI_KEY.

    Strips whitespace, drops empties, dedupes while preserving order.
    Caches the result in-process; pass force_reload=True to re-read.

    Returns empty list if no keys configured (caller decides what to do).
    """
    service_name = service_name.upper().strip()

    with _lock:
        if not force_reload and service_name in _keys_cache:
            return list(_keys_cache[service_name])  # defensive copy

    # Two env var families per service:
    #   Most services:  XXX_API_KEYS / XXX_API_KEY
    #   SerpApi:        SERPAPI_KEYS / SERPAPI_KEY (no _API_ infix)
    if service_name == "SERPAPI":
        plural_var = "SERPAPI_KEYS"
        singular_var = "SERPAPI_KEY"
    else:
        plural_var = f"{service_name}_API_KEYS"
        singular_var = f"{service_name}_API_KEY"

    raw_plural = os.environ.get(plural_var, "")
    raw_singular = os.environ.get(singular_var, "")

    # Parse plural (comma-separated)
    keys: List[str] = []
    if raw_plural:
        for part in raw_plural.split(","):
            cleaned = part.strip().strip('"').strip("'").strip()
            if cleaned:
                keys.append(cleaned)

    # Append singular (might be a different key — common during migration)
    if raw_singular:
        cleaned = raw_singular.strip().strip('"').strip("'").strip()
        if cleaned:
            keys.append(cleaned)

    # Also check legacy numbered env vars (CARTESIA_API_KEY_2, _3, ...) for
    # backward compatibility during migration. Stops at first missing slot.
    if service_name != "SERPAPI":
        for i in range(2, 100):  # _2 through _99
            extra = os.environ.get(f"{service_name}_API_KEY_{i}", "")
            if not extra:
                break
            cleaned = extra.strip().strip('"').strip("'").strip()
            if cleaned:
                keys.append(cleaned)

    # Legacy SerpApi numbered slots (SERPAPI_KEY_1 through _17 historically)
    if service_name == "SERPAPI":
        for i in range(1, 100):
            extra = os.environ.get(f"SERPAPI_KEY_{i}", "")
            if not extra:
                continue
            cleaned = extra.strip().strip('"').strip("'").strip()
            if cleaned:
                keys.append(cleaned)

    # Dedupe while preserving order
    seen = set()
    deduped: List[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            deduped.append(k)

    with _lock:
        _keys_cache[service_name] = list(deduped)

    if deduped:
        _log(f"{service_name}: loaded {len(deduped)} key(s)")
    else:
        _log(f"{service_name}: no keys configured (set {plural_var} or {singular_var})")

    return list(deduped)


# ─────────────────────────────────────────────────────────────────────────────
# Internal: pick next available (non-cooldowned) key
# ─────────────────────────────────────────────────────────────────────────────
def _pick_next_key(service_name: str, counter_dict: Dict[str, int]) -> Optional[str]:
    """Round-robin through keys, skipping ones in cooldown.

    counter_dict is either _request_counters (Mode B) or _session_counters
    (Mode A — for new session assignment).

    Returns None if no keys available (all in cooldown or list empty).
    Caller holds the lock.
    """
    keys = _keys_cache.get(service_name, [])
    if not keys:
        return None

    failed = _failed_keys.get(service_name, {})
    now = time.time()

    # Clean up expired cooldowns inline
    expired = [k for k, exp in failed.items() if exp <= now]
    for k in expired:
        failed.pop(k, None)

    # Try every key in round-robin order; return first non-failed one
    n = len(keys)
    start_idx = counter_dict.get(service_name, 0)

    for offset in range(n):
        idx = (start_idx + offset) % n
        candidate = keys[idx]
        if candidate not in failed:
            counter_dict[service_name] = (idx + 1) % n
            return candidate

    # All keys in cooldown — return least-recently-failed (lowest expiry)
    # so caller still gets *something* and can try; better than None for
    # services where total failure means user-visible breakage.
    if failed:
        oldest = min(failed.items(), key=lambda kv: kv[1])
        return oldest[0]

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public: Mode B — per-request rotation
# ─────────────────────────────────────────────────────────────────────────────
def key_for_request(service_name: str) -> Optional[str]:
    """Return the next key in round-robin order for a one-shot API call.

    Loads keys lazily on first call per service. Skips keys currently in
    cooldown. Returns None if no keys are configured at all.

    Use this for: Cartesia, SerpApi, Exa, Groq, Azure — anywhere a single
    API call should rotate to spread load.
    """
    service_name = service_name.upper().strip()

    # Lazy load
    if service_name not in _keys_cache:
        load_keys(service_name)

    with _lock:
        key = _pick_next_key(service_name, _request_counters)

    if key and _verbose():
        keys = _keys_cache.get(service_name, [])
        try:
            idx = keys.index(key) + 1
            _log(f"{service_name}: request → key #{idx} of {len(keys)}")
        except ValueError:
            pass

    return key


# ─────────────────────────────────────────────────────────────────────────────
# Public: Mode A — sticky per session/bot lifetime
# ─────────────────────────────────────────────────────────────────────────────
def key_for_session(service_name: str, session_id: str) -> Optional[str]:
    """Return the same key for the lifetime of one session/bot.

    First call with a new session_id picks the next key in rotation and
    binds it. Subsequent calls with the same session_id return the same
    key — guaranteeing all calls within one bot's lifetime use one key
    and get one rate-limit budget.

    Use this for: Recall.ai (bot lifetime), Deepgram (per-bot STT stream).

    Call release_session(...) when the session ends to free the key
    binding. Not strictly required (memory leak is bounded by session
    count), but tidier.
    """
    service_name = service_name.upper().strip()
    session_id = str(session_id)

    # Lazy load
    if service_name not in _keys_cache:
        load_keys(service_name)

    with _lock:
        bindings = _session_bindings.setdefault(service_name, {})

        # Already bound? Return existing binding (sticky guarantee).
        if session_id in bindings:
            existing = bindings[session_id]
            # Defensive: re-validate the bound key is still in the configured
            # list. If it was removed (e.g. env var changed), rebind.
            if existing in _keys_cache.get(service_name, []):
                return existing
            # Fall through to fresh assignment

        # Fresh assignment
        new_key = _pick_next_key(service_name, _session_counters)
        if new_key:
            bindings[session_id] = new_key

    if new_key and _verbose():
        keys = _keys_cache.get(service_name, [])
        try:
            idx = keys.index(new_key) + 1
            _log(f"{service_name}: session {session_id[:8]}… → locked to key #{idx} of {len(keys)}")
        except ValueError:
            pass

    return new_key


def release_session(service_name: str, session_id: str) -> None:
    """Release a session's key binding when the session ends.

    Optional — bindings are cheap. Useful for very long-running processes
    that handle thousands of sessions, to keep _session_bindings small.
    """
    service_name = service_name.upper().strip()
    session_id = str(session_id)

    with _lock:
        bindings = _session_bindings.get(service_name, {})
        if session_id in bindings:
            released_key = bindings.pop(session_id)
            if _verbose():
                try:
                    idx = _keys_cache.get(service_name, []).index(released_key) + 1
                    _log(f"{service_name}: session {session_id[:8]}… released key #{idx}")
                except ValueError:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# Public: failure marking and recovery
# ─────────────────────────────────────────────────────────────────────────────
def mark_key_failed(service_name: str, key: str,
                    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS) -> None:
    """Mark a key as failed (rate-limited / unauthorized).

    Puts the key in cooldown for `cooldown_seconds`. Rotator skips it
    during this window. Auto-recovers after the window expires — a
    permanently-bad key just stays in cooldown longer (caller can pass
    a very large cooldown if they know the key is dead).

    Safe to call multiple times for the same key — extends the cooldown.
    """
    service_name = service_name.upper().strip()

    with _lock:
        failed = _failed_keys.setdefault(service_name, {})
        failed[key] = time.time() + max(1.0, cooldown_seconds)

    if _verbose():
        keys = _keys_cache.get(service_name, [])
        try:
            idx = keys.index(key) + 1
            _log(f"{service_name}: key #{idx} cooldown {cooldown_seconds:.0f}s")
        except ValueError:
            _log(f"{service_name}: unknown key cooldown {cooldown_seconds:.0f}s")


def is_key_available(service_name: str, key: str) -> bool:
    """Is this key currently outside its cooldown window?"""
    service_name = service_name.upper().strip()

    with _lock:
        failed = _failed_keys.get(service_name, {})
        expiry = failed.get(key)
        if expiry is None:
            return True
        if expiry <= time.time():
            failed.pop(key, None)
            return True
        return False


def status(service_name: str) -> Dict[str, int]:
    """Return a small dict describing rotator state for a service.

    Useful for /healthz endpoints and debug logs.
    """
    service_name = service_name.upper().strip()

    if service_name not in _keys_cache:
        load_keys(service_name)

    with _lock:
        keys = _keys_cache.get(service_name, [])
        failed = _failed_keys.get(service_name, {})
        now = time.time()
        cooldowned = sum(1 for exp in failed.values() if exp > now)
        bindings = _session_bindings.get(service_name, {})

        return {
            "total": len(keys),
            "available": len(keys) - cooldowned,
            "cooldown": cooldowned,
            "active_sessions": len(bindings),
            "next_request_idx": _request_counters.get(service_name, 0),
            "next_session_idx": _session_counters.get(service_name, 0),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: warm-up at startup so keys are loaded before first use
# ─────────────────────────────────────────────────────────────────────────────
def warm_up(service_names: List[str]) -> None:
    """Pre-load keys for a list of services. Optional but recommended at
    server startup so the first request doesn't pay the env-parsing cost.

    Example:
        warm_up(["CARTESIA", "DEEPGRAM", "RECALLAI", "SERPAPI", "EXA",
                 "GROQ", "AZURE"])
    """
    for svc in service_names:
        load_keys(svc)
