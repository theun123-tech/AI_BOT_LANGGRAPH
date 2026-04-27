"""
addressee_decider.py — Multi-party addressee detection with Groq LLM

Decides WHO Sam should respond to in a multi-party meeting.

Flow:
    1. Flux sessions emit final transcripts from multiple speakers
    2. AddresseeDecider collects them in a buffer keyed by speaker
    3. Silence timer starts/restarts on every speech event
    4. After 1s of genuine silence (no speech, Sam not speaking):
       → Fire Groq LLM with full context
       → LLM returns decision: respond_to(speaker) | respond_both(speakers) | none
    5. Decision callback fires in BotSession to trigger response (or stay silent)

State machine:
    LISTENING        — waiting for turn completions
    SILENCE_WAITING  — collecting turns, silence timer running
    LLM_DECIDING     — Groq call in flight (cancellable on new speech)
    RESPONDING       — Sam is speaking; timer paused
    → back to LISTENING after response completes or is interrupted

Standup mode: NOT used. This module is for client mode only.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Optional

from openai import AsyncOpenAI
# Shared Groq rotator (from Fix B) — falls back to single-key if not available
try:
    from groq_client import get_shared_groq_rotator
    _ROTATOR_AVAILABLE = True
except ImportError:
    get_shared_groq_rotator = None
    _ROTATOR_AVAILABLE = False


# Silence window before firing LLM decision
SILENCE_WAIT_SECONDS = 0.1

# Max time to wait in SILENCE_WAITING before forcing a decision
# (prevents getting stuck when speakers rapid-fire without silence gaps)
MAX_SILENCE_WAIT_SECONDS = 5.0

# Groq call timeout — if LLM hangs, we default to "none" (stay silent)
LLM_TIMEOUT_SECONDS = 2.0

# How much history to include in the LLM context
HISTORY_TURNS = 10

# Buffer size for recent turns awaiting a decision
TURN_BUFFER_SIZE = 8


class DeciderState(Enum):
    LISTENING = "listening"
    SILENCE_WAITING = "silence_waiting"
    LLM_DECIDING = "llm_deciding"
    RESPONDING = "responding"


@dataclass
class CompletedTurn:
    """One speaker's finished utterance awaiting a response decision."""
    speaker: str
    text: str
    completed_at: float = field(default_factory=time.time)


@dataclass
class Decision:
    """What the LLM decided — passed to BotSession for action."""
    type: str                      # "respond_to" | "respond_both" | "none"
    speakers: list[str]            # which speaker(s) Sam should address
    text: str                      # combined utterance text for Sam to respond to
    reasoning: str = ""            # LLM's reasoning (for logging)


# ══════════════════════════════════════════════════════════════════════════
# Groq prompt for addressee decision
# ══════════════════════════════════════════════════════════════════════════

ADDRESSEE_PROMPT = """You are helping a voice assistant named Sam decide who to respond to in a multi-party meeting.

Sam is a senior project manager at AnavClouds. Sam answers questions, manages tickets, and helps with project work when addressed. Sam stays silent when participants talk among themselves.

PARTICIPANTS IN THE MEETING:
{participants}

SAM'S LAST RESPONSE:
{sam_last_response}

RECENT CONVERSATION HISTORY (last {history_limit} exchanges, oldest first):
{conversation_history}

UTTERANCES AWAITING SAM'S DECISION (these just finished):
{recent_utterances}

TASK:
Decide who Sam should respond to. Consider:
- Did any speaker explicitly say "Sam" or clearly address him?
- Is the utterance a direct follow-up to Sam's last response?
- Is the utterance clearly directed at another human (e.g., "Vanshita, ...")?
- Is the utterance general chatter or side-conversation between humans?
- Are multiple speakers all asking Sam something?

RESPOND WITH JSON ONLY, no other text:
{{
  "decision": "respond_to" | "respond_both" | "none",
  "speakers": ["SpeakerName"] or ["Speaker1", "Speaker2"] or [],
  "reasoning": "one sentence explanation"
}}

Examples:
- One speaker said "Sam, what's the status?" → {{"decision": "respond_to", "speakers": ["Sahil"], "reasoning": "direct address to Sam"}}
- Two speakers both asked Sam questions → {{"decision": "respond_both", "speakers": ["Sahil", "Vanshita"], "reasoning": "both addressed Sam"}}
- Speakers chatting with each other → {{"decision": "none", "speakers": [], "reasoning": "side conversation between humans"}}
"""


# ══════════════════════════════════════════════════════════════════════════
# Context provider type — BotSession provides this
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class ContextBundle:
    """Data the BotSession provides when AddresseeDecider asks for context."""
    participants: list[str]        # current meeting participants
    sam_last_response: str         # most recent thing Sam said (empty if none)
    conversation_history: list[str]  # formatted "Speaker: text" strings, oldest first


# ══════════════════════════════════════════════════════════════════════════
# AddresseeDecider — the main class
# ══════════════════════════════════════════════════════════════════════════

class AddresseeDecider:
    """Decides who Sam should respond to in multi-party meetings.

    Integration (in BotSession client mode):

        self._addressee_decider = AddresseeDecider(
            groq_api_key=GROQ_API_KEY,
            get_context=self._build_addressee_context,
            on_decision=self._handle_addressee_decision,
            tag=self.tag,
        )

        # On every Flux final transcript:
        self._addressee_decider.on_turn_completed(speaker, text)

        # On every speech event (final or interim, any speaker):
        self._addressee_decider.on_speech_activity()

        # When Sam starts/stops speaking:
        self._addressee_decider.on_sam_speaking_changed(speaking=True|False)

        # On session end:
        await self._addressee_decider.close()
    """

    def __init__(
        self,
        groq_api_key: str,
        get_context: Callable[[], ContextBundle],
        on_decision: Callable[[Decision], Awaitable[None]],
        tag: str = "",
        model: str = "llama-3.1-8b-instant",
    ):
        """
        Args:
            groq_api_key: API key for Groq
            get_context: sync function returning current ContextBundle
                         (called at decision time to snapshot current state)
            on_decision: async callback invoked with the Decision result
            tag: log prefix (e.g., "[S:abc123]")
            model: Groq model name
        """
        # Key management: prefer shared rotator (12-key load spreading),
        # fall back to single-key client if rotator unavailable.
        self._rotator = None
        self._client = None
        self._client_cache: dict = {}  # key → AsyncOpenAI (reused across calls)

        if _ROTATOR_AVAILABLE and get_shared_groq_rotator is not None:
            try:
                self._rotator = get_shared_groq_rotator(tag=tag or "[addressee]")
                if not self._rotator.is_enabled():
                    # Rotator has no keys — fall back to legacy single-key mode
                    self._rotator = None
            except Exception as e:
                print(f"[Addressee] {tag} ⚠️  Rotator init failed, "
                      f"using single key: {e}")
                self._rotator = None

        if self._rotator is None:
            # Legacy single-key path (backward compatible)
            self._client = AsyncOpenAI(
                api_key=groq_api_key,
                base_url="https://api.groq.com/openai/v1",
            )
            print(f"[Addressee] {tag} ℹ️  Using single Groq key "
                  f"(rotator not available)")
        else:
            print(f"[Addressee] {tag} ✅ Using shared Groq rotator "
                  f"({self._rotator.get_key_count()} keys)")

        self._get_context = get_context
        self._on_decision = on_decision
        self._tag = tag
        self._model = model

        # State
        self._state: DeciderState = DeciderState.LISTENING
        self._turn_buffer: list[CompletedTurn] = []
        self._last_activity_at: float = 0.0  # time of last speech event
        self._first_activity_at: float = 0.0  # first turn in current window
        self._sam_speaking: bool = False
        self._silence_timer_task: Optional[asyncio.Task] = None
        self._llm_task: Optional[asyncio.Task] = None
        self._closed: bool = False

    # ── Public API ──────────────────────────────────────────────────────

    def on_turn_completed(self, speaker: str, text: str) -> None:
        """Called when a Flux session emits a FINAL transcript.

        Adds the turn to the buffer and triggers the silence timer.
        """
        if self._closed:
            return

        # Ignore if Sam is speaking — his response handler will resume us
        if self._sam_speaking:
            return

        text = (text or "").strip()
        if not text:
            return

        now = time.time()
        turn = CompletedTurn(speaker=speaker, text=text, completed_at=now)
        self._turn_buffer.append(turn)

        # Cap buffer size (in case no silence ever arrives)
        if len(self._turn_buffer) > TURN_BUFFER_SIZE:
            self._turn_buffer = self._turn_buffer[-TURN_BUFFER_SIZE:]

        # Track first turn in this decision window
        if self._first_activity_at == 0.0:
            self._first_activity_at = now

        # Reset silence timer (treat final transcript as speech activity)
        self._mark_activity_and_restart_timer()

    def on_speech_activity(self) -> None:
        """Called when ANY speech event occurs (final OR interim).

        Resets the silence timer. Interims do NOT add to turn buffer
        (only finals do in on_turn_completed), but they reset the clock.
        """
        if self._closed or self._sam_speaking:
            return
        self._mark_activity_and_restart_timer()

    def on_sam_speaking_changed(self, speaking: bool) -> None:
        """Called when Sam starts or stops producing audio.

        Pauses the decider while Sam is speaking. When Sam stops, we don't
        auto-fire a decision — we wait for the next turn completion to trigger.
        """
        if self._closed:
            return

        previously = self._sam_speaking
        self._sam_speaking = speaking

        if speaking and not previously:
            # Sam just started talking — pause any pending silence timer
            self._cancel_silence_timer()
            self._state = DeciderState.RESPONDING
            # Cancel any in-flight LLM call (Sam's already responding)
            self._cancel_llm_task()

        elif not speaking and previously:
            # Sam just stopped — if there are buffered turns from interrupts,
            # we'll be re-triggered on next turn completion. Reset to listening.
            self._state = DeciderState.LISTENING

    async def close(self) -> None:
        """Cleanup on session end."""
        self._closed = True
        self._cancel_silence_timer()
        self._cancel_llm_task()

    # ── State snapshot (debugging) ──────────────────────────────────────

    def snapshot(self) -> dict:
        """Return current state for logging/debugging."""
        return {
            "state": self._state.value,
            "buffer_size": len(self._turn_buffer),
            "sam_speaking": self._sam_speaking,
            "last_activity_age_ms": (
                int((time.time() - self._last_activity_at) * 1000)
                if self._last_activity_at else 0
            ),
        }

    # ── Internal: timer management ──────────────────────────────────────

    def _mark_activity_and_restart_timer(self) -> None:
        """Record speech activity NOW and reset the silence timer."""
        self._last_activity_at = time.time()

        if self._state == DeciderState.LISTENING:
            self._state = DeciderState.SILENCE_WAITING

        # Cancel existing silence timer
        self._cancel_silence_timer()

        # If a decision is already in flight, cancel it (new speech = new info)
        if self._state == DeciderState.LLM_DECIDING:
            self._cancel_llm_task()
            self._state = DeciderState.SILENCE_WAITING

        # Start fresh 1s silence timer
        self._silence_timer_task = asyncio.create_task(
            self._silence_timer_coroutine()
        )

    def _cancel_silence_timer(self) -> None:
        if self._silence_timer_task and not self._silence_timer_task.done():
            self._silence_timer_task.cancel()
        self._silence_timer_task = None

    def _cancel_llm_task(self) -> None:
        if self._llm_task and not self._llm_task.done():
            self._llm_task.cancel()
        self._llm_task = None

    async def _silence_timer_coroutine(self) -> None:
        """Wait SILENCE_WAIT_SECONDS; if no new activity, fire decision.

        Also enforces MAX_SILENCE_WAIT_SECONDS from the FIRST turn in the window
        to prevent getting stuck when speakers rapid-fire without gaps.
        """
        try:
            # Wait for the silence window
            await asyncio.sleep(SILENCE_WAIT_SECONDS)

            # Double-check state hasn't changed
            if self._closed or self._sam_speaking:
                return
            if not self._turn_buffer:
                # No turns to decide on — back to listening
                self._state = DeciderState.LISTENING
                return

            # Check max-wait safety net
            if self._first_activity_at > 0:
                window_age = time.time() - self._first_activity_at
                if window_age > MAX_SILENCE_WAIT_SECONDS:
                    print(f"[Addressee] {self._tag} ⚠️  Max wait exceeded "
                          f"({window_age:.1f}s) — forcing decision")

            # Fire the LLM decision
            await self._fire_decision()

        except asyncio.CancelledError:
            # Timer was cancelled due to new activity — do nothing
            pass
        except Exception as e:
            print(f"[Addressee] {self._tag} ⚠️  Silence timer error: {e}")

    # ── Internal: decision making ───────────────────────────────────────

    async def _fire_decision(self) -> None:
        """Snapshot buffer, call LLM, emit decision."""
        if not self._turn_buffer or self._closed:
            return

        self._state = DeciderState.LLM_DECIDING
        turns = list(self._turn_buffer)  # snapshot
        self._turn_buffer.clear()
        self._first_activity_at = 0.0

        # Get context from BotSession
        try:
            context = self._get_context()
        except Exception as e:
            print(f"[Addressee] {self._tag} ⚠️  get_context failed: {e}")
            context = ContextBundle(
                participants=[],
                sam_last_response="",
                conversation_history=[],
            )

        # ── Patch A1: FAST-PATH — deterministic decision for clear cases ──
        # Avoids the 1500-2000ms LLM call on unambiguous utterances.
        fast_decision = self._try_fast_path(turns)
        if fast_decision is not None:
            print(f"[Addressee] {self._tag} ⚡ Fast-path: "
                  f"{fast_decision.reasoning} (skipped LLM)")
            try:
                await self._on_decision(fast_decision)
            except Exception as e:
                print(f"[Addressee] {self._tag} ⚠️  on_decision callback "
                      f"failed: {e}")
            self._state = DeciderState.LISTENING
            return
        # ── End fast-path ──

        # Launch LLM call as a cancellable task
        self._llm_task = asyncio.create_task(
            self._run_llm_decision(turns, context)
        )

        try:
            decision = await self._llm_task
        except asyncio.CancelledError:
            # New speech arrived during LLM call — abandon this decision
            print(f"[Addressee] {self._tag} 🚫 Decision cancelled "
                  f"(new speech arrived)")
            # Put turns back at the front of the buffer so we don't lose them
            self._turn_buffer = turns + self._turn_buffer
            return
        except Exception as e:
            print(f"[Addressee] {self._tag} ⚠️  LLM decision failed: {e}")
            self._state = DeciderState.LISTENING
            return
        finally:
            self._llm_task = None

        if decision is None:
            self._state = DeciderState.LISTENING
            return

        # Emit decision to BotSession
        try:
            await self._on_decision(decision)
        except Exception as e:
            print(f"[Addressee] {self._tag} ⚠️  on_decision callback failed: {e}")

        self._state = DeciderState.LISTENING

    def _try_fast_path(self, turns) -> Optional["Decision"]:
        """Patch A1: deterministic decision for unambiguous cases.

        Returns Decision when we are confident, else None (fall through to LLM).

        Rules (in order):
        1. Utterance contains "Sam" as standalone word → respond_to last speaker
        2. Combined text is <= 2 words AND no "Sam" → stay_silent (ack/noise)
        3. Otherwise → None (let LLM decide)

        Note: does NOT match "Samantha", "Samir", "Sammy" — uses word boundary.
        """
        if not turns:
            return None

        combined = " ".join((t.text or "").strip() for t in turns).strip()
        if not combined:
            return None

        import re as _re
        # Word-boundary match for "Sam" (case-insensitive). Followed by
        # non-letter ensures we do not match Samir, Samantha, Sammy, etc.
        sam_pattern = _re.compile(r"\b[Ss]am\b", _re.IGNORECASE)
        has_sam = bool(sam_pattern.search(combined))

        word_count = len(combined.split())

        # Rule 1: explicit "Sam" mention → respond_to
        if has_sam:
            # Pick the speaker who completed the turn (last in buffer —
            # that is the person who just finished talking to Sam).
            speaker = turns[-1].speaker if turns[-1].speaker else ""
            if not speaker:
                return None  # no speaker info, let LLM handle
            return Decision(
                type="respond_to",
                speakers=[speaker],
                text=combined,
                reasoning="fast_path: utterance contains 'Sam'",
            )

        # Rule 2: very short with no address → probably an ack / noise
        if word_count <= 2:
            return Decision(
                type="none",
                speakers=[],
                text=combined,
                reasoning=f"fast_path: {word_count} word(s), no address",
            )

        # Rule 3: ambiguous — fall through to LLM
        return None

    def _get_client_for_key(self, key: str) -> AsyncOpenAI:
        """Lazy cache of per-key AsyncOpenAI clients.

        Creating a new client per call would thrash the httpx connection
        pool (50-100ms cold start). Caching one client per key keeps
        connection reuse tight.
        """
        client = self._client_cache.get(key)
        if client is None:
            client = AsyncOpenAI(
                api_key=key,
                base_url="https://api.groq.com/openai/v1",
            )
            self._client_cache[key] = client
        return client

    async def _run_llm_decision(
        self,
        turns: list[CompletedTurn],
        context: ContextBundle,
    ) -> Optional[Decision]:
        """Groq call with rotator (12-key load spreading + 429 retry) or
        single-key fallback. Returns Decision or None on failure."""
        prompt = self._build_prompt(turns, context)

        # ── Multi-key path (shared rotator) ──
        if self._rotator is not None:
            max_attempts = max(1, self._rotator.get_key_count())
            for attempt in range(max_attempts):
                key_info = await self._rotator.get_next_key()
                if key_info is None:
                    print(f"[Addressee] {self._tag} ⚠️  All Groq keys "
                          f"cooling down — defaulting to 'none'")
                    return Decision(
                        type="none", speakers=[], text="",
                        reasoning="all_keys_rate_limited",
                    )
                key, label = key_info
                client = self._get_client_for_key(key)
                try:
                    resp = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=self._model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0,
                            max_tokens=150,
                            response_format={"type": "json_object"},
                        ),
                        timeout=LLM_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    print(f"[Addressee] {self._tag} ⏱️  LLM timeout "
                          f"on {label} — defaulting to 'none'")
                    await self._rotator.mark_error(key)
                    return Decision(
                        type="none", speakers=[], text="",
                        reasoning="llm_timeout",
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    err_str = str(e).lower()
                    if "429" in err_str or "rate" in err_str or "quota" in err_str:
                        # Rate-limited on this key — try next
                        await self._rotator.mark_rate_limited(key)
                        continue
                    print(f"[Addressee] {self._tag} ⚠️  LLM error "
                          f"on {label}: {e}")
                    await self._rotator.mark_error(key)
                    return None
                # Success — mark key good, break out to parse response
                await self._rotator.mark_success(key)
                break
            else:
                # Exhausted all keys with 429s
                print(f"[Addressee] {self._tag} ⚠️  Exhausted "
                      f"{max_attempts} key(s) with rate limits")
                return Decision(
                    type="none", speakers=[], text="",
                    reasoning="all_keys_rate_limited",
                )

        # ── Single-key path (legacy fallback) ──
        else:
            try:
                resp = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=self._model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=150,
                        response_format={"type": "json_object"},
                    ),
                    timeout=LLM_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                print(f"[Addressee] {self._tag} ⏱️  LLM timeout — "
                      f"defaulting to 'none'")
                return Decision(
                    type="none",
                    speakers=[],
                    text="",
                    reasoning="llm_timeout",
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[Addressee] {self._tag} ⚠️  LLM call error: {e}")
                return None

        # Parse JSON response
        raw = (resp.choices[0].message.content or "").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[Addressee] {self._tag} ⚠️  LLM returned invalid JSON: "
                  f"{raw[:100]}")
            # Fallback: default to first speaker
            return Decision(
                type="respond_to",
                speakers=[turns[0].speaker],
                text=turns[0].text,
                reasoning="json_parse_failed",
            )

        decision_type = data.get("decision", "none")
        speakers = data.get("speakers", [])
        reasoning = data.get("reasoning", "")

        # Build the text to respond to based on decision type
        if decision_type == "respond_to" and speakers:
            # Find this speaker's turn(s) in the buffer
            target = speakers[0]
            matching = [t.text for t in turns if t.speaker == target]
            text = " ".join(matching) if matching else turns[-1].text

        elif decision_type == "respond_both" and len(speakers) >= 2:
            # Format: "Speaker1: ...; Speaker2: ..."
            parts = []
            for s in speakers:
                matching = [t.text for t in turns if t.speaker == s]
                if matching:
                    parts.append(f"{s}: {' '.join(matching)}")
            text = "; ".join(parts)

        else:
            # "none" — nobody addressed Sam
            text = ""

        decision = Decision(
            type=decision_type,
            speakers=speakers,
            text=text,
            reasoning=reasoning,
        )

        # Log the decision
        if decision.type == "none":
            print(f"[Addressee] {self._tag} 🤐 LLM: stay silent "
                  f"({decision.reasoning[:60]})")
        elif decision.type == "respond_both":
            print(f"[Addressee] {self._tag} 👥 LLM: respond to both "
                  f"{decision.speakers} ({decision.reasoning[:60]})")
        else:
            print(f"[Addressee] {self._tag} 🎯 LLM: respond to "
                  f"{decision.speakers} ({decision.reasoning[:60]})")

        return decision

    def _build_prompt(
        self,
        turns: list[CompletedTurn],
        context: ContextBundle,
    ) -> str:
        """Assemble the Groq prompt from turns + context."""
        participants_str = ", ".join(context.participants) if context.participants \
            else "(unknown)"

        sam_last = context.sam_last_response.strip() or "(none yet)"
        if len(sam_last) > 300:
            sam_last = sam_last[:300] + "..."

        if context.conversation_history:
            history_str = "\n".join(context.conversation_history[-HISTORY_TURNS:])
        else:
            history_str = "(no prior exchanges)"

        utterances_str = "\n".join(
            f"{t.speaker}: {t.text}" for t in turns
        )

        return ADDRESSEE_PROMPT.format(
            participants=participants_str,
            sam_last_response=sam_last,
            history_limit=HISTORY_TURNS,
            conversation_history=history_str,
            recent_utterances=utterances_str,
        )