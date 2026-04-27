"""
Trigger.py — Should Sam respond?

Fast-path patterns cover ~80% of cases (0-3ms):
  - Direct address ("sam", "tell me", "can you") → YES
  - Trivial fillers → NO
  - Incomplete endings → wait
  - Questions ending "?" → YES
  - 2+ PM keywords → YES
  - Recent follow-up (<3s) → YES

LLM fallback for ambiguous cases — capped at 500ms, defaults YES on timeout.
"""

import time
import os
import asyncio
from openai import AsyncOpenAI

# Fix B: rotating Groq client (shared 12-key pool with Agent + NLU)
from groq_client import GroqRotatingClient

COOLDOWN_SECONDS = 1.5

TRIGGER_PROMPT = """You are Sam, a senior PM in a live meeting.
Decide whether YOU should speak next.

Context:
{context}

Memory:
{memory}

Latest from {speaker}: "{text}"

YES if: directed at you, question expecting your input, follow-up, greeting.
NO if: directed at someone else, two others talking, filler/acknowledgment, mid-sentence.

Reply ONLY: YES or NO."""

PM_KEYWORDS = [
    "deadline", "deliver", "blocker", "issue", "plan", "decide",
    "approved", "timeline", "task", "owner", "risk", "budget",
    "scope", "stakeholder", "milestone", "sprint", "feature",
    "requirement", "sign-off", "contract", "report", "project",
    "team", "priority", "update", "review", "status", "delay",
    "launch", "release", "client", "dependency", "estimate",
]

FILLERS = {
    "okay", "ok", "sure", "thanks", "thank you", "yep", "nope",
    "alright", "hmm", "uh huh", "got it", "bye", "yeah", "yes",
    "no", "cool", "nice", "great", "perfect", "sounds good",
    "i see", "right", "okay okay", "ok ok",
    "interesting", "noted", "understood", "makes sense",
    "fair enough", "true", "exactly", "absolutely", "definitely",
    "of course", "certainly", "indeed", "wow", "oh", "ah",
    "mhm", "mm", "uh", "um", "hm", "okay sam", "ok sam",
    "stop", "stop sam", "wait", "hold on", "one sec",
}

INCOMPLETE_ENDINGS = {"and", "so", "then", "but", "the", "a", "an", "or", "if", "when"}

DIRECT_ADDRESS = [
    "sam", "about you", "about yourself", "tell me",
    "what do you", "can you", "could you", "would you",
    "your opinion", "your thoughts", "what's your",
    "introduce yourself", "who are you",
]

RECALL_KEYWORDS = [
    "before", "earlier", "told you", "mentioned",
    "remember", "what did i say", "recall", "last time",
    "previously", "you said",
]


class TriggerDetector:
    def __init__(self):
        self._last_response_at: float = 0.0
        # Fix B: shared rotating Groq client (12-key pool)
        self._client = GroqRotatingClient(tag="[Trigger]")

    async def should_respond(self, text, speaker="Unknown", context="", memory=None) -> bool:
        now   = time.monotonic()
        lower = text.lower().strip()

        # Direct address — instant YES
        if any(p in lower for p in DIRECT_ADDRESS):
            self._last_response_at = 0
            print("  ⚡ Direct address — YES")
            return True

        # Fillers — instant NO
        if lower in FILLERS:
            return False

        # Incomplete — wait
        last_word = lower.split()[-1] if lower.split() else ""
        if last_word in INCOMPLETE_ENDINGS:
            print("  ⏸ Incomplete — waiting")
            return False

        # Recall — YES
        if any(k in lower for k in RECALL_KEYWORDS):
            print("  🧠 Recall — YES")
            return True

        # Question — YES
        if lower.endswith("?"):
            print("  ❓ Question — YES")
            return True

        # 2+ PM keywords — YES
        hits = {k for k in PM_KEYWORDS if k in lower}
        if len(hits) >= 2:
            print(f"  🏷️  PM keywords ({hits}) — YES")
            return True

        # Follow-up boost — if Sam responded recently, likely still in conversation
        if now - self._last_response_at < 15:
            print("  🔁 Follow-up (within 15s) — YES")
            return True

        # Cooldown
        if now - self._last_response_at < COOLDOWN_SECONDS:
            return False

        # LLM fallback — 500ms cap
        mem_hint = "\n".join(memory[-5:]) if memory else "None"
        return await self._llm_decide(text, speaker, context, mem_hint)

    async def _llm_decide(self, text, speaker, context, memory) -> bool:
        try:
            resp = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{
                        "role": "user",
                        "content": TRIGGER_PROMPT.format(
                            context=context or "No context",
                            speaker=speaker, text=text, memory=memory,
                        )
                    }],
                    temperature=0,
                    max_tokens=3,
                ),
                timeout=0.5,
            )
            decision = resp.choices[0].message.content.strip().upper()
            result = "YES" in decision
            print(f"  🤖 LLM trigger: {'YES' if result else 'NO'}")
            return result
        except asyncio.TimeoutError:
            print("  ⏱️  Trigger timeout — defaulting YES")
            return True
        except Exception as e:
            print(f"[Trigger] Error: {e}")
            return text.strip().endswith("?")

    def mark_responded(self):
        self._last_response_at = time.monotonic()
