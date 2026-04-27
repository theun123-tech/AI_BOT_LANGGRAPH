"""
standup.py — Developer Standup State Machine (Production)

Architecture:
  CONVERSATION (Groq — fast, user-facing):
    Q&A:        classify + ack (parallel, ~200ms)
    Summary:    from raw answers (no extraction, fast)
    Corrections: update raw → re-summarize (fast)
    Confirm → bot leaves in 2 seconds

  BACKGROUND (Azure — reliable, after bot leaves):
    Extract structured data from confirmed raw answers
    Comment on Jira tickets
    Transition ticket statuses
    Save enriched standup data

User never waits for Azure. Entire conversation is Groq-speed.
"""

import asyncio
import time
import re
import json
import os
from enum import Enum, auto


class StandupState(Enum):
    GREETING      = auto()
    ASK_YESTERDAY = auto()
    ASK_TODAY     = auto()
    ASK_BLOCKERS  = auto()
    SUMMARY       = auto()
    CONFIRM       = auto()
    DONE          = auto()


_JIRA_ID_PATTERN = re.compile(r'\b([A-Z][A-Z0-9]+-\d+)\b')


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT 1: Q&A PHASE (replaces META + CLASSIFY + ACK — one LLM call)
# ══════════════════════════════════════════════════════════════════════════════

QA_PROMPT = """You are Sam, AI PM. Developer {developer} is doing a standup.

Question asked: {topic}
They said: "{text}"

Respond in this format: KEYWORD | one sentence acknowledgment

Keywords:
ANSWER = real work/task/ticket content for the question asked
FILLER = meaningless noise, gibberish, unintelligible
COPIES_PREVIOUS = "same as yesterday", "nothing changed", "ditto"
EMPTY = "no blockers", "none" (blockers question only)
OUT_OF_CONTEXT = NOT a work answer. Includes:
  - Resistance/pushback: "why should I tell you", "that's none of your business", "I don't want to answer", "who are you to ask"
  - Personal/chit-chat: "how are you Sam", "what's up", "are you a robot", "how's your day"
  - Trivia/general knowledge: "what's the weather", "tell me a joke", "who is the president", "what's 2+2"
  - Unrelated work questions: "what's my priority", "show me tickets", "what did John work on"
  - Meta questions about the bot: "what can you do", "how do you work"
REDO = explicit request to restart: "let me start over", "redo this standup"
STOP = explicit cancel: "stop the standup", "cancel this", "I don't want to do standup"
UNCLEAR = input does NOT clearly match any category above. Transcription may be garbled,
          input may be partial/ambiguous, or intent is genuinely unclear. When in doubt,
          use UNCLEAR — it's always safer to ask than to guess.

Tone guidelines:
- Be warm, encouraging, like a supportive teammate — not a robot
- VARY your phrasing — NEVER start every ack with "Got it"
- For yesterday's work: celebrate or appreciate completed work
- For today's plan: be supportive and encouraging about the plan
- For blockers: be empathetic, acknowledge the difficulty
- Keep acks SHORT — under 12 words, no questions (the next question is added automatically)

Examples (notice the variety in starters and warmth):
ANSWER | Nice work wrapping up the login fix yesterday.
ANSWER | Awesome, great to hear the payment module is done.
ANSWER | Solid progress on the video upload issue.
ANSWER | Sounds like a focused day on SCRUM-32.
ANSWER | Perfect, that's a good one to tackle today.
ANSWER | Love that you're picking up the API work.
ANSWER | Oof, sorry to hear about that blocker — hope it clears up.
ANSWER | Thanks for flagging that, we'll keep an eye on it.
FILLER |
EMPTY | Perfect, smooth sailing then.
OUT_OF_CONTEXT |
REDO |
STOP |
UNCLEAR |"""

# ══════════════════════════════════════════════════════════════════════════════
# PROMPT 2: SUMMARY + CONFIRM PHASE (replaces SUMMARY + CONFIRM prompts)
# ══════════════════════════════════════════════════════════════════════════════

PHASE2_PROMPT = """You are Sam, an AI PM wrapping up {developer}'s standup.

{instructions}"""

# Mode A: Generate summary
PHASE2_SUMMARIZE = """Create a BRIEF spoken summary from the developer's answers.

What they said:
  Yesterday: "{yesterday}"
  Today: "{today}"
  Blockers: "{blockers}"

Fill in this EXACT template:
"Yesterday: [phrase]. Today: [phrase]. Blockers: [phrase]. Sound right, or anything to change?"

RULES:
- Use the EXACT template with all three sections.
- Each phrase must be under 10 words.
- Fix speech-to-text errors. Convert garbled words to proper ticket IDs.
- If the raw text is unclear or garbled, preserve the developer's original words as closely as possible. NEVER invent or hallucinate content that wasn't said.
- For blockers: if "no blockers" or "none" → "Blockers: None."
- For blockers: if ANY issue, delay, problem described → MUST include it. NEVER write "None" when a blocker was described.
- Total MUST be under 30 words before the confirmation question."""

# Mode B: Handle confirmation/correction
PHASE2_CONFIRM = """Current standup: Yesterday: "{yesterday}" | Today: "{today}" | Blockers: "{blockers}"
Developer said: "{response}"

Output ONLY one keyword from below. No explanation. No preamble.

CONFIRMED — agrees/done: "yes", "sounds good", "correct", "nothing", "no changes", "done", "bye", "ok", "that's it", "save it", "looks good"
CORRECTION_YESTERDAY_REPLACE — new content for yesterday ("actually yesterday I did X")
CORRECTION_YESTERDAY_ADD — adds to yesterday ("also worked on X")
CORRECTION_TODAY_REPLACE — new content for today
CORRECTION_TODAY_ADD — adds to today
CORRECTION_BLOCKERS_REPLACE — new blocker content
CORRECTION_BLOCKERS_ADD — adds a blocker
COPIES_PREVIOUS_YESTERDAY — same as last standup for yesterday
COPIES_PREVIOUS_TODAY — same as last standup for today
COPIES_PREVIOUS_BLOCKERS — same as last standup for blockers
GUIDE_CHANGE — wants to change but gives NO content ("I need to change", "that's wrong")
OUT_OF_CONTEXT — unrelated question ("what's my priority?", "check this ticket")
REPEAT — repeat summary ("say that again", "repeat", "can you say again")
REDO — restart
UNCLEAR — input does NOT clearly match ANY of the above. Includes:
  - Garbled transcription: "Savior Sam", "Start Sam", random word jumbles
  - Very short ambiguous utterances: "hmm", "uh", "yeah no"
  - Mixed/contradictory signals: "yes no wait"
  - Anything you're not confident about — prefer UNCLEAR over guessing

IMPORTANT: When in doubt, output UNCLEAR. Never guess on destructive actions.
CORRECTION needs ACTUAL work content. No content = GUIDE_CHANGE.
"Today I'm working on X" with actual task = CORRECTION_TODAY_REPLACE, NOT COPIES_PREVIOUS."""

# ══════════════════════════════════════════════════════════════════════════════
# AZURE PROMPT (background extraction after bot leaves)
# ══════════════════════════════════════════════════════════════════════════════

FULL_EXTRACT_PROMPT = """You are an AI PM assistant. Extract structured standup data from a developer's confirmed answers.

Developer: {developer}
Jira project key: {project_key}
Date: {date}

AVAILABLE JIRA TICKETS:
{available_tickets}

CONFIRMED STANDUP ANSWERS (from speech-to-text — may contain errors):
  Yesterday: "{yesterday}"
  Today: "{today}"
  Blockers: "{blockers}"

STEP-BY-STEP EXTRACTION PROCESS:

STEP 1 — SPEECH-TO-TEXT CLEANUP:
The input comes from a live voice conversation transcribed by speech-to-text. It WILL contain:
- Numbers as words: "five" = 5, "twenty three" = 23, "fourteen" = 14
- Project key spoken as a word: if the project key is "{project_key}", the developer may say it as a regular word followed by a number. For example, if the key is "SCRUM", then "scrum five" = {project_key}-5
- Informal ticket references: "ticket five", "number twenty three", "issue fourteen" all refer to {project_key}-N
- Garbled words: speech-to-text may mishear words — focus on INTENT, not exact spelling
First, mentally clean up each answer and identify what tickets the developer is referring to.

STEP 2 — TICKET ID RESOLUTION:
For each ticket reference found in Step 1:
- Convert spoken numbers to digits (e.g., "five" → 5, "twenty three" → 23)
- Prepend the project key: number N → {project_key}-N
- VERIFY the resulting ID exists in the AVAILABLE JIRA TICKETS list above
- If a ticket ID does NOT exist in the available list, DO NOT include it — the developer may have misspoken
- If the developer describes work that matches an available ticket's summary (even without mentioning a number), include that ticket

STEP 3 — STATUS DETECTION:
Analyze the LANGUAGE the developer used to determine what happened to each ticket:
- Yesterday answers with completion language (resolved, completed, finished, done, fixed, wrapped up, closed) → action: "done"
- Today answers with planning language (will work on, starting, continuing, picking up, moving to) → action: "in_progress"
- Blocker answers mentioning a specific ticket being stuck or blocked → action: "blocked"
- If a ticket appears in yesterday with completion language AND in today with continuation language, prioritize the TODAY action (in_progress)
- If no clear status language, do NOT include a status_update for that ticket

STEP 4 — OUTPUT:
Create the JSON with clean summaries, matched ticket IDs, and detected status updates.
Every ticket mentioned by the developer MUST appear in the output if it exists in the available list.

CRITICAL RULES:
- EVERY spoken number near a project reference is a ticket ID — do not ignore them
- The project key "{project_key}" spoken as a word IS a ticket reference, not a methodology reference
- If in doubt whether something is a ticket reference, CHECK the available tickets list — if {project_key}-N exists there, include it
- Do NOT invent ticket IDs that are not in the available list
- Do NOT skip tickets just because the reference was informal or spoken as words

Return ONLY valid JSON (no explanation, no markdown):
{{
  "yesterday": {{
    "summary": "clean summary of yesterday's work",
    "tasks": ["task 1", "task 2"],
    "jira_ids": ["{project_key}-N"],
    "status_updates": [{{"ticket": "{project_key}-N", "action": "done"}}]
  }},
  "today": {{
    "summary": "clean summary of today's plan",
    "tasks": ["task 1"],
    "jira_ids": ["{project_key}-N"],
    "status_updates": [{{"ticket": "{project_key}-N", "action": "in_progress"}}]
  }},
  "blockers": {{
    "summary": "No blockers" or "description of blockers",
    "items": [],
    "jira_ids": []
  }}
}}"""


# ══════════════════════════════════════════════════════════════════════════════
# PM DASHBOARD ONE-LINER (fast scannable summary shown on collapsed card)
# ══════════════════════════════════════════════════════════════════════════════

PM_ONE_LINER_PROMPT = """You are a concise summary writer for a PM dashboard.

Developer: {developer}
Yesterday's work: {yesterday}
Today's plan: {today}
Blockers: {blockers}
Tickets mentioned: {jira_ids}

Write a ONE-LINE summary (under 18 words) that a PM can scan at a glance.

Format rules:
- Lead with yesterday's completion/work, then today's focus
- If there's a real blocker, mention it at the end with clear signal ("blocked on X")
- If no blockers, do NOT mention blockers at all
- Use past tense for yesterday, present continuous for today
- NO filler words: no "the developer", "they said", "yesterday the developer"
- Use actual ticket IDs mentioned (e.g. "SCRUM-32"), never invent new ones
- Keep neutral/professional tone — no "crushed it", "nailed it", etc.

Examples of good one-liners:
- "Finished SCRUM-20 login fix, starting SCRUM-32 payment flow today."
- "Completed API refactor, writing documentation — blocked on manager approval."
- "Continuing SCRUM-15 video upload bug investigation, no blockers."
- "Tested lead flow yesterday, picking up SCRUM-32 today."
- "On leave yesterday, resuming dashboard work today."

Output ONLY the one-liner. No explanation, no preamble, no quotes."""


# ══════════════════════════════════════════════════════════════════════════════
# BLOCKER CLASSIFIER (LLM-based — replaces brittle keyword matching)
# ══════════════════════════════════════════════════════════════════════════════

BLOCKER_CLASSIFY_PROMPT = """Classify whether this standup answer describes a REAL work blocker.

A REAL blocker is anything slowing or preventing work: waiting on someone, stuck on a problem, team member unavailable, external dependencies, technical issues, approval delays.

NOT a real blocker: phrasings meaning "no blockers", "all clear", "nothing", "none", "everything fine", "smooth sailing".

Examples of REAL blockers:
- "Manager on leave today, delays expected"
- "API keeps timing out"
- "Waiting on design review"
- "Feeling stuck on the auth flow"
- "Dependency package not updated yet"

Examples of NOT real blockers:
- "No blockers"
- "None"
- "All clear"
- "Nothing to report"
- "Smooth sailing"
- "Everything is fine"
- "N/A"

Blocker text: "{text}"

Respond with ONLY one word: YES or NO"""


class StandupFlow:
    """Production standup: Groq-speed conversation, Azure background extraction."""

    def __init__(self, developer_name: str, agent, speaker_fn,
                 jira_client=None, jira_context: str = "", azure_extractor=None):
        self.developer = developer_name
        self.agent = agent
        self.speak = speaker_fn
        self.jira = jira_client
        self.azure = azure_extractor
        self._jira_context = jira_context or "(no tickets loaded)"

        self._project_key = "SCRUM"
        if self.jira and hasattr(self.jira, 'project') and self.jira.project:
            self._project_key = self.jira.project

        self.state = StandupState.GREETING
        self.data = {
            "developer": developer_name,
            "date": time.strftime("%Y-%m-%d", time.gmtime()),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "yesterday": {"summary": "", "tasks": [], "jira_ids": [], "raw": ""},
            "today":     {"summary": "", "tasks": [], "jira_ids": [], "raw": ""},
            "blockers":  {"summary": "", "items": [], "jira_ids": [], "raw": ""},
            "completed": False,
            "one_line_summary": "",  # PM dashboard one-liner, generated in background_finalize
            "has_real_blocker": False,  # LLM classification in background_finalize
        }

        self._silence_task = None
        self._generation = 0
        self._all_jira_ids = set()
        self._all_status_updates = []
        self._processing = False
        self._history = []
        self._confirmed_summary = ""  # Clean summary from Groq (used for extraction)

        # Load previous standup for "same as yesterday" feature
        self._previous_standup = None
        try:
            import storage as session_store
            prev = session_store.get_previous_standup(developer_name)
            if prev:
                self._previous_standup = prev
                print(f"[Standup] 📋 Loaded previous standup for {developer_name} ({prev.get('date', '?')})")
            else:
                print(f"[Standup] 📋 No previous standup found for {developer_name}")
        except Exception as e:
            print(f"[Standup] ⚠️  Failed to load previous standup: {e}")
        self._check_buffer_fn = None  # Callback to check if user started speaking (set by websocket_server)

        # Speculative EagerEndOfTurn cache — pre-computed Groq result
        self._cached_qa_result = None   # Raw Groq output string
        self._cached_qa_text = ""       # Transcript used for the cached result

        # Track when re-prompt is playing — allows fast interrupt (user finally speaking)
        self._playing_reprompt = False

        # Unclear input counter — resets on state change or valid classification.
        # After 2 UNCLEARs in the same state, fallback action kicks in (save what we have).
        self._unclear_count = 0
        self._unclear_state = None  # state where unclears accumulated (for reset detection)

    @property
    def is_done(self) -> bool:
        return self.state == StandupState.DONE

    def _add_history(self, speaker: str, text: str):
        self._history.append(f"{speaker}: {text}")
        if len(self._history) > 20:
            self._history = self._history[-20:]

    def _get_context(self) -> str:
        return "\n".join(self._history) if self._history else "(standup just started)"

    # ── Unclear input tracking ────────────────────────────────────────────────

    def _reset_unclear(self):
        """Reset unclear counter — called on valid classification or state change."""
        if self._unclear_count > 0:
            print(f"[Standup] 🔄 Unclear counter reset (was {self._unclear_count})")
        self._unclear_count = 0
        self._unclear_state = None

    def _track_unclear(self) -> int:
        """Increment unclear counter, reset if state changed. Returns new count."""
        if self._unclear_state != self.state:
            self._unclear_count = 0
            self._unclear_state = self.state
        self._unclear_count += 1
        return self._unclear_count

    # ── Groq (fast, user-facing) ──────────────────────────────────────────────
    # Standup uses llama-3.3-70b-versatile (not the agent's default 8b) — the larger
    # model is significantly more accurate at classification (CONFIRM intent, OUT_OF_CONTEXT
    # detection, CORRECTION detection). Temperature 0 for deterministic classification.
    # Latency is similar to 8b on Groq (500-1000ms) so no user-perceptible cost.

    STANDUP_MODEL = "llama-3.3-70b-versatile"
    STANDUP_TEMPERATURE = 0.0

    async def _groq(self, system: str, user_msg: str, max_tokens: int = 100) -> str:
        import time as _t
        t0 = _t.time()
        try:
            response = await asyncio.wait_for(
                self.agent.client.chat.completions.create(
                    model=self.STANDUP_MODEL,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
                    temperature=self.STANDUP_TEMPERATURE, max_tokens=max_tokens,
                ),
                timeout=5.0,
            )
            result = response.choices[0].message.content.strip()
            ms = (_t.time() - t0) * 1000
            print(f"[Standup] 🤖 Groq ({ms:.0f}ms): \"{result[:80]}\"")
            self._log_prompt("GROQ", system, user_msg, result, ms)
            return result
        except asyncio.TimeoutError:
            print(f"[Standup] ⏱ Groq timeout (5s) — defaulting")
            self._log_prompt("GROQ", system, user_msg, "TIMEOUT", 5000)
            raise
        except Exception as e:
            ms = (_t.time() - t0) * 1000
            print(f"[Standup] ⚠️  Groq error ({ms:.0f}ms): {e}")
            self._log_prompt("GROQ", system, user_msg, f"ERROR: {e}", ms)
            raise

    def _log_prompt(self, label: str, system: str, user_msg: str, result: str, ms: float):
        """Debug prompt logging — disabled in production."""
        pass

    # ── Speculative EagerEndOfTurn pre-computation ────────────────────────────

    async def pre_classify(self, text: str) -> str:
        """Pre-compute Groq classify+ack during EagerEndOfTurn window.

        Called by websocket_server's EagerEndOfTurn handler before EndOfTurn
        confirms. Returns raw Groq result string for caching.
        Only runs for Q&A states (ASK_YESTERDAY/TODAY/BLOCKERS).
        Returns None for other states or on failure.
        """
        if self.state not in (StandupState.ASK_YESTERDAY, StandupState.ASK_TODAY, StandupState.ASK_BLOCKERS):
            return None
        topic = self._current_question_label()
        try:
            result = await self._groq(
                QA_PROMPT.format(developer=self.developer, context=self._get_context(),
                                topic=topic, text=text),
                text, max_tokens=60)
            return result
        except Exception:
            return None

    def set_cached_result(self, result: str, text: str):
        """Cache a pre-computed Groq result from EagerEndOfTurn."""
        self._cached_qa_result = result
        self._cached_qa_text = text

    def clear_cached_result(self):
        """Clear cached result (called on TurnResumed or state change)."""
        self._cached_qa_result = None
        self._cached_qa_text = ""

    # ── Azure (reliable, background only) ─────────────────────────────────────

    async def _azure(self, system: str, user_msg: str, max_tokens: int = 500) -> str:
        if not self.azure or not self.azure.enabled:
            return await self._groq(system, user_msg, max_tokens)

        import httpx
        url = f"{self.azure.endpoint}/openai/deployments/{self.azure.deployment}/chat/completions?api-version={self.azure.api_version}"
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(url,
                        headers={"api-key": self.azure.api_key, "Content-Type": "application/json"},
                        json={"messages": [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
                              "temperature": 0.2, "max_tokens": max_tokens})
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                if attempt < 2:
                    print(f"[Standup] ⚠️  Azure attempt {attempt+1}/3: {e}")
                    await asyncio.sleep(1)
                else:
                    print(f"[Standup] ❌ Azure failed, falling back to Groq")
                    return await self._groq(system, user_msg, max_tokens)

    def _current_question_label(self) -> str:
        return {
            StandupState.ASK_YESTERDAY: "yesterday's work",
            StandupState.ASK_TODAY: "today's plan",
            StandupState.ASK_BLOCKERS: "blockers",
            StandupState.CONFIRM: "confirming the summary",
        }.get(self.state, "standup")

    # ══════════════════════════════════════════════════════════════════════════
    # USER-FACING FLOW (all Groq, fast)
    # ══════════════════════════════════════════════════════════════════════════

    async def start(self, gen: int):
        self._generation = gen
        greeting = f"Hey {self.developer}, let's do your standup real quick. What did you work on yesterday?"
        self._add_history("Sam", greeting)
        await self.speak(greeting, "standup-greeting", gen)
        self.state = StandupState.ASK_YESTERDAY
        self._start_silence_timer()

    async def handle(self, text: str, speaker: str, gen: int) -> bool:
        self._generation = gen
        self._cancel_silence_timer()
        if speaker.lower() == "sam":
            return not self.is_done
        self._add_history(speaker, text)
        print(f"[Standup] 📥 handle() state={self.state.name} processing={self._processing} text=\"{text[:60]}\"")
        if self._processing:
            print(f"[Standup] ⚠️  handle() SKIPPED — _processing is True")
            return not self.is_done
        self._processing = True
        try:
            if self.state in (StandupState.ASK_YESTERDAY, StandupState.ASK_TODAY, StandupState.ASK_BLOCKERS):
                print(f"[Standup] 📋 → _handle_question ({self.state.name})")
                await self._handle_question(text, gen)
            elif self.state in (StandupState.CONFIRM, StandupState.SUMMARY):
                print(f"[Standup] 📋 → _handle_confirmation ({self.state.name})")
                await self._handle_confirmation(text, gen)
            else:
                print(f"[Standup] ⚠️  handle() — unhandled state: {self.state.name}")
        finally:
            self._processing = False
            print(f"[Standup] 📤 handle() done — _processing released, state={self.state.name}")
        return not self.is_done

    # ── Q&A (classify + ack, all parallel, no extraction) ─────────────────────

    async def _handle_question(self, text: str, gen: int):
        topic = self._current_question_label()
        field = {StandupState.ASK_YESTERDAY: "yesterday",
                 StandupState.ASK_TODAY: "today",
                 StandupState.ASK_BLOCKERS: "blockers"}[self.state]

        # ── Single LLM call: classify + ack (replaces META + CLASSIFY + ACK) ──
        words = text.strip().split()
        text_lower = text.strip().lower()

        # Fast path: skip LLM for obvious cases
        if len(words) > 4:
            classification = "ANSWER"
            ack = None  # will get from LLM below
            print(f"[Standup] 📋 Classify: ANSWER (fast — {len(words)} words)")
        elif field == "blockers" and any(neg in text_lower for neg in ["no block", "no blocker", "none", "nope", "nothing", "all clear", "all good", "no issues"]):
            classification = "EMPTY"
            ack = None
            print(f"[Standup] 📋 Classify: EMPTY (fast — blocker negation)")
        else:
            classification = None  # need LLM

        if classification == "ANSWER" or classification is None:
            # Use cached EagerEndOfTurn result if available, otherwise call Groq
            try:
                if self._cached_qa_result and self._cached_qa_text == text:
                    result = self._cached_qa_result
                    self._cached_qa_result = None
                    self._cached_qa_text = ""
                    print(f"[Standup] ⚡ Using cached EagerEOT result (saved ~200ms)")
                else:
                    self.clear_cached_result()
                    result = await self._groq(
                        QA_PROMPT.format(developer=self.developer, context=self._get_context(),
                                        topic=topic, text=text),
                        text, max_tokens=60)
                # Parse "CLASSIFICATION | ack text"
                if "|" in result:
                    parts = result.split("|", 1)
                    llm_class = parts[0].strip().upper()
                    ack = parts[1].strip() if len(parts) > 1 else ""
                    # Safety: reject ack if Groq copied prompt instructions
                    if ack and any(x in ack.lower() for x in ["classification", "keyword", "sentence, no question", "work/task/ticket"]):
                        ack = "Got it."
                else:
                    llm_class = result.strip().upper()
                    ack = ""
                if classification is None:
                    # Reject if Groq literally copied the word "CLASSIFICATION"
                    if llm_class in ("CLASSIFICATION", "KEYWORD", ""):
                        classification = "ANSWER"
                    else:
                        classification = llm_class
                elif llm_class in ("OUT_OF_CONTEXT", "REDO", "STOP", "FILLER"):
                    # Escape-hatch override: fast-path said ANSWER but LLM caught
                    # resistance/trivia/refusal — trust the LLM's judgment
                    print(f"[Standup] 📋 LLM override: fast-path ANSWER → {llm_class}")
                    classification = llm_class
                elif ack:
                    pass  # keep fast-path classification, use LLM ack
                print(f"[Standup] 📋 Classify: {classification}" + (f" (LLM)" if llm_class else ""))
            except Exception:
                if classification is None:
                    classification = "ANSWER"
                ack = "Got it."

        # Handle REDO/STOP
        if classification == "REDO":
            r = "No problem, let's start over. What did you work on yesterday?"
            self._add_history("Sam", r)
            await self.speak(r, "standup-redo", gen)
            self._reset_data()
            self.state = StandupState.ASK_YESTERDAY
            self._start_silence_timer()
            return

        if classification == "STOP":
            r = "Okay, standup cancelled. Let me know if you want to do it later."
            self._add_history("Sam", r)
            await self.speak(r, "standup-stop", gen)
            self.state = StandupState.DONE
            return

        if classification == "FILLER":
            reprompts = {
                "yesterday": "Sorry, I didn't catch that. What tasks or tickets did you work on yesterday?",
                "today": "Sorry, could you repeat that? What are you planning to work on today?",
                "blockers": "Sorry, I didn't get that. Are there any blockers, or are you all clear?",
            }
            r = reprompts[field]
            self._add_history("Sam", r)
            await self.speak(r, f"standup-clarify-{field}", gen)
            self._start_silence_timer()
            return

        if classification == "OUT_OF_CONTEXT":
            redirect = {
                "yesterday": "Let's stay focused on the standup for now — what did you work on yesterday?",
                "today": "Let's stick with the standup for now — what are you planning for today?",
                "blockers": "Let's wrap up the standup first — any blockers, or are you all clear?",
            }
            r = redirect[field]
            self._add_history("Sam", r)
            await self.speak(r, f"standup-redirect-{field}", gen)
            self._start_silence_timer()
            return

        if classification == "UNCLEAR":
            attempt = self._track_unclear()
            print(f"[Standup] ❓ UNCLEAR ({field}, attempt {attempt}/2)")

            if attempt == 1:
                # 1st clarification — gentle, warm
                clarify = {
                    "yesterday": "Sorry, I didn't quite catch that. Could you tell me what you worked on yesterday?",
                    "today": "Sorry, could you say that again? What are you planning for today?",
                    "blockers": "Sorry, didn't catch that. Any blockers, or are you all clear?",
                }
                r = clarify[field]
                self._add_history("Sam", r)
                await self.speak(r, f"standup-unclear-{field}-1", gen)
                self._start_silence_timer()
                return

            elif attempt == 2:
                # 2nd clarification — more explicit options
                clarify = {
                    "yesterday": "No worries — just briefly, what was yesterday's work? Even a short summary is fine.",
                    "today": "No worries — what's on your plate today? Even a rough idea works.",
                    "blockers": "No worries — just say 'yes, blockers' or 'no blockers'.",
                }
                r = clarify[field]
                self._add_history("Sam", r)
                await self.speak(r, f"standup-unclear-{field}-2", gen)
                self._start_silence_timer()
                return

            else:
                # 3rd UNCLEAR — fallback: save what we have and advance
                print(f"[Standup] ⚠️  UNCLEAR max attempts reached — saving last input and advancing")
                self.data[field]["raw"] = text if text.strip() else "(not provided)"
                self._reset_unclear()
                response = "I'll note that down and move on."
                # Fall through to state advancement below

        else:
            # Any non-UNCLEAR classification resets the counter
            self._reset_unclear()

        # Store raw answer
        if classification == "COPIES_PREVIOUS":
            copied = False
            if field == "yesterday" and self._previous_standup:
                # Copy from previous standup's TODAY (their plan = what they actually did)
                # April 14 standup's "today" = April 14's work = April 15's "yesterday"
                prev_today = self._previous_standup.get("today", {})
                prev_raw = prev_today.get("raw", "") or prev_today.get("summary", "")
                prev_ids = prev_today.get("jira_ids", [])
                if prev_raw:
                    self.data["yesterday"]["raw"] = prev_raw
                    self.data["yesterday"]["jira_ids"] = list(prev_ids)
                    copied = True
                    response = f"Got it, same as last time — {prev_raw[:60]}."
                    print(f"[Standup] 📋 COPIES_PREVIOUS (yesterday): copied from previous standup's TODAY ({self._previous_standup.get('date', '?')})")
                    print(f"[Standup] 📋 Copied: \"{prev_raw[:60]}\" + {len(prev_ids)} ticket IDs")

            elif field == "today" and self.data["yesterday"]["raw"]:
                # Copy from THIS standup's yesterday answer
                self.data["today"]["raw"] = self.data["yesterday"]["raw"]
                self.data["today"]["jira_ids"] = list(self.data["yesterday"].get("jira_ids", []))
                copied = True
                response = f"Got it, continuing with {self.data['yesterday']['raw'][:60]}."
                print(f"[Standup] 📋 COPIES_PREVIOUS (today): copied from current yesterday")
                print(f"[Standup] 📋 Copied: \"{self.data['yesterday']['raw'][:60]}\" + {len(self.data['today']['jira_ids'])} ticket IDs")

            elif field == "blockers" and self._previous_standup:
                # Copy from previous standup's blockers
                prev_blockers = self._previous_standup.get("blockers", {})
                prev_raw = prev_blockers.get("raw", "") or prev_blockers.get("summary", "")
                if prev_raw and prev_raw.lower() not in ("no blockers", "none", ""):
                    self.data["blockers"]["raw"] = prev_raw
                    copied = True
                    response = f"Got it, same blockers — {prev_raw[:60]}."
                    print(f"[Standup] 📋 COPIES_PREVIOUS (blockers): copied from previous standup")
                else:
                    self.data["blockers"]["raw"] = "No blockers"
                    copied = True
                    response = "No blockers last time either. All clear."
                    print(f"[Standup] 📋 COPIES_PREVIOUS (blockers): previous had no blockers")

            if not copied:
                # No previous standup found or no data to copy
                reprompts = {
                    "yesterday": f"I don't have your previous standup on file, {self.developer}. Could you tell me what you worked on?",
                    "today": "I didn't catch yesterday's work yet. What are you planning for today?",
                    "blockers": "I don't have previous blockers on file. Any blockers right now?",
                }
                r = reprompts[field]
                self._add_history("Sam", r)
                await self.speak(r, f"standup-no-previous-{field}", gen)
                self._start_silence_timer()
                print(f"[Standup] 📋 COPIES_PREVIOUS: no previous data for {field} — re-asking")
                return

        elif classification == "EMPTY" and field == "blockers":
            self.data["blockers"]["raw"] = "No blockers"
            response = "All clear, no blockers."
        elif classification == "UNCLEAR":
            # UNCLEAR fallback already set data and response above — don't overwrite
            pass
        else:
            self.data[field]["raw"] = text
            response = ack if ack else "Got it."

        # Advance state
        if self.state == StandupState.ASK_YESTERDAY:
            response += " What's on your plate for today?"
            self._add_history("Sam", response)
            await self.speak(response, "standup-ack-yesterday", gen)
            self.state = StandupState.ASK_TODAY
        elif self.state == StandupState.ASK_TODAY:
            response += " Any blockers?"
            self._add_history("Sam", response)
            await self.speak(response, "standup-ack-today", gen)
            self.state = StandupState.ASK_BLOCKERS
        elif self.state == StandupState.ASK_BLOCKERS:
            self._add_history("Sam", response)
            await self.speak(response, "standup-ack-blockers", gen)
            # Set CONFIRM state BEFORE summary — if user interrupts during summary,
            # their text should be handled as a confirmation/correction
            self.state = StandupState.CONFIRM
            await self._speak_summary(gen)

        self._start_silence_timer()

    # ── Summary (Groq, from raw answers — fast) ──────────────────────────────

    async def _speak_summary(self, gen: int):
        yesterday = self.data["yesterday"]["raw"] or "(no answer)"
        today = self.data["today"]["raw"] or "(no answer)"
        blockers = self.data["blockers"]["raw"] or "No blockers"

        try:
            summary = await self._groq(
                PHASE2_PROMPT.format(
                    developer=self.developer,
                    instructions=PHASE2_SUMMARIZE.format(
                        yesterday=yesterday, today=today, blockers=blockers)),
                "Summarize standup", max_tokens=60)
        except Exception:
            summary = (f"Yesterday: {yesterday}. Today: {today}. {blockers}. "
                       f"Sound right?")

        self._confirmed_summary = summary
        print(f"[Standup] 📋 Clean summary saved for extraction ({len(summary.split())} words)")

        self._add_history("Sam", summary)
        await self.speak(summary, "standup-summary", gen)

    # ── Confirmation ──────────────────────────────────────────────────────────

    async def _handle_confirmation(self, text: str, gen: int):
        yesterday = self.data["yesterday"]["raw"]
        today = self.data["today"]["raw"]
        blockers = self.data["blockers"]["raw"] or "No blockers"

        print(f"[Standup] 🔍 Confirm input: \"{text[:60]}\"")
        print(f"[Standup] 🔍 Current data — Y: \"{yesterday[:40]}\" T: \"{today[:40]}\" B: \"{blockers[:40]}\"")

        try:
            raw_intent = await self._groq(
                PHASE2_PROMPT.format(
                    developer=self.developer,
                    instructions=PHASE2_CONFIRM.format(
                        yesterday=yesterday, today=today, blockers=blockers, response=text)),
                text, max_tokens=80)
            print(f"[Standup] 🔍 Raw LLM output: \"{raw_intent}\"")
            intent = raw_intent.strip().upper().replace(" ", "_")
            # Extract valid classification from verbose responses
            # Check longest keywords first to avoid partial matches
            _VALID_INTENTS = [
                "COPIES_PREVIOUS_YESTERDAY", "COPIES_PREVIOUS_TODAY", "COPIES_PREVIOUS_BLOCKERS",
                "CORRECTION_YESTERDAY_ADD", "CORRECTION_YESTERDAY_REPLACE",
                "CORRECTION_TODAY_ADD", "CORRECTION_TODAY_REPLACE",
                "CORRECTION_BLOCKERS_ADD", "CORRECTION_BLOCKERS_REPLACE",
                "GUIDE_CHANGE", "OUT_OF_CONTEXT", "CONFIRMED", "REDO", "REPEAT",
                "UNCLEAR",
            ]
            matched = None
            for valid in _VALID_INTENTS:
                if valid in intent:
                    matched = valid
                    break
            # Handle truncated responses (max_tokens cut off "CORRECTION_YESTERDAY_REPL...")
            if not matched and "CORRECTION" in intent:
                if "YESTERDAY" in intent:
                    matched = "CORRECTION_YESTERDAY_ADD" if "ADD" in intent else "CORRECTION_YESTERDAY_REPLACE"
                elif "TODAY" in intent:
                    matched = "CORRECTION_TODAY_ADD" if "ADD" in intent else "CORRECTION_TODAY_REPLACE"
                elif "BLOCKER" in intent:
                    matched = "CORRECTION_BLOCKERS_ADD" if "ADD" in intent else "CORRECTION_BLOCKERS_REPLACE"
            if not matched and "COPIES_PREVIOUS" in intent:
                matched = "COPIES_PREVIOUS_YESTERDAY"  # safe default
            # Default to UNCLEAR — safer than guessing CONFIRMED (which would silently save bad data)
            intent = matched or "UNCLEAR"
            print(f"[Standup] 🔍 Confirm intent: {intent}")
        except Exception as e:
            print(f"[Standup] ⚠️  Confirm LLM failed: {e} — defaulting UNCLEAR")
            intent = "UNCLEAR"

        if intent == "UNCLEAR":
            attempt = self._track_unclear()
            print(f"[Standup] ❓ UNCLEAR (CONFIRM, attempt {attempt}/2)")

            if attempt == 1:
                # 1st clarification — natural, warm
                r = "Sorry, I didn't quite catch that. Did you want to save the standup, change something, or hear it again?"
                self._add_history("Sam", r)
                await self.speak(r, "standup-confirm-unclear-1", gen)
                self._start_silence_timer()
                return

            elif attempt == 2:
                # 2nd clarification — explicit options (closed-set easier for STT)
                r = "Let me give you clear options — please say 'save', 'change', or 'cancel'."
                self._add_history("Sam", r)
                await self.speak(r, "standup-confirm-unclear-2", gen)
                self._start_silence_timer()
                return

            else:
                # 3rd UNCLEAR — fallback: save what we have
                print(f"[Standup] ⚠️  UNCLEAR max attempts reached — saving standup as-is")
                self._reset_unclear()
                r = "I'll save what we have. You can always update it in Jira. Have a good day!"
                self._add_history("Sam", r)
                await self.speak(r, "standup-unclear-fallback-save", gen)
                self.data["completed"] = True
                self.data["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                self.state = StandupState.DONE
                print(f"[Standup] ✅ {self.developer}'s standup saved via UNCLEAR fallback")
                return

        # Any non-UNCLEAR intent — reset unclear counter
        self._reset_unclear()

        if intent == "CONFIRMED":
            r = "Great, standup saved. Have a productive day!"
            self._add_history("Sam", r)
            await self.speak(r, "standup-confirmed", gen)
            self.data["completed"] = True
            self.data["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.state = StandupState.DONE
            print(f"[Standup] ✅ {self.developer}'s standup confirmed")

        elif intent == "REDO":
            r = "No problem, let's start over. What did you work on yesterday?"
            self._add_history("Sam", r)
            await self.speak(r, "standup-redo", gen)
            self._reset_data()
            self.state = StandupState.ASK_YESTERDAY
            self._start_silence_timer()

        elif intent == "REPEAT":
            r = "Sure, let me repeat that."
            self._add_history("Sam", r)
            await self.speak(r, "standup-repeat-ack", gen)
            self.state = StandupState.CONFIRM
            await self._speak_summary(gen)
            self._start_silence_timer()

        elif intent == "GUIDE_CHANGE":
            r = "Sure, what would you like to change?"
            self._add_history("Sam", r)
            await self.speak(r, "standup-guide", gen)
            self._start_silence_timer()

        elif intent == "OUT_OF_CONTEXT":
            r = "I can help with that after the standup. Does the summary look right, or anything to change?"
            self._add_history("Sam", r)
            await self.speak(r, "standup-out-of-context", gen)
            self._start_silence_timer()

        elif "COPIES_PREVIOUS" in intent:
            field = {"COPIES_PREVIOUS_YESTERDAY": "yesterday",
                     "COPIES_PREVIOUS_TODAY": "today",
                     "COPIES_PREVIOUS_BLOCKERS": "blockers"}.get(intent)
            if field:
                await self._apply_copies_previous(field, gen)
            else:
                r = "Sure, what would you like to copy from last time?"
                self._add_history("Sam", r)
                await self.speak(r, "standup-unclear-copy", gen)
                self._start_silence_timer()

        elif "CORRECTION_YESTERDAY" in intent:
            is_add = intent.endswith("_ADD")
            await self._apply_correction("yesterday", text, gen, is_add)
        elif "CORRECTION_TODAY" in intent:
            is_add = intent.endswith("_ADD")
            await self._apply_correction("today", text, gen, is_add)
        elif "CORRECTION_BLOCKER" in intent:
            is_add = intent.endswith("_ADD")
            await self._apply_correction("blockers", text, gen, is_add)
        else:
            r = "Sure, go ahead. What would you like to update?"
            self._add_history("Sam", r)
            await self.speak(r, "standup-unclear", gen)
            self._start_silence_timer()

    # ── Copies Previous (reuse data from last standup — during CONFIRM phase) ──

    async def _apply_copies_previous(self, field: str, gen: int):
        field_label = {"yesterday": "yesterday's work", "today": "today's plan", "blockers": "blockers"}[field]
        print(f"[Standup] 📋 COPIES_PREVIOUS ({field}) during CONFIRM phase")

        copied = False

        if field == "today":
            # For today: copy from THIS standup's yesterday
            if self.data["yesterday"]["raw"]:
                self.data["today"]["raw"] = self.data["yesterday"]["raw"]
                self.data["today"]["jira_ids"] = list(self.data["yesterday"].get("jira_ids", []))
                prev_raw = self.data["yesterday"]["raw"]
                copied = True
                print(f"[Standup] 📋 Copied from current yesterday: \"{prev_raw[:60]}\"")
        elif field == "yesterday" and self._previous_standup:
            # For yesterday: copy from previous standup's TODAY
            # April 14's "today" plan = what they did on April 14 = April 15's "yesterday"
            prev_today = self._previous_standup.get("today", {})
            prev_raw = prev_today.get("raw", "") or prev_today.get("summary", "")
            prev_ids = prev_today.get("jira_ids", [])
            if prev_raw:
                self.data["yesterday"]["raw"] = prev_raw
                self.data["yesterday"]["jira_ids"] = list(prev_ids)
                copied = True
                print(f"[Standup] 📋 Copied from previous standup's TODAY ({self._previous_standup.get('date', '?')}): \"{prev_raw[:60]}\" + {len(prev_ids)} IDs")
        elif field == "blockers" and self._previous_standup:
            # For blockers: copy from previous standup's blockers
            prev_blockers = self._previous_standup.get("blockers", {})
            prev_raw = prev_blockers.get("raw", "") or prev_blockers.get("summary", "")
            prev_ids = prev_blockers.get("jira_ids", [])
            if prev_raw and prev_raw.lower() not in ("no blockers", "none", ""):
                self.data["blockers"]["raw"] = prev_raw
                self.data["blockers"]["jira_ids"] = list(prev_ids)
                copied = True
                print(f"[Standup] 📋 Copied from previous standup's BLOCKERS ({self._previous_standup.get('date', '?')}): \"{prev_raw[:60]}\" + {len(prev_ids)} IDs")
            else:
                self.data["blockers"]["raw"] = "No blockers"
                copied = True
                print(f"[Standup] 📋 Previous had no blockers — keeping 'No blockers'")

        if copied:
            r = f"Got it, same as last time for {field_label}."
            self._add_history("Sam", r)
            await self.speak(r, "standup-copies-previous", gen)
            self.state = StandupState.CONFIRM
            await self._speak_summary(gen)
            self._start_silence_timer()
        else:
            r = f"I don't have previous data for {field_label}. What should it be?"
            self._add_history("Sam", r)
            await self.speak(r, "standup-no-previous", gen)
            self._start_silence_timer()
            print(f"[Standup] 📋 No previous data for {field} — re-asking")

    # ── Correction (update raw answer, re-summarize — Groq, fast) ─────────────

    async def _apply_correction(self, field: str, correction_text: str, gen: int, is_additive: bool = False):
        field_label = {"yesterday": "yesterday's work", "today": "today's plan", "blockers": "blockers"}[field]
        print(f"[Standup] ✏️  Correcting {field}: {correction_text[:60]}")
        print(f"[Standup] ✏️  Mode: {'ADD' if is_additive else 'REPLACE'}")

        current_raw = self.data[field]["raw"]
        if is_additive and current_raw:
            self.data[field]["raw"] = f"{current_raw}. {correction_text}"
            print(f"[Standup] ✏️  Before: \"{current_raw[:50]}\"")
            print(f"[Standup] ✏️  After:  \"{self.data[field]['raw'][:80]}\"")
        else:
            print(f"[Standup] ✏️  Replaced: \"{current_raw[:50]}\" → \"{correction_text[:50]}\"")
            self.data[field]["raw"] = correction_text

        r = f"Got it, I've updated {field_label}."
        self._add_history("Sam", r)
        await self.speak(r, "standup-correction-ack", gen)
        self.state = StandupState.CONFIRM
        await self._speak_summary(gen)
        self._start_silence_timer()

    # ══════════════════════════════════════════════════════════════════════════
    # BACKGROUND PROCESSING (Azure, after bot leaves)
    # ══════════════════════════════════════════════════════════════════════════

    async def background_finalize(self):
        """Called AFTER bot leaves. Smart-fetches relevant tickets, extracts data, creates subtasks.
        Creates fresh JiraClient since session's client gets closed during cleanup."""

        print(f"[Standup] 🔧 Background: extracting structured data...")
        t0 = time.time()

        # Prepare variables — use clean Groq summary if available
        if self._confirmed_summary:
            clean = self._confirmed_summary
            print(f"[Standup] 📋 Using clean summary for extraction: {clean[:80]}")
            yesterday_for_extraction = self.data["yesterday"]["raw"]
            today_for_extraction = self.data["today"]["raw"]
            blockers_for_extraction = self.data["blockers"]["raw"] or "No blockers"

            import re as _re_local
            y_match = _re_local.search(r'[Yy]esterday:\s*(.+?)(?:\.\s*[Tt]oday:)', clean)
            t_match = _re_local.search(r'[Tt]oday:\s*(.+?)(?:\.\s*(?:No blocker|Blocker|Sound|Does))', clean)
            # Blockers: capture after "Blockers:" up to "Sound right"/"Does" or end
            b_match = _re_local.search(r'[Bb]lockers?:\s*(.+?)(?:\.\s*(?:Sound|Does)|$)', clean)
            if y_match:
                yesterday_for_extraction = y_match.group(1).strip().rstrip(".")
                print(f"[Standup] 📋 Clean yesterday: {yesterday_for_extraction}")
            if t_match:
                today_for_extraction = t_match.group(1).strip().rstrip(".")
                print(f"[Standup] 📋 Clean today: {today_for_extraction}")
            if b_match:
                blockers_for_extraction = b_match.group(1).strip().rstrip(".")
                print(f"[Standup] 📋 Clean blockers: {blockers_for_extraction}")
        else:
            yesterday_for_extraction = self.data["yesterday"]["raw"]
            today_for_extraction = self.data["today"]["raw"]
            blockers_for_extraction = self.data["blockers"]["raw"] or "No blockers"

        # ── Smart fetch: build targeted ticket context ──
        # Only search with yesterday + today text (blockers describe problems, not tickets)
        search_raw = f"{self.data['yesterday']['raw']} {self.data['today']['raw']}"

        # Step 1: Find explicit ticket IDs in raw text + pre-copied IDs from COPIES_PREVIOUS
        ticket_pattern = re.compile(r'\b' + re.escape(self._project_key) + r'-\d+\b', re.IGNORECASE)
        explicit_ids = list(set(m.upper() for m in ticket_pattern.findall(search_raw)))
        # Also include ticket IDs copied during conversation (from COPIES_PREVIOUS)
        for field in ("yesterday", "today", "blockers"):
            for tid in self.data[field].get("jira_ids", []):
                if tid and tid.upper() not in [x.upper() for x in explicit_ids]:
                    explicit_ids.append(tid.upper())
        print(f"[Standup] 🔍 Smart fetch: explicit IDs found: {explicit_ids}")

        # Step 2: Groq extracts SEPARATE search phrases for each ticket/feature mentioned
        # Only for text WITHOUT explicit ticket IDs (description-based matching)
        text_without_ids = ticket_pattern.sub('', search_raw).strip()
        search_phrases = []
        if text_without_ids and len(text_without_ids.split()) > 2:
            try:
                cleaned = await self._groq(
                    "Extract search phrases from this developer's standup answer. "
                    "Each SEPARATE task, feature, or bug mentioned should be its own search phrase. "
                    "Keep feature names, module names, page names, and technical terms. "
                    "Remove filler words, greetings, and conversational noise. "
                    "Separate each phrase with | delimiter. "
                    "Example: 'CSV export feature | login page crash | dark mode feature'\n"
                    "Return ONLY the search phrases separated by |, nothing else.",
                    text_without_ids, max_tokens=60)
                cleaned = cleaned.strip().strip('"').strip("'")
                if cleaned:
                    for phrase in cleaned.split("|"):
                        phrase = phrase.strip()
                        if phrase and len(phrase.split()) >= 2:
                            search_phrases.append(phrase)
                    if search_phrases:
                        print(f"[Standup] 🔍 Smart fetch: Groq extracted {len(search_phrases)} search phrase(s): {search_phrases}")
            except Exception as e:
                print(f"[Standup] ⚠️  Groq search cleanup failed: {e}")

        # Create a fresh JiraClient for background work
        bg_jira = None
        if self.jira and self.jira.enabled:
            try:
                from external_apis import JiraClient
                bg_jira = JiraClient()
                if not bg_jira.enabled:
                    bg_jira = None
            except Exception as e:
                print(f"[Standup] ⚠️  Background JiraClient failed: {e}")

        smart_context_lines = []
        fetched_ids = set()

        if bg_jira:
            # Fetch explicit ticket IDs
            for tid in explicit_ids:
                try:
                    ticket = await bg_jira.get_ticket(tid)
                    if ticket.get("key") and ticket["key"] != "?":
                        line = f"  {ticket['key']}: {ticket['summary']} [{ticket['status']}] ({ticket['priority']}, {ticket['assignee']})"
                        if ticket.get('description'):
                            line += f" — {ticket['description'][:100]}"
                        smart_context_lines.append(line)
                        fetched_ids.add(ticket['key'].upper())
                except Exception as e:
                    print(f"[Standup] ⚠️  Failed to fetch {tid}: {e}")

            # Search Jira with Groq-cleaned text
            for phrase in search_phrases:
                try:
                    desc_results = await bg_jira.search_text(phrase, max_results=5)
                    for ticket in desc_results:
                        tid = ticket.get("key", "").upper()
                        if tid and tid not in fetched_ids and not ticket.get("summary", "").startswith("Standup —"):
                            line = f"  {ticket['key']}: {ticket['summary']} [{ticket['status']}] ({ticket['priority']}, {ticket['assignee']})"
                            if ticket.get('description'):
                                line += f" — {ticket['description'][:100]}"
                            smart_context_lines.append(line)
                            fetched_ids.add(tid)
                            print(f"[Standup] 🔍 Smart fetch: found {tid} via description match")
                except Exception as e:
                    print(f"[Standup] ⚠️  Jira search failed: {e}")

        smart_context = "JIRA TICKETS:\n" + "\n".join(smart_context_lines) if smart_context_lines else self._jira_context
        print(f"[Standup] 📋 Smart fetch: {len(smart_context_lines)} relevant tickets")

        # Build the formatted prompt
        formatted_prompt = FULL_EXTRACT_PROMPT.format(
            developer=self.developer,
            project_key=self._project_key,
            date=self.data["date"],
            available_tickets=smart_context,
            yesterday=yesterday_for_extraction,
            today=today_for_extraction,
            blockers=blockers_for_extraction,
        )

        try:
            raw = await self._azure(
                formatted_prompt,
                f"Extract standup for {self.developer}",
                max_tokens=500,
            )
            print(f"[Standup] ⏱ Azure extraction: {(time.time()-t0)*1000:.0f}ms")

            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r'^```(?:json)?\s*', '', raw)
                raw = re.sub(r'\s*```$', '', raw)
            extracted = json.loads(raw)

            for field in ("yesterday", "today", "blockers"):
                section = extracted.get(field, {})
                # Prefer the Groq confirmation clean summary (produced right before CONFIRM)
                # over Azure's extraction summary — Azure often returns near-raw text,
                # while Groq's "Yesterday: X. Today: Y. Blockers: Z." parsing gives cleaner display text.
                if field == "yesterday" and self._confirmed_summary and yesterday_for_extraction and yesterday_for_extraction != self.data["yesterday"]["raw"]:
                    self.data[field]["summary"] = yesterday_for_extraction
                elif field == "today" and self._confirmed_summary and today_for_extraction and today_for_extraction != self.data["today"]["raw"]:
                    self.data[field]["summary"] = today_for_extraction
                elif field == "blockers" and self._confirmed_summary and blockers_for_extraction and blockers_for_extraction != (self.data["blockers"]["raw"] or "No blockers"):
                    self.data[field]["summary"] = blockers_for_extraction
                else:
                    self.data[field]["summary"] = section.get("summary", self.data[field]["raw"])
                self.data[field]["tasks"] = section.get("tasks", [])
                ids = self._filter_jira_ids(section.get("jira_ids", []))
                self.data[field]["jira_ids"] = ids
                self._all_jira_ids.update(ids)
                if field == "blockers":
                    self.data[field]["items"] = section.get("items", section.get("tasks", []))

                for su in section.get("status_updates", []):
                    ticket, action = su.get("ticket", ""), su.get("action", "")
                    if ticket and action:
                        filtered = self._filter_jira_ids([ticket])
                        if filtered:
                            self._all_status_updates.append({"ticket": filtered[0], "action": action})
                            print(f"[Standup] 📌 Status: {filtered[0]} → {action}")

            # ── Post-extraction safety net (Change #3) ──
            # Scan raw text AND summaries for ticket IDs that Azure missed
            # Also infer status_updates from completion/planning language
            _TICKET_PATTERN = re.compile(r'\b(' + re.escape(self._project_key) + r'-\d+)\b', re.IGNORECASE)
            _DONE_WORDS = {"completed", "resolved", "finished", "done", "fixed", "wrapped", "closed"}
            _PROGRESS_WORDS = {"begin", "starting", "continuing", "work", "working", "picking", "moving"}

            for field in ("yesterday", "today", "blockers"):
                sources = [self.data[field].get("raw", ""), self.data[field].get("summary", "")]
                found_ids = set()
                for source in sources:
                    for match in _TICKET_PATTERN.finditer(source):
                        found_ids.add(match.group(1).upper())

                existing_ids = set(self.data[field].get("jira_ids", []))
                missing_ids = found_ids - existing_ids
                if missing_ids:
                    print(f"[Standup] 🔍 Safety net: found {missing_ids} in {field} text (missed by Azure)")
                    self.data[field]["jira_ids"].extend(list(missing_ids))
                    self._all_jira_ids.update(missing_ids)

                    # Infer status_updates for missing IDs based on language
                    existing_status_tickets = {su["ticket"] for su in self._all_status_updates}
                    combined_text = " ".join(sources).lower()
                    for tid in missing_ids:
                        if tid in existing_status_tickets:
                            continue
                        if field == "yesterday" and any(w in combined_text for w in _DONE_WORDS):
                            self._all_status_updates.append({"ticket": tid, "action": "done"})
                            print(f"[Standup] 🔍 Safety net status: {tid} → done (completion language in {field})")
                        elif field == "today" and any(w in combined_text for w in _PROGRESS_WORDS):
                            self._all_status_updates.append({"ticket": tid, "action": "in_progress"})
                            print(f"[Standup] 🔍 Safety net status: {tid} → in_progress (planning language in {field})")

        except Exception as e:
            print(f"[Standup] ⚠️  Azure extraction failed: {e}")
            for field in ("yesterday", "today", "blockers"):
                if not self.data[field]["summary"]:
                    self.data[field]["summary"] = self.data[field]["raw"]

        # Generate PM dashboard one-liner (runs after extraction — uses clean summaries)
        try:
            one_liner = await self._generate_one_liner()
            self.data["one_line_summary"] = one_liner
            print(f"[Standup] 📋 PM one-liner: \"{one_liner}\"")
        except Exception as e:
            print(f"[Standup] ⚠️  One-liner generation failed: {e}")
            # Fallback handled in _generate_one_liner itself

        # Classify whether blocker is real (LLM — replaces brittle keyword matching)
        try:
            has_blocker = await self._classify_real_blocker()
            self.data["has_real_blocker"] = has_blocker
            print(f"[Standup] 📋 Real blocker classified: {'YES' if has_blocker else 'NO'}")
        except Exception as e:
            print(f"[Standup] ⚠️  Blocker classification failed: {e}")
            # Conservative fallback: if text is non-trivially long, assume real blocker
            blocker_text = (self.data["blockers"].get("summary") or self.data["blockers"].get("raw") or "").strip()
            self.data["has_real_blocker"] = len(blocker_text.split()) >= 3

        # bg_jira already created above for smart fetch
        if bg_jira:
            await self._auto_create_subtasks(bg_jira)
            await self._auto_create_daily_summary(bg_jira)
            await self._auto_transition_jira(bg_jira)
            await self._auto_assign_sprint(bg_jira)
            await bg_jira.close()

        print(f"[Standup] ✅ Background processing complete ({(time.time()-t0)*1000:.0f}ms total)")

    async def _generate_one_liner(self) -> str:
        """Generate a PM-scannable one-line summary for the dashboard.

        Runs AFTER Azure extraction so we use clean summaries (not raw STT).
        Falls back to simple concatenation if Groq fails.
        """
        yesterday = self.data["yesterday"].get("summary") or self.data["yesterday"].get("raw") or "(no update)"
        today = self.data["today"].get("summary") or self.data["today"].get("raw") or "(no update)"
        blockers = self.data["blockers"].get("summary") or self.data["blockers"].get("raw") or "No blockers"
        jira_ids = ", ".join(sorted(self._all_jira_ids)) if self._all_jira_ids else "none"

        try:
            prompt = PM_ONE_LINER_PROMPT.format(
                developer=self.developer,
                yesterday=yesterday,
                today=today,
                blockers=blockers,
                jira_ids=jira_ids,
            )
            result = await self._groq(prompt, "Generate PM one-liner", max_tokens=60)
            # Clean up: strip quotes, remove trailing periods that LLM sometimes adds
            result = result.strip().strip('"').strip("'").strip()
            # Sanity check: if LLM returned something too long or empty, use fallback
            if not result or len(result.split()) > 30:
                raise ValueError(f"Invalid one-liner length: {len(result.split())} words")
            return result
        except Exception as e:
            print(f"[Standup] ⚠️  One-liner LLM failed, using fallback: {e}")
            # Simple template fallback
            y = yesterday[:50].rstrip(".") if yesterday != "(no update)" else ""
            t = today[:50].rstrip(".") if today != "(no update)" else ""
            parts = []
            if y:
                parts.append(f"Yesterday: {y}")
            if t:
                parts.append(f"today: {t}")
            if blockers and blockers.lower().strip() not in ("no blockers", "none", ""):
                parts.append(f"blocked: {blockers[:40]}")
            return ". ".join(parts) + "." if parts else "Standup completed."

    async def _classify_real_blocker(self) -> bool:
        """Use LLM to decide if the blockers field describes a real work blocker.

        Replaces brittle keyword matching that false-positives on substrings
        (e.g. "manager" contains "na" which was treated as "no blocker" shortcut).

        Returns True for real blockers, False for "no blockers"-style phrasings.
        Runs during background_finalize, result stored in has_real_blocker field.
        """
        # Get the blocker text — prefer summary (cleaner), fall back to raw
        text = (self.data["blockers"].get("summary") or self.data["blockers"].get("raw") or "").strip()

        # Empty text = no blocker (efficiency shortcut, not classification logic)
        if not text:
            return False

        # Ask LLM
        prompt = BLOCKER_CLASSIFY_PROMPT.format(text=text)
        result = await self._groq(prompt, "Classify real blocker", max_tokens=5)
        # Parse: look for YES/NO in the response (robust to whitespace, periods, quotes)
        normalized = result.strip().upper().strip('"').strip("'").strip(".").strip()
        return normalized.startswith("YES")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _reset_data(self):
        self.data["yesterday"] = {"summary": "", "tasks": [], "jira_ids": [], "raw": ""}
        self.data["today"]     = {"summary": "", "tasks": [], "jira_ids": [], "raw": ""}
        self.data["blockers"]  = {"summary": "", "items": [], "jira_ids": [], "raw": ""}
        self._all_jira_ids.clear()
        self._all_status_updates.clear()

    def _filter_jira_ids(self, ids: list) -> list:
        conversation_text = " ".join(self._history).upper()
        valid = []
        for tid in ids:
            if tid.upper().startswith(self._project_key.upper() + "-"):
                valid.append(tid)
            elif tid.upper() in conversation_text:
                valid.append(tid)
            else:
                print(f"[Standup] 🚫 Filtered hallucinated ID: {tid}")
        return valid

    # ── Jira (background only) ────────────────────────────────────────────────

    async def _auto_create_subtasks(self, jira_client=None):
        """Create consolidated subtasks under each mentioned ticket.
        Blockers are included in ALL subtask descriptions (not just blocker-specific tickets)."""
        jira = jira_client or self.jira
        if not jira or not jira.enabled or not self._all_jira_ids:
            return
        date_str = time.strftime("%B %d, %Y", time.gmtime())
        date_short = time.strftime("%Y-%m-%d", time.gmtime())
        blockers_summary = self.data["blockers"].get("summary") or self.data["blockers"].get("raw") or "No blockers"
        has_real_blockers = blockers_summary.lower().strip() not in ("no blockers", "none", "no blockers.")

        for tid in self._all_jira_ids:
            sections = []
            in_yesterday = tid in self.data["yesterday"]["jira_ids"]
            in_today = tid in self.data["today"]["jira_ids"]
            in_blockers = tid in self.data["blockers"]["jira_ids"]

            if in_yesterday:
                yesterday_text = self.data["yesterday"].get("summary") or self.data["yesterday"].get("raw") or ""
                sections.append(f"✅ Yesterday: {yesterday_text}")

            if in_today:
                today_text = self.data["today"].get("summary") or self.data["today"].get("raw") or ""
                sections.append(f"📋 Today: {today_text}")

            # Always include blockers in every subtask (not just blocker-specific tickets)
            if in_blockers:
                sections.append(f"🚫 Blocker: {blockers_summary}")
            elif has_real_blockers:
                # Include blocker info even for non-blocker tickets (team-wide impact)
                sections.append(f"⚠️ Team blocker: {blockers_summary}")
            else:
                sections.append(f"✅ No blockers reported")

            if not sections:
                sections.append("Mentioned in standup")

            description = f"📋 Standup Update — {date_str}\nDeveloper: {self.developer}\n\n" + "\n".join(sections)
            summary = f"Standup — {self.developer} ({date_short})"

            try:
                result = await jira.create_subtask(
                    parent_key=tid,
                    summary=summary,
                    description=description,
                    priority="Medium",
                    labels=["standup", f"standup-{date_short}"],
                )
                print(f"[Standup] 📝 Subtask {result['key']} created under {tid}")
            except Exception as e:
                print(f"[Standup] ⚠️  Subtask under {tid} failed: {e}")
                try:
                    await jira.add_comment(tid, description)
                    print(f"[Standup] 💬 Fallback comment on {tid}")
                except Exception as e2:
                    print(f"[Standup] ⚠️  Comment fallback on {tid} also failed: {e2}")

    async def _auto_create_daily_summary(self, jira_client=None):
        """Create or update a single daily standup summary ticket with all developers' updates."""
        jira = jira_client or self.jira
        if not jira or not jira.enabled:
            return
        date_short = time.strftime("%Y-%m-%d", time.gmtime())
        date_str = time.strftime("%B %d, %Y", time.gmtime())
        summary_title = f"Daily Standup — {date_short}"

        # Build this developer's section
        yesterday_text = self.data["yesterday"].get("summary") or self.data["yesterday"].get("raw") or "(no answer)"
        today_text = self.data["today"].get("summary") or self.data["today"].get("raw") or "(no answer)"
        blockers_text = self.data["blockers"].get("summary") or self.data["blockers"].get("raw") or "No blockers"

        yesterday_ids = self.data["yesterday"].get("jira_ids", [])
        today_ids = self.data["today"].get("jira_ids", [])
        all_ids = list(set(yesterday_ids + today_ids + list(self._all_jira_ids)))

        developer_section = (
            f"👤 {self.developer}\n"
            f"  Yesterday: {yesterday_text}\n"
            f"  Today: {today_text}\n"
            f"  Blockers: {blockers_text}\n"
            f"  Tickets: {', '.join(all_ids) if all_ids else 'None'}\n"
        )

        try:
            # Search for existing daily standup ticket
            jql = f'project = {self._project_key} AND summary ~ "Daily Standup — {date_short}" ORDER BY created DESC'
            results = await jira.search_jql(jql, max_results=1)

            if results and results[0].get("key"):
                # UPDATE: add this developer as a comment
                existing_key = results[0]["key"]
                comment = f"📋 Standup Update — {date_str}\n\n{developer_section}"
                await jira.add_comment(existing_key, comment)
                print(f"[Standup] 📋 Updated daily summary {existing_key} with {self.developer}'s standup")
            else:
                # CREATE: new daily standup ticket
                description = f"📋 Daily Standup — {date_str}\n\n{developer_section}"
                result = await jira.create_ticket(
                    summary=summary_title,
                    issue_type="Task",
                    priority="Low",
                    description=description,
                    labels=["daily-standup", f"standup-{date_short}"],
                )
                print(f"[Standup] 📋 Created daily summary {result.get('key', '?')}: {summary_title}")
        except Exception as e:
            print(f"[Standup] ⚠️  Daily summary failed: {e}")

    async def _auto_transition_jira(self, jira_client=None):
        jira = jira_client or self.jira
        if not jira or not jira.enabled or not self._all_status_updates:
            return
        final = {}
        for su in self._all_status_updates:
            final[su["ticket"]] = su["action"]
        ACTION_MAP = {"done": "Done", "in_progress": "In Progress", "blocked": None}
        for tid, action in final.items():
            target = ACTION_MAP.get(action)
            if not target:
                if action == "blocked":
                    print(f"[Standup] ⚠️  {tid} blocked — noted in comment")
                continue
            try:
                result = await jira.transition_ticket(tid, target)
                if result.get("action") == "already_done":
                    print(f"[Standup] ℹ️  {tid} already at '{result['already_at']}'")
                else:
                    print(f"[Standup] 🔄 {tid}: → {result.get('new_status', target)}")
            except Exception as e:
                print(f"[Standup] ⚠️  Transition {tid} → {target} failed: {e}")

    async def _auto_assign_sprint(self, jira_client=None):
        """Assign all mentioned tickets to active sprint."""
        jira = jira_client or self.jira
        if not jira or not jira.enabled or not self._all_jira_ids:
            return
        ticket_ids = list(self._all_jira_ids)
        try:
            success = await jira.move_to_sprint(ticket_ids)
            if not success:
                print(f"[Standup] ⚠️  No active sprint — tickets remain in backlog")
        except Exception as e:
            print(f"[Standup] ⚠️  Sprint assignment failed: {e}")

    # ── Silence timer ─────────────────────────────────────────────────────────

    def _start_silence_timer(self):
        self._cancel_silence_timer()
        self._silence_task = asyncio.create_task(self._silence_reprompt())

    def _cancel_silence_timer(self):
        if self._silence_task and not self._silence_task.done():
            self._silence_task.cancel()

    async def _silence_reprompt(self):
        try:
            await asyncio.sleep(10.0)
            prompts = {
                StandupState.ASK_YESTERDAY: "Still there? What did you work on yesterday?",
                StandupState.ASK_TODAY: "What's your plan for today?",
                StandupState.ASK_BLOCKERS: "Any blockers holding you up?",
                StandupState.CONFIRM: "Is the summary correct, or do you want to change something?",
            }
            prompt = prompts.get(self.state)
            if prompt:
                # Skip re-prompt if new text already arrived (user started speaking)
                if hasattr(self, '_check_buffer_fn') and self._check_buffer_fn and self._check_buffer_fn():
                    print(f"[Standup] ⏰ Skipped re-prompt — user already speaking")
                    # IMPORTANT: restart timer anyway, in case the detection was stale data.
                    # If user really is speaking, their transcript will process and cancel this timer.
                    # If detection was wrong (stale partial_text), we'll re-check in another 10s.
                    self._start_silence_timer()
                    return
                print(f"[Standup] ⏰ Re-prompting ({self.state.name})")
                self._add_history("Sam", prompt)
                self._playing_reprompt = True
                try:
                    await self.speak(prompt, "standup-reprompt", self._generation)
                finally:
                    self._playing_reprompt = False
                self._start_silence_timer()
        except asyncio.CancelledError:
            pass

    def get_result(self) -> dict:
        return {
            "developer": self.data["developer"],
            "date": self.data["date"],
            "started_at": self.data["started_at"],
            "completed_at": self.data.get("completed_at", ""),
            "completed": self.data["completed"],
            "yesterday": {
                "summary": self.data["yesterday"].get("summary") or self.data["yesterday"]["raw"],
                "raw": self.data["yesterday"].get("raw", ""),
                "tasks": self.data["yesterday"]["tasks"],
                "jira_ids": self.data["yesterday"]["jira_ids"],
            },
            "today": {
                "summary": self.data["today"].get("summary") or self.data["today"]["raw"],
                "raw": self.data["today"].get("raw", ""),
                "tasks": self.data["today"]["tasks"],
                "jira_ids": self.data["today"]["jira_ids"],
            },
            "blockers": {
                "summary": self.data["blockers"].get("summary") or self.data["blockers"]["raw"],
                "raw": self.data["blockers"].get("raw", ""),
                "items": self.data["blockers"]["items"],
                "jira_ids": self.data["blockers"]["jira_ids"],
            },
            "all_jira_ids": list(self._all_jira_ids),
            "status_updates": self._all_status_updates,
            "one_line_summary": self.data.get("one_line_summary", ""),
            "has_real_blocker": self.data.get("has_real_blocker", False),
        }