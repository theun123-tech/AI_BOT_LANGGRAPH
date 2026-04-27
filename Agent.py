
"""
Agent.py — Groq Llama 3.3 70B + In-Memory RAG

Memory architecture:
  1. RAG store: Every exchange is embedded (Azure OpenAI text-embedding-3-small)
     and stored in-memory. On query, cosine similarity finds relevant past exchanges.
     Catches semantic matches like "money" → "budget" that keyword search misses.
  2. Meeting log: Full text of every exchange (fallback + debugging)
  3. Recent history: Last 10 LLM turns for conversation flow

Embedding happens async in background — never blocks the response pipeline.
If embeddings fail, falls back to keyword search automatically.
"""

import os
import asyncio
import re
import time
import json
import numpy as np
from openai import AsyncOpenAI
from typing import List, Optional

# Fix B: rotating Groq client (shared 12-key pool with Trigger + NLU)
from groq_client import GroqRotatingClient


# ── Debug logger — writes all prompt inputs to file ──────────────────────────
DEBUG_PROMPTS_FILE = "debug_prompts.txt"

def _debug_log(label: str, **kwargs):
    """Append a debug entry to debug_prompts.txt with timestamp and all variables."""
    if os.environ.get("DEBUG_SAVE_AUDIO", "").lower() not in ("1", "true", "yes"):
        return
    try:
        ts = time.strftime("%H:%M:%S")
        with open(DEBUG_PROMPTS_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"[{ts}] {label}\n")
            f.write(f"{'='*80}\n")
            for key, val in kwargs.items():
                val_str = str(val) if val else "(EMPTY)"
                f.write(f"  {key}:\n    {val_str}\n")
            f.write(f"\n")
    except Exception:
        pass  # never crash on debug logging


# ══════════════════════════════════════════════════════════════════════════════
# IN-MEMORY RAG STORE
# ══════════════════════════════════════════════════════════════════════════════

class MeetingRAG:
    """In-memory vector store for meeting transcripts.
    Uses fastembed (free, local, ~200MB). No API key needed.
    Model: BAAI/bge-small-en-v1.5 — 130MB, 384-dim, fast on CPU.
    """

    def __init__(self):
        self._entries: list[dict] = []
        self._embed_queue: asyncio.Queue = asyncio.Queue()
        self._embed_task: Optional[asyncio.Task] = None
        self._model = None
        self._ready = False

        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            self._ready = True
            print("[RAG] Local embeddings ready (BAAI/bge-small-en-v1.5, fastembed)")
        except ImportError:
            print("[RAG] ⚠️  fastembed not installed — keyword fallback only")
        except Exception as e:
            print(f"[RAG] ⚠️  Model load failed: {e} — keyword fallback only")

    def start_background_embedder(self):
        if self._ready and not self._embed_task:
            self._embed_task = asyncio.create_task(self._embedding_worker())

    async def _embedding_worker(self):
        loop = asyncio.get_event_loop()
        while True:
            try:
                entry = await self._embed_queue.get()
                vector = await loop.run_in_executor(
                    None, self._embed_sync, entry["text"]
                )
                if vector is not None:
                    entry["vector"] = vector
                    self._entries.append(entry)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[RAG] Embed worker error: {e}")

    def _embed_sync(self, text: str) -> Optional[np.ndarray]:
        if not self._model:
            return None
        try:
            # fastembed returns a generator — get first result
            vectors = list(self._model.embed([text]))
            return np.array(vectors[0], dtype=np.float32)
        except Exception as e:
            print(f"[RAG] Embedding failed: {e}")
            return None

    def add(self, speaker: str, text: str):
        """Queue an exchange for embedding (non-blocking)."""
        entry = {
            "text": f"{speaker}: {text}",
            "speaker": speaker,
            "time": time.time(),
            "vector": None,
        }
        if self._ready:
            try:
                self._embed_queue.put_nowait(entry)
            except Exception:
                pass
        else:
            self._entries.append(entry)

    async def search(self, query: str, top_k: int = 5) -> List[str]:
        """Find relevant past exchanges by cosine similarity.
        Falls back to keyword matching if embeddings unavailable.
        """
        if not self._entries:
            return []

        # Vector search
        if self._ready and self._model:
            loop = asyncio.get_event_loop()
            query_vector = await loop.run_in_executor(
                None, self._embed_sync, query
            )

            if query_vector is not None:
                scored = []
                for entry in self._entries:
                    if entry["vector"] is not None:
                        sim = self._cosine_sim(query_vector, entry["vector"])
                        scored.append((sim, entry["text"]))

                if scored:
                    scored.sort(key=lambda x: x[0], reverse=True)
                    results = [text for sim, text in scored[:top_k] if sim > 0.3]
                    if results:
                        print(f"[RAG] Vector search: {len(results)} hits for \"{query[:50]}\"")
                        return results

        # Fallback: keyword search
        return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int = 5, exclude_text: str = "") -> List[str]:
        stop = {"the", "a", "an", "is", "are", "was", "were", "what", "who",
                "how", "when", "where", "why", "did", "do", "does", "can",
                "could", "would", "should", "we", "i", "you", "they", "it",
                "about", "tell", "me", "something", "discuss", "talked", "sam"}
        query_words = {w for w in query.lower().split() if w not in stop and len(w) > 2}
        if not query_words:
            return []
        # Normalize exclude text for comparison
        exclude_lower = exclude_text.lower().strip() if exclude_text else ""
        scored = []
        for entry in self._entries:
            entry_lower = entry["text"].lower().strip()
            # Skip if this entry IS the current utterance (prevent echo)
            if exclude_lower and (exclude_lower in entry_lower or entry_lower.endswith(exclude_lower)):
                continue
            hits = sum(1 for w in query_words if w in entry_lower)
            if hits > 0:
                scored.append((hits, entry["text"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [text for _, text in scored[:top_k]]
        if results:
            print(f"[RAG] Keyword fallback: {len(results)} hits for \"{query[:50]}\"")
        return results

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / norm) if norm > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._entries)

    def clear(self):
        self._entries.clear()


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

# ── FAST ROUTER — classifies before main LLM ────────────────────────────────
ROUTER_PROMPT = """Classify this message. Reply with ONLY one tag:

[PM] — Sam can answer this PERFECTLY using only the conversation context and his personality. No data lookup needed.
[RESEARCH] — Answering this correctly REQUIRES looking something up. Data, facts, ticket info, actions, or external knowledge.

THE KEY QUESTION: "Can Sam give a CORRECT, SPECIFIC answer without checking any data?"
- If YES → [PM]
- If NO, or if there's ANY doubt → [RESEARCH]

[RESEARCH] is the SAFE default. Routing to [PM] when data is needed gives a WRONG answer (Sam invents facts). Routing to [RESEARCH] when it's not needed just adds 1 second. Always prefer [RESEARCH] over risking a wrong answer.

[PM] is ONLY for:
- Greetings and small talk ("hi", "how are you", "thanks")
- Acknowledgments ("sounds good", "okay", "got it")
- Asking Sam to repeat or recall what was said in THIS conversation
- Questions about Sam himself (who are you, what do you do)
- Reporting bugs or requesting features (these get captured automatically)
- Simple opinions or reactions that don't need facts

[RESEARCH] is for EVERYTHING ELSE, including but not limited to:
- Any question about tickets, sprints, project status, or team work
- Any request to perform an action (move, create, update, check, list)
- Any question needing real-world facts (people, companies, tech, history)
- Any technical or implementation question
- Any "how to", "what is", "tell me about", "who is" question
- Follow-up questions asking for more detail about a previous topic
- Anything involving numbers, dates, status, or data the user expects to be accurate

CRITICAL — Speech-to-text often garbles ticket references:
- "thirty one" / "ticket thirty one" / "scrum thirty one" = a ticket ID
- "move it to done" / "get it to done" / "makes from X to done" = a Jira transition
- If the user mentions ANY number and ANY status in the same sentence, it's likely a Jira action → [RESEARCH]

ONLY reply with [PM] or [RESEARCH]. Nothing else."""

# ── PM PROMPT — only handles personality + PM answers ────────────────────────
PM_PROMPT = """You are Sam, a senior PM at AnavClouds Software Solutions. You're on a live voice call in a meeting.

HOW YOU TALK:
- Like a real person in a meeting, not a chatbot. Use "yeah", "so", "sure", "right".
- NEVER use tonal fillers that imply frustration or correction: "seriously", "honestly",
  "actually", "obviously", "as I said", "I just said". These sound condescending when
  the user is legitimately asking for clarification.
- React to what the person ACTUALLY said before answering. Show you're listening.
- Use their name naturally (not every time — that's weird).
- Contractions always. Never say "I am" when you can say "I'm".
- Throw in light humor when it fits, but never at the user's expense.
- If they ask for a joke, deliver the FULL joke with setup AND punchline in one response.
- If they ask you to repeat yourself, just repeat it warmly — no hint of "I already told you".

YOUR BACKGROUND:
- Senior PM at AnavClouds (Salesforce + AI company). You handle sprints, budgets, timelines, CRM rollouts.
- Confident and warm. You deflect weird off-topic questions with light humor,
  but never sarcasm directed AT the user.
- You remember what was said in the meeting (see MEETING MEMORY below).

LENGTH — THIS IS CRITICAL:
- You are SPEAKING on a live call. Every extra word wastes everyone's time.
- 1-2 sentences MAXIMUM. Aim for 15-25 words total.
- Think of it like a quick reply in a meeting — not a presentation.
- If your response would take more than 5 seconds to say out loud, it's TOO LONG. Cut it.
- NEVER give lengthy explanations, lists, or multi-point answers. That's what [RESEARCH] is for.

OTHER RULES:
- If user asks to REPEAT something: rephrase your last answer differently.
- Say something NEW each time — don't repeat yourself.
- No markdown, no lists, no asterisks, no bullet points, no parenthetical actions like (laughs) or (sighs) — this is voice output, not a script.
- If you don't have specific data (ticket IDs, numbers, facts), DON'T invent them. Say something general or offer to look it up.

MEETING MEMORY: Use this to answer about past discussions if provided."""

RESEARCH_PROMPT = """You are Sam, a senior PM at AnavClouds Software Solutions. You're on a live voice call.
The user asked a question that needs data. You have access to project data from Jira AND web research results.
Use whatever data is relevant to give the BEST possible answer.

PROJECT CONTEXT (your current Jira tickets):
{jira_context}

RELATED TICKETS FOUND (matching this question):
{related_tickets}

JIRA ACTION RESULTS (if any action was taken):
{jira_action}

WEB RESEARCH:
{web_results}

CONVERSATION SO FAR:
{conversation}

HOW TO RESPOND:
- For factual questions: give the answer directly, add useful context
- For Jira queries (sprint status, ticket details): summarize the data with insight, not just numbers
- For Jira actions (move ticket, create ticket): confirm what was done and add context
- For feasibility/implementation questions: assess feasibility, reference related tickets by ID, outline approach, estimate complexity (Low/Medium/High), offer to create a ticket
- Reference specific ticket IDs (e.g. SCRUM-12) when relevant
- If the data doesn't have the answer, say so naturally

VOICE RULES:
- You are SPEAKING on a live call, not writing. Be natural and conversational.
- No bullet points, no markdown, no lists, no parenthetical actions.
- Use "yeah", "honestly", "look", "so basically" — sound like a real PM in a meeting.
- For simple answers: 2-3 sentences. For complex analysis: up to 120 words.
- Skip filler — the user already heard one. Jump straight to the answer.
- Be confident and specific, not vague."""

INTERRUPT_PROMPT = """You are Sam, a witty senior PM. You were interrupted.
Reply in ONE sentence — 15 words max. Be quick, natural.
Start with: "Oh," / "Right," / "Sure," / "Got it," — then pivot to their question."""

SEARCH_QUERY_PROMPT = """Convert the user's message into a DESCRIPTIVE Google search query (8-15 words).
Make the query specific and detailed so Google returns the best results.
Replace 'our company'/'my company'/'our'/'we' with 'AnavClouds Software Solutions'.
If the user does NOT mention the company, do NOT add AnavClouds.
Remove 'Sam' and filler words.
For multi-part questions, include all parts in a single descriptive query.

IMPORTANT — Use the project context (if provided) to make SMARTER queries:
- If tickets mention specific technologies (React, Node.js, MongoDB, etc.), include them in the query
- If the user asks about a feature, add the tech stack from related tickets
- Example: user asks "add real-time notifications" and tickets mention WebSocket + Node.js
  → "real-time notifications implementation WebSocket Node.js best practices"

OUTPUT "SKIP" (no search needed) when a web search would NOT help:
- Questions about internal project data, tickets, sprints, tasks, or team activity
- Requests to perform Jira actions (move, create, update, list tickets)
- Questions about what was discussed in this conversation (memory/recall)
- Anything only the project team would know — Google has no access to private project data

OUTPUT a search query when a web search WOULD help:
- Technical questions, best practices, implementation approaches
- Real-world facts about people, companies, news, events
- Technology comparisons, architecture decisions
- Feasibility and "how to build X" questions — these ALWAYS benefit from web search

Examples:
"What are the hourly rates?" → "What is the hourly rate and pricing of AnavClouds Software Solutions"
"Who is the CEO?" → "Who is the CEO and founder of AnavClouds Software Solutions company"
"Can we add a sales funnel?" → "sales funnel implementation approach best practices"
"Should we switch to PostgreSQL?" → "MongoDB vs PostgreSQL migration comparison pros cons"
"Best way to implement auth?" → "authentication implementation best practices Node.js React"
"What are my open tickets?" → SKIP
"Move SCRUM-15 to done" → SKIP
"What's the sprint status?" → SKIP
"How many tickets are in progress?" → SKIP
"What was my previous question?" → SKIP
"Create a ticket for the login bug" → SKIP

Output ONLY the search query or SKIP. No quotes, no explanation."""


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 4 PROMPTS — Unified research planner + SerpAPI-direct + dynamic filler
# ═══════════════════════════════════════════════════════════════════════════════

# Unified research planner: ONE Groq call outputs a JSON plan for the research
# flow. Replaces separate route sub-classifier + query builder + ticket relevance
# detection. Uses llama-3.3-70b for structured-output reliability.
UNIFIED_RESEARCH_PROMPT = """You are a research planner for Sam, a PM on a live voice call. The user asked a question that needs a data lookup. Your job is to produce a JSON plan that tells downstream code what to do.

User question: "{user_text}"

Recent conversation:
{conversation}

Available tickets in project cache (top entries, by recent/priority):
{ticket_previews}

Output a JSON object with EXACTLY these fields:
{{
  "question_type": "feasibility" | "tech_switch" | "best_practices" | "jira_status" | "internal_org" | "general",
  "relevant_cached_tickets": ["SCRUM-1", "SCRUM-2"],
  "needs_fresh_jira": true | false,
  "jira_search_terms": "short search phrase or empty",
  "web_search_query": "short Google query 6-12 words",
  "serpapi_context_features": ["feature 1 description", "feature 2 description"]
}}

Field rules:
- "question_type": Pick the best single match:
  - "feasibility" = can we build/add X (technical)
  - "tech_switch" = should we change from A to B (technical stack decision)
  - "best_practices" = how to do X well (technical how-to)
  - "jira_status" = show me tickets / move tickets / sprint status
  - "internal_org" = questions about the COMPANY, TEAM, PEOPLE, or INTERNAL STRUCTURE. Examples: "who is our CEO", "how many employees do we have", "what's our office address", "who's on the team", "what's our company policy", "who runs engineering", "what do we do as a company". Anything that asks about private/internal company facts that wouldn't be on the public web reliably.
  - "general" = anything else needing research (news, facts, public knowledge)
- "relevant_cached_tickets": List ticket keys from the cache above that relate to this question. Max 5. Empty list if none relate.
- "needs_fresh_jira": true ONLY if the question mentions concepts/features not covered by any cached ticket (the cache might be missing relevant work). Default false. For internal_org questions, set to false.
- "jira_search_terms": Short phrase (2-5 words) for a Jira text search. Only populated if needs_fresh_jira is true.
- "web_search_query": A clean Google search query. For jira_status and internal_org questions, output exactly "SKIP".
- "serpapi_context_features": Natural-language descriptions of project features relevant to this question (no ticket IDs). Max 6 items. Derived from cached tickets + conversation. Empty list for jira_status and internal_org questions.

CRITICAL: When in doubt about whether a question is about the COMPANY/TEAM/PEOPLE vs a TECHNICAL topic, classify as "internal_org". This prevents the system from hallucinating company-specific facts.

Output ONLY the JSON object. No markdown, no explanation, no code fences."""


# SerpAPI-direct persona prompt: builds the actual query sent to Google AI Mode.
# Proven through testing to return TTS-ready responses when TTS hint is included.
SERPAPI_DIRECT_PROMPT = """I'm a senior developer at AnavClouds Software Solutions in a live client meeting. Client asked: "{question}"

Our project context:
{project_context}

{conversation_hint}Answer {length} for TTS, clean sentences. No citations, no markdown, no bullet points, no source references. Speak naturally like telling this to a client in real-time.

Always use your trusted web sources for any specific facts about companies, people, products, or technologies. If you cannot verify a specific fact from trusted sources, say so plainly rather than guessing. STRICTLY make sure to word limit MAX 3-4 sentences dont exceed 60 words. Be concise, clear, and client-friendly in your tone."""


# Dynamic filler prompt: generates ONE sentence of contextual acknowledgment.
# Runs in parallel with router/trigger decisions. 400ms timeout.
FILLER_PROMPT = """You are Sam, a senior PM on a live voice call. Someone just asked you a question that needs a quick research lookup. Generate ONE warm, natural filler sentence to say BEFORE looking up the answer.

Rules:
- ONE sentence, 12 to 22 words (warm and conversational, not clipped)
- Acknowledge the topic naturally with a touch of warmth (do NOT answer the question)
- Sound like a friendly colleague pausing to check
- Use natural phrases like "let me", "one sec", "checking", "hmm", "yeah", "good question", "let me see", "give me a moment"
- Feel free to briefly acknowledge what the question is about, in your own words
- Do NOT commit to any facts you don't have
- No meta-commentary ("as an AI", "sure, here's")

Examples:
Question: "Can we migrate to PostgreSQL?"
Filler: "Hmm, good question — let me check what moving to Postgres would involve for our setup."

Question: "What's the status of SCRUM-31?"
Filler: "Yeah, one sec, let me pull up SCRUM-31 and see where it's currently at."

Question: "How's the sprint going?"
Filler: "Right, let me check the sprint board real quick and I'll give you the rundown."

Question: "Who's the CEO of the company?"
Filler: "Oh good question — let me pull up that info for you, give me just a second."

Now generate a filler for: "{question}"

Output ONLY the filler sentence. No quotes, no explanation, no prefix."""


# Warmer, more conversational fillers (~2-4s each).
# Earlier "ultra-short" versions (e.g. "One sec, checking that.") felt
# robotic and curt. Real PMs pause and acknowledge naturally before they
# look something up — these capture that warmth.
# Trade-off: ~1-2s longer audio per filler, but feels human, not clipped.
FILLERS = [
    "Hmm, good question — let me pull that up real quick for you.",
    "Yeah, hold on a sec, let me check on that and get back to you.",
    "Right, give me just a moment to look into that one.",
    "Oh interesting — let me dig into that and see what I can find.",
    "Sure thing, one sec while I check the details on that.",
    "Yeah let me grab the latest on that, won't take long.",
    "Good question — give me a second to pull up the info.",
    "Hmm, let me have a look and figure out where things stand.",
    "Alright, hold on — checking on that for you right now.",
    "Yeah, let me see what I can find on that one.",
    "Okay, give me a moment to look that up properly.",
    "Sure, let me pull up the details and get back to you.",
]


# ── End-of-Turn Classifier (RESPOND vs WAIT) ────────────────────────────────
EOT_PROMPT = """You are Sam, an AI participant in a live meeting. Someone is speaking to you. Based on the conversation, decide:

Should you RESPOND now, or WAIT for them to continue?

{context_block}Current utterance: "{text}"

RESPOND if ANY of these are true:
- The utterance contains a question (even rhetorical like "Right?" or "You know?")
- The utterance is a request or command ("Tell me...", "Can you...")
- The utterance is a short reaction or statement directed at you
- The speaker seems to be done talking and waiting for your input
- You are unsure — RESPOND is always safer than making someone wait

WAIT only if ALL of these are true:
- The utterance is clearly mid-sentence (grammatically incomplete)
- OR the speaker is obviously setting up a longer explanation and hasn't asked anything yet
- AND there is no question, request, or floor-handoff signal

Reply with one word: RESPOND or WAIT"""


STANDUP_EOT_PROMPT = """You are Sam, an AI PM running a developer standup. Decide if the developer finished their answer.

The developer is answering about: {standup_phase}

{context_block}Current utterance: "{text}"

RESPOND (developer is done) if ANY of these:
- Has a verb + ticket ID: "worked on SCRUM-5", "completed SCRUM-1" → DONE
- Has a complete thought: "no blockers", "nothing", "same as yesterday" → DONE
- Confirmation/disagreement: "sounds right", "yes", "no", "I want to change" → DONE
- Sentence ends with a period and has 4+ words → likely DONE
- You are unsure → RESPOND (don't make them wait)

WAIT (developer is still talking) ONLY if:
- Ends with a preposition with no object: "worked on", "looking at", "waiting for"
- Ends with a conjunction: "and", "but", "or"
- Ends with a comma: "I worked on scrum five,"
- Has a ticket trigger word with no number: ends with "scrum", "ticket", "number"
- Clearly mid-sentence: "actually is a blocker like"

DEFAULT TO RESPOND. Only WAIT when the sentence is obviously cut mid-thought.

Reply with one word: RESPOND or WAIT"""


# ══════════════════════════════════════════════════════════════════════════════
# PM AGENT
# ══════════════════════════════════════════════════════════════════════════════

class PMAgent:
    def __init__(self):
        # Fix B: shared rotating Groq client (12-key pool)
        self.client = GroqRotatingClient(tag="[Agent]")
        self.model = "llama-3.1-8b-instant"

        # Phase 3.5: reference to DialogueManager for meeting state access
        # Set by websocket_server.BotSession.setup() via set_dialogue_manager()
        self._dialogue_manager = None

        # Recent LLM history — last 10 turns
        self.history: list[dict] = []

        # RAG store — embeds + retrieves meeting exchanges
        self.rag = MeetingRAG()

    def start(self):
        """Call once after event loop is running to start background embedder + warmup."""
        self.rag.start_background_embedder()
        asyncio.create_task(self._warmup())

    async def _warmup(self):
        """Pre-establish TCP connection to Groq — saves ~300ms on first real call."""
        try:
            await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            print("[Agent] ✅ Groq connection warmed up")
        except Exception:
            pass

    def _get_web_search(self):
        if not hasattr(self, '_web_search') or self._web_search is None:
            from external_apis import WebSearch
            self._web_search = WebSearch()
        return self._web_search

    # ── Memory ────────────────────────────────────────────────────────────────

    def log_exchange(self, speaker: str, text: str):
        """Store an exchange in RAG. Called by websocket_server for every transcript."""
        self.rag.add(speaker, text)

    async def _build_context(self, user_text: str, context: str) -> str:
        """Build meeting context for the system prompt (NOT the user message).
        Returns a string to append to the system prompt with relevant memory + recent convo.
        Filters current utterance from RAG to prevent echo."""
        parts = []

        # Phase 3.5: inject DialogueManager meeting state (agenda/scope/prior) at TOP
        # so Sam naturally grounds in today's meeting context before RAG/history.
        meeting_state_block = self._format_meeting_state()
        if meeting_state_block:
            parts.append(meeting_state_block)

        # Fast keyword search — exclude current utterance to prevent echo
        rag_results = self.rag._keyword_search(user_text, top_k=2, exclude_text=user_text)
        if rag_results:
            parts.append("MEETING MEMORY (relevant past exchanges):\n" + "\n".join(rag_results))

        # Recent conversation — last 4 lines for flow
        if context:
            recent = "\n".join(context.split("\n")[-4:])
            parts.append(f"RECENT CONVERSATION:\n{recent}")

        if not parts:
            return ""

        full_context = "\n\n".join(parts)

        _debug_log("BUILD CONTEXT (system prompt appendix)",
                   user_text=user_text,
                   convo_history_raw=context or "(EMPTY)",
                   rag_results=rag_results or "(NONE)",
                   built_context=full_context)

        return full_context

    # ── Phase 3.5: DialogueManager state formatting ──────────────────────────

    def set_dialogue_manager(self, dm) -> None:
        """Register the session's DialogueManager so Agent can read live state.

        Called by BotSession.setup() after DialogueManager.initialize().
        Safe to call with None to clear (e.g., on session teardown).
        """
        self._dialogue_manager = dm

    def _format_meeting_state(self) -> str:
        """Format DialogueManager state as a natural-language block for the
        system prompt. Includes agenda, scope, prior meeting summaries.

        Returns empty string if no DialogueManager wired, not initialized,
        or nothing useful to add. Keeps additions concise to control token cost
        (~200-400 tokens typical).
        """
        if self._dialogue_manager is None:
            return ""
        try:
            state = self._dialogue_manager.get_state_snapshot()
        except Exception:
            return ""
        if not state or not state.get("_initialized"):
            return ""

        lines: list[str] = []

        # Agenda — the most important signal Sam needs
        agenda = state.get("agenda") or []
        if agenda:
            lines.append("TODAY'S AGENDA (what Sam is here to cover):")
            for i, item in enumerate(agenda, 1):
                if isinstance(item, dict):
                    title = (item.get("title") or "").strip()
                    status = item.get("status", "pending")
                else:
                    title = str(item).strip()
                    status = "pending"
                if not title:
                    continue
                marker = ""
                if status == "resolved":
                    marker = " [DONE]"
                elif status == "in_progress":
                    marker = " [CURRENT]"
                elif status == "deferred":
                    marker = " [SKIPPED]"
                lines.append(f"  {i}. {title}{marker}")

        # Scope — help Sam politely redirect out-of-scope topics
        scope_in = state.get("scope_in") or []
        scope_out = state.get("scope_out") or []
        if scope_in:
            if lines:
                lines.append("")
            lines.append(f"IN SCOPE: {', '.join(str(s) for s in scope_in if s)}")
        if scope_out:
            if not scope_in and lines:
                lines.append("")
            lines.append(
                f"OUT OF SCOPE (acknowledge but gently redirect): "
                f"{', '.join(str(s) for s in scope_out if s)}"
            )

        # Prior meetings — short summaries of recent calls with these participants
        priors = state.get("prior_meeting_summaries") or []
        if priors:
            if lines:
                lines.append("")
            lines.append(f"FROM {len(priors)} PRIOR MEETING(S) WITH THESE PEOPLE:")
            for p in priors[:3]:
                text = self._extract_prior_summary_text(p)
                if text:
                    # Cap each summary to keep the prompt compact
                    if len(text) > 220:
                        text = text[:217] + "..."
                    lines.append(f"  - {text}")

        if not lines:
            return ""
        return "MEETING CONTEXT FOR THIS CALL:\n" + "\n".join(lines)

    def _extract_prior_summary_text(self, prior) -> str:
        """Pull a human-readable string out of a prior-meeting summary object.

        Feature 4 Memory stores summaries as dicts with variable shape. Try
        common fields before falling back to string coercion.
        """
        if isinstance(prior, str):
            return prior.strip()
        if not isinstance(prior, dict):
            return ""
        # Direct string fields on outer object
        for field in ("summary", "text", "content", "narrative"):
            v = prior.get(field)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # Nested object under "summary"
        inner = prior.get("summary") if isinstance(prior.get("summary"), dict) else None
        if inner:
            for field in ("text", "content", "narrative"):
                v = inner.get(field)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            bullets = inner.get("bullets") or inner.get("key_points") or []
            if isinstance(bullets, list) and bullets:
                return "; ".join(
                    str(b) for b in bullets[:3] if b
                )
        return ""

    # ── Search signal ─────────────────────────────────────────────────────────

    def _is_search_signal(self, text: str) -> bool:
        upper = text.strip().upper()
        return upper.strip("[]").strip() == "SEARCH" or "[SEARCH]" in upper

    # ── Fast Router — [PM] or [FT] classification ──────────────────────────

    async def _route(self, user_text: str, context: str = "") -> str:
        """Ultra-fast classification: [PM] or [FT]. ~100-150ms on 8b."""
        import time as _t
        t0 = _t.time()
        _debug_log("ROUTER", user_text=user_text)

        # Add recent conversation so router can see what's been discussed
        ctx_hint = ""
        if context:
            lines = [l for l in context.strip().split('\n') if l.strip()][-3:]
            if lines:
                ctx_hint = "\n\nRecent conversation:\n" + "\n".join(lines) + "\n\nNow classify the LATEST message only:"

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ROUTER_PROMPT + ctx_hint},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.0,
                max_tokens=5,
            )
            tag = response.choices[0].message.content.strip().upper()
            ms = (_t.time() - t0) * 1000
            if "[RESEARCH]" in tag or "RESEARCH" in tag:
                route = "RESEARCH"
            else:
                route = "PM"
            print(f"[Agent] ⏱ Router: [{route}] ({ms:.0f}ms)")
            return route
        except Exception as e:
            print(f"[Agent] Router failed: {e} — defaulting PM")
            return "PM"

    # ── End-of-Turn Classifier ────────────────────────────────────────────────

    async def check_end_of_turn(self, text: str, context: str = "") -> str:
        """Decide if the speaker expects a response now or is still talking.
        Returns 'RESPOND' or 'WAIT'.
        Uses conversation context for better decisions.
        Defaults to RESPOND on timeout/error (don't leave user hanging)."""
        import time as _t
        import re as _re
        t0 = _t.time()
        # Clean transcript artifacts
        clean_text = _re.sub(r'\s+', ' ', text).strip()

        # Build context block (last 3 exchanges)
        context_block = ""
        if context:
            lines = [l for l in context.strip().split('\n') if l.strip()][-3:]
            if lines:
                context_block = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

        _debug_log("EOT CHECK",
                   utterance=clean_text,
                   context_raw=context or "(EMPTY)",
                   context_block=context_block or "(EMPTY)",
                   full_prompt=EOT_PROMPT.format(text=clean_text, context_block=context_block))

        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": EOT_PROMPT.format(
                            text=clean_text,
                            context_block=context_block
                        )},
                        {"role": "user", "content": "RESPOND or WAIT?"},
                    ],
                    temperature=0.0,
                    max_tokens=3,
                ),
                timeout=0.5,
            )
            result = response.choices[0].message.content.strip().upper()
            # Parse: check for WAIT first (since RESPOND doesn't contain WAIT)
            if "WAIT" in result:
                decision = "WAIT"
            else:
                decision = "RESPOND"
            ms = (_t.time() - t0) * 1000
            emoji = "🟢" if decision == "RESPOND" else "🟡"
            print(f"[EOT] {emoji} {decision} ({ms:.0f}ms): \"{clean_text[:60]}\"")
            return decision
        except asyncio.TimeoutError:
            print(f"[EOT] ⏱ Timeout — defaulting RESPOND")
            return "RESPOND"
        except Exception as e:
            print(f"[EOT] Error: {e} — defaulting RESPOND")
            return "RESPOND"

    async def check_standup_eot(self, text: str, context: str = "", standup_phase: str = "standup") -> str:
        """Standup-specific EOT check. Understands short standup answers are complete.
        Returns 'RESPOND' or 'WAIT'. Defaults to RESPOND on timeout/error."""
        import time as _t
        import re as _re
        t0 = _t.time()
        clean_text = _re.sub(r'\s+', ' ', text).strip()

        context_block = ""
        if context:
            lines = [l for l in context.strip().split('\n') if l.strip()][-3:]
            if lines:
                context_block = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": STANDUP_EOT_PROMPT.format(
                            text=clean_text,
                            context_block=context_block,
                            standup_phase=standup_phase
                        )},
                        {"role": "user", "content": "RESPOND or WAIT?"},
                    ],
                    temperature=0.0,
                    max_tokens=3,
                ),
                timeout=0.5,
            )
            result = response.choices[0].message.content.strip().upper()
            if "WAIT" in result:
                decision = "WAIT"
            else:
                decision = "RESPOND"
            ms = (_t.time() - t0) * 1000
            emoji = "🟢" if decision == "RESPOND" else "🟡"
            print(f"[Standup-EOT] {emoji} {decision} ({ms:.0f}ms): \"{clean_text[:60]}\"")
            return decision
        except asyncio.TimeoutError:
            print(f"[Standup-EOT] ⏱ Timeout — defaulting RESPOND")
            return "RESPOND"
        except Exception as e:
            print(f"[Standup-EOT] Error: {e} — defaulting RESPOND")
            return "RESPOND"

    # ── LLM search query conversion ──────────────────────────────────────────

    async def _to_english_search_query(self, user_text: str, context: str, ticket_context: str = "") -> str:
        clean = re.sub(r'\[LANG:\w+\]\s*', '', user_text).strip()
        context_hint = ""
        if context:
            recent = context.split("\n")[-3:]
            context_hint = "\nRecent conversation:\n" + "\n".join(recent)
        if ticket_context:
            context_hint += "\n\n" + ticket_context

        _debug_log("SEARCH QUERY CONVERSION",
                   user_text=clean,
                   context_hint=context_hint or "(EMPTY)",
                   full_system_prompt=SEARCH_QUERY_PROMPT + context_hint)
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SEARCH_QUERY_PROMPT + context_hint},
                    {"role": "user", "content": clean},
                ],
                temperature=0.0,
                max_tokens=30,
            )
            query = response.choices[0].message.content.strip().strip('"\'')
            query = query.split('\n')[0].strip()
            print(f"[Agent] LLM search query: \"{clean}\" → \"{query}\"")
            return query
        except Exception as e:
            print(f"[Agent] Query conversion failed: {e}")
            return clean

    # ── Background search (runs independently, survives interrupts) ──────────

    async def search_and_summarize(self, user_text: str, context: str) -> str:
        """Full search pipeline. Returns summary text. Safe to run as background task."""
        search_query = await self._to_english_search_query(user_text, context)
        try:
            results = await self._get_web_search().search(search_query)
            if not results:
                _debug_log("SEARCH SUMMARY", search_query=search_query, results="(NONE)")
                return "Hmm, couldn't find that online right now."
            system = SEARCH_SUMMARY_PROMPT.format(search_results=results[:800])
            _debug_log("SEARCH SUMMARY",
                       search_query=search_query,
                       results_preview=results[:200],
                       user_text=user_text)
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.5,
                max_tokens=120,
            )
            answer = response.choices[0].message.content.strip()
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": answer})
            return answer
        except Exception as e:
            print(f"[Agent] search_and_summarize failed: {e}")
            return "Hmm, I couldn't look that up right now."

    # ── Core: respond (non-streaming) ────────────────────────────────────────

    async def respond(self, user_text: str) -> str:
        return await self.respond_with_context(user_text, "")

    async def respond_with_context(self, user_text: str, context: str, interrupted: bool = False) -> str:
        meeting_context = await self._build_context(user_text, context)

        if interrupted:
            system = INTERRUPT_PROMPT
            if meeting_context:
                system = INTERRUPT_PROMPT + "\n\n" + meeting_context
            return await self._llm_call(user_text, system, max_tokens=25)

        # Fast route: [PM] or [FT]?
        route = await self._route(user_text, context)

        if route == "FT":
            # Skip LLM entirely — go straight to search
            print(f"[Agent] Router → [FT] — searching: {user_text}")
            search_query = await self._to_english_search_query(user_text, context)
            try:
                results = await self._get_web_search().search(search_query)
                if not results:
                    return "Hmm, couldn't find that online right now."
                system = SEARCH_SUMMARY_PROMPT.format(search_results=results[:800])
                return await self._llm_call(user_text, system, max_tokens=120)
            except Exception as e:
                print(f"[Agent] Web search failed: {e}")
                return "Hmm, I couldn't look that up right now."

        # PM path — answer directly, context in system prompt
        system = PM_PROMPT
        if meeting_context:
            system = PM_PROMPT + "\n\n" + meeting_context
        return await self._llm_call(user_text, system, max_tokens=120)

    # ── Core: streaming (used by websocket_server) ───────────────────────────

    async def stream_sentences_to_queue(self, user_text: str, context: str, queue: asyncio.Queue,
                                          memory_context: str = ""):
        """Stream PM response sentences to queue. PM-only — [FT] handled by websocket_server.

        memory_context: optional conversation memory from past meetings with the same
        group (Feature 4 Memory). Prepended to system prompt when present.
        """
        import time as _t

        t0 = _t.time()
        meeting_context = await self._build_context(user_text, context)
        rag_ms = (_t.time() - t0) * 1000
        print(f"[Agent] ⏱ RAG context: {rag_ms:.0f}ms")

        # Store CLEAN user text in history (not context blocks!)
        self.history.append({"role": "user", "content": user_text})
        if len(self.history) > 6:
            self.history = self.history[-6:]

        # Build system prompt: memory + PM personality + meeting context
        system_prompt = PM_PROMPT
        if meeting_context:
            system_prompt = PM_PROMPT + "\n\n" + meeting_context

        # Prepend memory context if present (from prior meetings with this group)
        if memory_context:
            system_prompt = memory_context + "\n\n" + system_prompt

        total_chars = len(system_prompt) + sum(len(m["content"]) for m in self.history)
        print(f"[Agent] ⏱ Context size: {total_chars} chars (~{total_chars//4} tokens)"
              + (f" [memory: {len(memory_context)} chars]" if memory_context else ""))

        _debug_log("PM STREAM",
                   user_text=user_text,
                   convo_history_raw=context or "(EMPTY)",
                   meeting_context=meeting_context or "(NONE)",
                   llm_history=[f"{m['role']}: {m['content'][:80]}" for m in self.history],
                   system_prompt_preview=system_prompt[-300:])

        try:
            t1 = _t.time()
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}] + self.history,
                temperature=0.7,
                max_tokens=120,
                stream=True,
            )
            stream_open_ms = (_t.time() - t1) * 1000
            print(f"[Agent] ⏱ Stream opened: {stream_open_ms:.0f}ms")

            buffer = ""
            full_response = ""
            first_token_time = None
            sentence_count = 0
            async for chunk in stream:
                token = chunk.choices[0].delta.content if chunk.choices else None
                if not token:
                    continue

                if first_token_time is None:
                    first_token_time = _t.time()
                    ttft_ms = (first_token_time - t1) * 1000
                    print(f"[Agent] ⏱ First token: {ttft_ms:.0f}ms")

                buffer += token
                full_response += token

                while True:
                    indices = [buffer.find(c) for c in ".!?" if buffer.find(c) != -1]
                    if not indices:
                        break
                    idx = min(indices)
                    sentence = buffer[:idx+1].strip()
                    buffer = buffer[idx+1:].lstrip()
                    if sentence and len(sentence) > 2:  # skip punctuation-only fragments like "."
                        # Strip emote/action markers — TTS would say "(laughs)" literally
                        sentence = re.sub(r'\([^)]*\)', '', sentence).strip()
                        if not sentence or len(sentence) <= 2:
                            continue
                        sentence_count += 1
                        sent_ms = (_t.time() - t1) * 1000
                        print(f"[Agent] ⏱ Sentence {sentence_count} ready: {sent_ms:.0f}ms")
                        await queue.put(sentence)

            llm_total_ms = (_t.time() - t1) * 1000
            print(f"[Agent] ⏱ LLM total: {llm_total_ms:.0f}ms ({len(full_response.split())} words)")

            if buffer.strip() and len(buffer.strip()) > 2:
                clean_buf = re.sub(r'\([^)]*\)', '', buffer).strip()
                if clean_buf and len(clean_buf) > 2:
                    await queue.put(clean_buf)
            self.history.append({"role": "assistant", "content": full_response.strip()})

        except Exception as e:
            print(f"[Agent] LLM error: {e}")
            await queue.put("Hmm, something went wrong on my end.")
        finally:
            await queue.put(None)

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _llm_call(self, user_msg: str, system: str, max_tokens: int = 60) -> str:
        """LLM call with clean history. user_msg goes as clean text, not context dump."""
        self.history.append({"role": "user", "content": user_msg})
        if len(self.history) > 6:
            self.history = self.history[-6:]

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}] + self.history,
            temperature=0.7,
            max_tokens=max_tokens,
            stream=True,
        )

        tokens = []
        async for chunk in stream:
            t = chunk.choices[0].delta.content if chunk.choices else None
            if t:
                tokens.append(t)

        result = "".join(tokens).strip()
        self.history.append({"role": "assistant", "content": result})
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE 4: Unified research planner + SerpAPI-direct + dynamic filler
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_tech_stack(ticket_cache: list) -> str:
        """Scan ticket descriptions for common technology mentions.

        Returns a comma-separated tech stack line (e.g. "Django, React, MongoDB")
        or empty string if nothing detected. Used in SerpAPI-direct context.
        """
        if not ticket_cache:
            return ""
        all_text = " ".join(
            ((t.get("summary") or "") + " " + (t.get("description") or "")).lower()
            for t in ticket_cache
        )
        tech_markers = [
            ("Django", "django"), ("Python", "python"), ("Node.js", "node.js"),
            ("React", "react"), ("Vue", "vue"), ("Angular", "angular"),
            ("MongoDB", "mongodb"), ("PostgreSQL", "postgres"), ("MySQL", "mysql"),
            ("Redis", "redis"), ("AWS", "aws"), ("Docker", "docker"),
            ("Kubernetes", "kubernetes"), ("Salesforce", "salesforce"),
            ("Stripe", "stripe"), ("GraphQL", "graphql"),
        ]
        detected = []
        for display, search in tech_markers:
            if search in all_text:
                detected.append(display)
        return ", ".join(detected)

    def _build_project_context_from_tickets(self, tickets: list,
                                             feature_descriptions: list = None) -> str:
        """Build a feature-list prose context from tickets for SerpAPI.

        feature_descriptions: if provided (from unified research plan), used as-is.
        Otherwise, derives from ticket summaries + truncated descriptions.

        Returns a '- feature\\n- feature\\n- Tech stack: X' formatted string.
        Empty string if no tickets/features available.
        """
        lines = []
        if feature_descriptions:
            for feat in feature_descriptions[:6]:
                if feat and isinstance(feat, str):
                    lines.append(f"- {feat.strip()}")
        elif tickets:
            for t in tickets[:6]:
                summary = (t.get("summary") or "").strip()
                desc = (t.get("description") or "").strip()
                if summary and desc:
                    lines.append(f"- {summary}: {desc[:100]}")
                elif summary:
                    lines.append(f"- {summary}")
        # Always add tech stack line if detectable
        tech = self._detect_tech_stack(tickets) if tickets else ""
        if tech:
            lines.append(f"- Tech stack: {tech}")
        return "\n".join(lines) if lines else ""

    @staticmethod
    def _get_ticket_previews_for_llm(ticket_cache: list, max_tickets: int = 30) -> str:
        """Compact ticket previews for the unified research planner prompt.

        Format: 'SCRUM-12: Summary text [status]' — one per line.
        Used so the LLM can pick relevant_cached_tickets without seeing full descriptions.
        """
        if not ticket_cache:
            return "(no tickets loaded)"
        lines = []
        for t in ticket_cache[:max_tickets]:
            key = t.get("key", "?")
            summary = (t.get("summary") or "").strip()[:80]
            status = t.get("status", "?")
            lines.append(f"{key}: {summary} [{status}]")
        return "\n".join(lines)

    async def generate_unified_research_plan(self, user_text: str, context: str,
                                              ticket_cache: list,
                                              memory_header: str = "") -> Optional[dict]:
        """Single Groq call that outputs a JSON research plan.

        The plan tells the caller:
        - What type of question this is
        - Which cached tickets are relevant
        - Whether fresh Jira fetch is needed
        - What to search for (web + jira)
        - What feature descriptions to include in SerpAPI context

        memory_header: optional short memory header (~50 tokens) from past
        meetings. Helps planner understand returning-user context without
        bloating the prompt. Empty string = no memory.

        Returns parsed dict on success, None on failure (caller falls back).
        Uses llama-3.3-70b for structured output reliability.
        """
        import time as _t
        t0 = _t.time()

        # Trim conversation to last 5 lines to keep prompt compact
        conv = ""
        if context:
            conv_lines = [l for l in context.strip().split("\n") if l.strip()][-5:]
            conv = "\n".join(conv_lines) if conv_lines else "(start of call)"
        else:
            conv = "(start of call)"

        previews = self._get_ticket_previews_for_llm(ticket_cache, max_tickets=30)

        prompt = UNIFIED_RESEARCH_PROMPT.format(
            user_text=user_text,
            conversation=conv,
            ticket_previews=previews,
        )

        # Prepend memory header if present — gives planner awareness of past topics
        if memory_header:
            prompt = memory_header + "\n\n" + prompt

        try:
            resp = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",  # 70b for reliable JSON output
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,  # low for structured output
                    max_tokens=400,
                ),
                timeout=3.0,
            )
            raw = resp.choices[0].message.content.strip()
            ms = (_t.time() - t0) * 1000

            # Strip markdown fences if the LLM added them despite instructions
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw
                raw = raw.rsplit("```", 1)[0].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

            try:
                plan = json.loads(raw)
            except json.JSONDecodeError as je:
                print(f"[Agent] ⚠️  Unified plan JSON parse failed ({ms:.0f}ms): {je}")
                print(f"[Agent]    Raw: {raw[:200]}")
                return None

            # Validate required fields with safe defaults
            if not isinstance(plan, dict):
                print(f"[Agent] ⚠️  Unified plan not a dict ({ms:.0f}ms)")
                return None

            valid_types = {"feasibility", "tech_switch", "best_practices", "jira_status", "internal_org", "general"}
            plan["question_type"] = plan.get("question_type", "general")
            if plan["question_type"] not in valid_types:
                plan["question_type"] = "general"

            # Normalize list fields
            rct = plan.get("relevant_cached_tickets", [])
            plan["relevant_cached_tickets"] = [t for t in rct if isinstance(t, str)][:5] if isinstance(rct, list) else []

            scf = plan.get("serpapi_context_features", [])
            plan["serpapi_context_features"] = [f for f in scf if isinstance(f, str)][:6] if isinstance(scf, list) else []

            plan["needs_fresh_jira"] = bool(plan.get("needs_fresh_jira", False))
            plan["jira_search_terms"] = (plan.get("jira_search_terms") or "").strip()
            plan["web_search_query"] = (plan.get("web_search_query") or "").strip()

            print(f"[Agent] ⏱ Unified plan ({ms:.0f}ms): type={plan['question_type']}, "
                  f"cached_refs={len(plan['relevant_cached_tickets'])}, "
                  f"fresh_jira={plan['needs_fresh_jira']}, "
                  f"features={len(plan['serpapi_context_features'])}")
            return plan

        except asyncio.TimeoutError:
            ms = (_t.time() - t0) * 1000
            print(f"[Agent] ⏱ Unified plan TIMEOUT ({ms:.0f}ms) — falling back")
            return None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[Agent] ⚠️  Unified plan error: {type(e).__name__}: {e}")
            return None

    async def serpapi_direct_research(self, user_text: str, project_context: str,
                                        conversation: str = "",
                                        length: str = "2-3 sentences") -> Optional[str]:
        """Direct SerpAPI call with persona + project context + TTS hint.

        Returns cleaned voice-ready text or None on failure. Uses search_raw()
        to bypass query trimming (the persona prompt is long by design).

        Caller is responsible for falling back to Azure synthesis on None.
        """
        import time as _t
        t0 = _t.time()

        # Build conversation hint only if meaningful context exists
        conversation_hint = ""
        if conversation:
            conv_lines = [l for l in conversation.strip().split("\n") if l.strip()][-3:]
            if conv_lines:
                conversation_hint = "Conversation so far:\n" + "\n".join(conv_lines) + "\n\n"

        # Build the rich persona query
        if not project_context:
            project_context = "- (no specific project context available)"

        rich_query = SERPAPI_DIRECT_PROMPT.format(
            question=user_text,
            project_context=project_context,
            conversation_hint=conversation_hint,
            length=length,
        )

        try:
            web = self._get_web_search()
            raw_response = await web.search_raw(rich_query, max_length=2000)
            ms = (_t.time() - t0) * 1000

            if not raw_response:
                print(f"[Agent] SerpAPI-direct returned nothing ({ms:.0f}ms)")
                return None

            cleaned = self._clean_serpapi_for_tts(raw_response)

            if not cleaned or len(cleaned) < 30:
                print(f"[Agent] SerpAPI-direct too short after cleanup ({ms:.0f}ms): \"{cleaned[:60]}\"")
                return None

            print(f"[Agent] ⏱ SerpAPI-direct ({ms:.0f}ms): {len(cleaned)} chars")
            return cleaned

        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[Agent] ⚠️  SerpAPI-direct error: {type(e).__name__}: {e}")
            return None

    @staticmethod
    def _clean_serpapi_for_tts(text: str) -> str:
        """Strip residual markdown/citations from SerpAPI output before TTS.

        Handles patterns that slip through even with the TTS hint:
        - 'Microsoft Azure +2' style inline citations
        - '**bold markers**'
        - '* bullet lists'
        - URL references
        - 'N sites' source footer
        """
        if not text:
            return ""
        # Source footer like '\n8 sites\n' or '\n5 sites'
        text = re.split(r'\n\s*\d+\s*sites?\b', text, maxsplit=1)[0]
        # Inline citations: 'Microsoft Azure +2', 'Some Source +3'
        text = re.sub(r'\b[A-Z][A-Za-z .]+?\s+\+\d+\b', '', text)
        # Bold markdown
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        # Italic markdown (single asterisk pairs)
        text = re.sub(r'(?<!\*)\*(?!\*)([^*\n]+?)\*(?!\*)', r'\1', text)
        # Bullet points at line starts
        text = re.sub(r'^\s*[\*\-•]\s+', '', text, flags=re.MULTILINE)
        # Headers
        text = re.sub(r'^\s*#{1,6}\s+', '', text, flags=re.MULTILINE)
        # URLs
        text = re.sub(r'https?://\S+', '', text)
        # Collapse multiple newlines to single space
        text = re.sub(r'\n+', ' ', text)
        # Collapse whitespace runs
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # ── Dynamic Filler Generation ─────────────────────────────────────────────

    async def generate_dynamic_filler(self, question: str) -> Optional[str]:
        """Generate a contextual one-sentence filler using Groq 8b.

        Runs in parallel with trigger/route decisions. 400ms timeout ensures
        pipeline isn't stalled. Returns None on any failure — caller falls back
        to random choice from hardcoded FILLERS.
        """
        import time as _t
        t0 = _t.time()
        try:
            resp = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": FILLER_PROMPT.format(question=question)},
                    ],
                    temperature=0.8,
                    max_tokens=35,
                ),
                timeout=0.4,
            )
            text = resp.choices[0].message.content.strip()
            text = text.strip('"').strip("'").strip()
            ms = (_t.time() - t0) * 1000
            if self._is_valid_filler(text):
                print(f"[Agent] ⏱ Dynamic filler ({ms:.0f}ms): \"{text}\"")
                return text
            else:
                print(f"[Agent] ⚠️  Dynamic filler rejected ({ms:.0f}ms): \"{text[:60]}\"")
                return None
        except asyncio.TimeoutError:
            print(f"[Agent] ⏱ Dynamic filler timeout (>400ms)")
            return None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[Agent] ⚠️  Dynamic filler error: {type(e).__name__}: {e}")
            return None

    @staticmethod
    def _is_valid_filler(text: str) -> bool:
        """Validate dynamic filler output before using for TTS."""
        if not text or not text.strip():
            return False
        text = text.strip()
        if len(text) > 150 or len(text) < 10:
            return False
        words = text.split()
        if len(words) < 3 or len(words) > 30:
            return False
        lower = text.lower()
        bad_phrases = [
            "as an ai", "i'm an ai", "i am an ai", "language model",
            "i cannot", "i don't have the ability", "here's my response",
            "here is my response", "sure, here", "filler:", "response:",
            "acknowledgment:", "sam:", "user:", "assistant:",
        ]
        for phrase in bad_phrases:
            if phrase in lower:
                return False
        if "**" in text or text.startswith(("*", "-", "#", "`")):
            return False
        checking_signals = [
            "let me", "one sec", "one second", "hold on", "give me",
            "checking", "check", "look", "pull", "see", "think",
            "find", "grab", "digging", "dig", "moment",
        ]
        if not any(sig in lower for sig in checking_signals):
            return False
        return True

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    def reset(self):
        self.history.clear()
        self.rag.clear()

    async def stream_research_to_queue(self, user_text: str, jira_context: str,
                                        related_tickets: str, web_results: str,
                                        jira_action: str, conversation: str,
                                        azure_extractor, queue: asyncio.Queue,
                                        memory_context: str = ""):
        """Stream RESEARCH response with Jira + web data.

        PRIMARY:  Groq openai/gpt-oss-20b   (~947 t/s, ~500-900ms first token)
        FALLBACK: Azure gpt-4o-mini         (~80 t/s, ~2500-3000ms first token)

        Toggle via env var RESEARCH_PROVIDER:
            "groq"   (default) — use Groq gpt-oss-20b for speed
            "azure"            — force Azure gpt-4o-mini for max quality
            "auto"             — try Groq first, fall back to Azure on error

        memory_context: optional conversation memory from past meetings.
        Prepended to system prompt when present.
        """
        import time as _t
        import httpx
        import os as _os

        system = RESEARCH_PROMPT.format(
            jira_context=jira_context or "(no project context loaded)",
            related_tickets=related_tickets or "(no related tickets found)",
            jira_action=jira_action or "(no action taken)",
            web_results=web_results or "(no web results)",
            conversation=conversation or "(start of call)",
        )

        # Prepend memory context if present (from prior meetings with this group)
        if memory_context:
            system = memory_context + "\n\n" + system

        self.history.append({"role": "user", "content": user_text})
        if len(self.history) > 6:
            self.history = self.history[-6:]

        # Provider selection
        provider = _os.environ.get("RESEARCH_PROVIDER", "groq").lower().strip()
        azure_available = bool(azure_extractor and azure_extractor.enabled)

        # If env says azure but azure is disabled, fall through to groq
        if provider == "azure" and not azure_available:
            print("[Agent] ⚠️  RESEARCH_PROVIDER=azure but Azure unavailable — using Groq")
            provider = "groq"

        # ── PRIMARY PATH: Groq gpt-oss-20b ─────────────────────────────
        if provider in ("groq", "auto"):
            try:
                await self._stream_research_groq(
                    system, user_text, queue
                )
                return  # success — queue.put(None) already sent
            except Exception as e:
                print(f"[Agent] ⚠️  Groq research failed: {type(e).__name__}: {e}")
                if provider == "auto" and azure_available:
                    print("[Agent] 🔄 Falling back to Azure gpt-4o-mini")
                    # Fall through to Azure block below
                else:
                    await queue.put("Sorry, I ran into an issue researching that.")
                    await queue.put(None)
                    return

        # ── FALLBACK PATH: Azure gpt-4o-mini ───────────────────────────
        if not azure_available:
            # No Azure either — last-resort Groq 8b synchronous call
            print("[Agent] ⚠️  Azure unavailable — last-resort Groq 8b")
            try:
                stream = await self.client.chat.completions.create(
                    model=self.model,  # llama-3.1-8b-instant
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user_text}],
                    temperature=0.7, max_tokens=200, stream=True,
                )
                buffer = ""
                full = ""
                async for chunk in stream:
                    token = chunk.choices[0].delta.content if chunk.choices else None
                    if not token:
                        continue
                    buffer += token
                    full += token
                    while ". " in buffer or "? " in buffer or "! " in buffer:
                        for sep in [". ", "? ", "! "]:
                            idx = buffer.find(sep)
                            if idx != -1:
                                sentence = buffer[:idx + 1].strip()
                                buffer = buffer[idx + 2:]
                                if sentence:
                                    await queue.put(sentence)
                                break
                if buffer.strip():
                    await queue.put(buffer.strip())
                self.history.append({"role": "assistant", "content": full})
            except Exception as e:
                print(f"[Agent] ⚠️  Last-resort Groq 8b failed: {e}")
                await queue.put("Sorry, I couldn't process that right now.")
            await queue.put(None)
            return

        # Azure 4o-mini streaming
        url = f"{azure_extractor.endpoint}/openai/deployments/{azure_extractor.deployment}/chat/completions?api-version={azure_extractor.api_version}"

        t0 = _t.time()
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                async with client.stream("POST", url,
                    headers={"api-key": azure_extractor.api_key, "Content-Type": "application/json"},
                    json={
                        "messages": [{"role": "system", "content": system},
                                     {"role": "user", "content": user_text}],
                        "temperature": 0.7,
                        "max_tokens": 250,
                        "stream": True,
                    }) as response:

                    response.raise_for_status()

                    buffer = ""
                    full_response = ""
                    first_token = False
                    sentence_count = 0

                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            token = delta.get("content", "")
                            if not token:
                                continue

                            if not first_token:
                                first_token = True
                                print(f"[Agent] ⏱ RESEARCH first token: {(_t.time()-t0)*1000:.0f}ms")

                            buffer += token
                            full_response += token

                            while ". " in buffer or "? " in buffer or "! " in buffer:
                                for sep in [". ", "? ", "! "]:
                                    idx = buffer.find(sep)
                                    if idx != -1:
                                        sentence = buffer[:idx + 1].strip()
                                        buffer = buffer[idx + 2:]
                                        if sentence:
                                            sentence_count += 1
                                            print(f"[Agent] ⏱ RESEARCH sentence {sentence_count}: {(_t.time()-t0)*1000:.0f}ms")
                                            await queue.put(sentence)
                                        break
                        except (json.JSONDecodeError, IndexError, KeyError):
                            continue

                    if buffer.strip():
                        sentence_count += 1
                        await queue.put(buffer.strip())

                    total_ms = (_t.time() - t0) * 1000
                    words = len(full_response.split())
                    print(f"[Agent] ⏱ RESEARCH total: {total_ms:.0f}ms ({words} words, {sentence_count} sentences)")
                    self.history.append({"role": "assistant", "content": full_response})

        except Exception as e:
            print(f"[Agent] ❌ Azure stream failed: {e}")
            await queue.put("Sorry, I ran into an issue researching that.")
            try:
                fallback = await self._llm_call(user_text, system, max_tokens=120)
                await queue.put(fallback)
            except Exception:
                pass

        await queue.put(None)

    async def _stream_research_groq(self, system: str, user_text: str,
                                     queue: asyncio.Queue) -> None:
        """Stream research response from Groq openai/gpt-oss-20b.

        Uses reasoning_effort='none' to skip thinking tokens — we want fast
        synthesis, not chain-of-thought. Groq's LPU hardware + MoE arch
        gives ~500-900ms first token vs ~2900ms for Azure gpt-4o-mini.

        Prompt caching: if this system prompt shares a prefix with a recent
        request (usually yes — Jira context is mostly identical), Groq
        caches that prefix and returns first token even faster with cached
        tokens not counting toward rate limits.

        Raises on failure so the caller can fall back to Azure.
        """
        import time as _t

        t0 = _t.time()

        # Build kwargs — put reasoning_effort in extra_body so the OpenAI
        # SDK forwards it to Groq. 'none' means no thinking tokens.
        kwargs = {
            "model": "openai/gpt-oss-20b",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.7,
            "max_tokens": 250,
            "stream": True,
            "extra_body": {"reasoning_effort": "low"},  # 'low' minimizes thinking; 'none' may not be accepted
        }

        stream = await self.client.chat.completions.create(**kwargs)

        buffer = ""
        full_response = ""
        first_token = False
        sentence_count = 0

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            token = getattr(delta, "content", None) or ""
            if not token:
                continue

            if not first_token:
                first_token = True
                print(f"[Agent] ⏱ RESEARCH first token: {(_t.time()-t0)*1000:.0f}ms (groq/gpt-oss-20b)")

            buffer += token
            full_response += token

            while ". " in buffer or "? " in buffer or "! " in buffer:
                for sep in [". ", "? ", "! "]:
                    idx = buffer.find(sep)
                    if idx != -1:
                        sentence = buffer[:idx + 1].strip()
                        buffer = buffer[idx + 2:]
                        if sentence:
                            sentence_count += 1
                            print(f"[Agent] ⏱ RESEARCH sentence {sentence_count}: {(_t.time()-t0)*1000:.0f}ms")
                            await queue.put(sentence)
                        break

        if buffer.strip():
            sentence_count += 1
            await queue.put(buffer.strip())

        total_ms = (_t.time() - t0) * 1000
        words = len(full_response.split())
        print(f"[Agent] ⏱ RESEARCH total: {total_ms:.0f}ms ({words} words, "
              f"{sentence_count} sentences) [groq/gpt-oss-20b]")
        self.history.append({"role": "assistant", "content": full_response})

        await queue.put(None)