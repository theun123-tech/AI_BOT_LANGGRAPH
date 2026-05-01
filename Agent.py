# """
# Agent.py — Groq Llama 3.3 70B + In-Memory RAG

# Memory architecture:
#   1. RAG store: Every exchange is embedded (Azure OpenAI text-embedding-3-small)
#      and stored in-memory. On query, cosine similarity finds relevant past exchanges.
#      Catches semantic matches like "money" → "budget" that keyword search misses.
#   2. Meeting log: Full text of every exchange (fallback + debugging)
#   3. Recent history: Last 10 LLM turns for conversation flow

# Embedding happens async in background — never blocks the response pipeline.
# If embeddings fail, falls back to keyword search automatically.
# """

# import os
# import asyncio
# import re
# import time
# import numpy as np
# from openai import AsyncOpenAI
# from typing import List, Optional


# # ── Debug logger — writes all prompt inputs to file ──────────────────────────
# DEBUG_PROMPTS_FILE = "debug_prompts.txt"

# def _debug_log(label: str, **kwargs):
#     """Append a debug entry to debug_prompts.txt with timestamp and all variables."""
#     try:
#         ts = time.strftime("%H:%M:%S")
#         with open(DEBUG_PROMPTS_FILE, "a", encoding="utf-8") as f:
#             f.write(f"\n{'='*80}\n")
#             f.write(f"[{ts}] {label}\n")
#             f.write(f"{'='*80}\n")
#             for key, val in kwargs.items():
#                 val_str = str(val) if val else "(EMPTY)"
#                 f.write(f"  {key}:\n    {val_str}\n")
#             f.write(f"\n")
#     except Exception:
#         pass  # never crash on debug logging


# # ══════════════════════════════════════════════════════════════════════════════
# # IN-MEMORY RAG STORE
# # ══════════════════════════════════════════════════════════════════════════════

# class MeetingRAG:
#     """In-memory vector store for meeting transcripts.
#     Uses fastembed (free, local, ~200MB). No API key needed.
#     Model: BAAI/bge-small-en-v1.5 — 130MB, 384-dim, fast on CPU.
#     """

#     def __init__(self):
#         self._entries: list[dict] = []
#         self._embed_queue: asyncio.Queue = asyncio.Queue()
#         self._embed_task: Optional[asyncio.Task] = None
#         self._model = None
#         self._ready = False

#         try:
#             from fastembed import TextEmbedding
#             self._model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
#             self._ready = True
#             print("[RAG] Local embeddings ready (BAAI/bge-small-en-v1.5, fastembed)")
#         except ImportError:
#             print("[RAG] ⚠️  fastembed not installed — keyword fallback only")
#         except Exception as e:
#             print(f"[RAG] ⚠️  Model load failed: {e} — keyword fallback only")

#     def start_background_embedder(self):
#         if self._ready and not self._embed_task:
#             self._embed_task = asyncio.create_task(self._embedding_worker())

#     async def _embedding_worker(self):
#         loop = asyncio.get_event_loop()
#         while True:
#             try:
#                 entry = await self._embed_queue.get()
#                 vector = await loop.run_in_executor(
#                     None, self._embed_sync, entry["text"]
#                 )
#                 if vector is not None:
#                     entry["vector"] = vector
#                     self._entries.append(entry)
#             except asyncio.CancelledError:
#                 break
#             except Exception as e:
#                 print(f"[RAG] Embed worker error: {e}")

#     def _embed_sync(self, text: str) -> Optional[np.ndarray]:
#         if not self._model:
#             return None
#         try:
#             # fastembed returns a generator — get first result
#             vectors = list(self._model.embed([text]))
#             return np.array(vectors[0], dtype=np.float32)
#         except Exception as e:
#             print(f"[RAG] Embedding failed: {e}")
#             return None

#     def add(self, speaker: str, text: str):
#         """Queue an exchange for embedding (non-blocking)."""
#         entry = {
#             "text": f"{speaker}: {text}",
#             "speaker": speaker,
#             "time": time.time(),
#             "vector": None,
#         }
#         if self._ready:
#             try:
#                 self._embed_queue.put_nowait(entry)
#             except Exception:
#                 pass
#         else:
#             self._entries.append(entry)

#     async def search(self, query: str, top_k: int = 5) -> List[str]:
#         """Find relevant past exchanges by cosine similarity.
#         Falls back to keyword matching if embeddings unavailable.
#         """
#         if not self._entries:
#             return []

#         # Vector search
#         if self._ready and self._model:
#             loop = asyncio.get_event_loop()
#             query_vector = await loop.run_in_executor(
#                 None, self._embed_sync, query
#             )

#             if query_vector is not None:
#                 scored = []
#                 for entry in self._entries:
#                     if entry["vector"] is not None:
#                         sim = self._cosine_sim(query_vector, entry["vector"])
#                         scored.append((sim, entry["text"]))

#                 if scored:
#                     scored.sort(key=lambda x: x[0], reverse=True)
#                     results = [text for sim, text in scored[:top_k] if sim > 0.3]
#                     if results:
#                         print(f"[RAG] Vector search: {len(results)} hits for \"{query[:50]}\"")
#                         return results

#         # Fallback: keyword search
#         return self._keyword_search(query, top_k)

#     def _keyword_search(self, query: str, top_k: int = 5, exclude_text: str = "") -> List[str]:
#         stop = {"the", "a", "an", "is", "are", "was", "were", "what", "who",
#                 "how", "when", "where", "why", "did", "do", "does", "can",
#                 "could", "would", "should", "we", "i", "you", "they", "it",
#                 "about", "tell", "me", "something", "discuss", "talked", "sam"}
#         query_words = {w for w in query.lower().split() if w not in stop and len(w) > 2}
#         if not query_words:
#             return []
#         # Normalize exclude text for comparison
#         exclude_lower = exclude_text.lower().strip() if exclude_text else ""
#         scored = []
#         for entry in self._entries:
#             entry_lower = entry["text"].lower().strip()
#             # Skip if this entry IS the current utterance (prevent echo)
#             if exclude_lower and (exclude_lower in entry_lower or entry_lower.endswith(exclude_lower)):
#                 continue
#             hits = sum(1 for w in query_words if w in entry_lower)
#             if hits > 0:
#                 scored.append((hits, entry["text"]))
#         scored.sort(key=lambda x: x[0], reverse=True)
#         results = [text for _, text in scored[:top_k]]
#         if results:
#             print(f"[RAG] Keyword fallback: {len(results)} hits for \"{query[:50]}\"")
#         return results

#     @staticmethod
#     def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
#         dot = np.dot(a, b)
#         norm = np.linalg.norm(a) * np.linalg.norm(b)
#         return float(dot / norm) if norm > 0 else 0.0

#     @property
#     def size(self) -> int:
#         return len(self._entries)

#     def clear(self):
#         self._entries.clear()


# # ══════════════════════════════════════════════════════════════════════════════
# # PROMPTS
# # ══════════════════════════════════════════════════════════════════════════════

# # ── FAST ROUTER — classifies before main LLM ────────────────────────────────
# ROUTER_PROMPT = """Classify this message. Reply with ONLY one tag:
# [PM] — greetings, small talk, personal questions, jokes, opinions, PM work topics (agenda, blockers, sprint status, timeline, team updates), conversation about ongoing work, AND any question about what was said earlier in THIS conversation (memory, recall, "what did I ask", "repeat that", "come again", "previous question").
# [FT] — ANY question needing verifiable facts from the OUTSIDE WORLD: who is someone (CEO, founder, president), company info (rates, pricing, services), real-world knowledge (people, places, events, news, weather, science, history, dates, numbers, statistics).

# IMPORTANT:
# - If the user asks WHO someone is (CEO, founder, manager), or asks about PRICES/RATES/COSTS — always [FT], even if they say "our company".
# - If the user asks about what THEY said earlier, what was discussed, or asks you to repeat — always [PM], this is conversation memory not web search.

# Examples:
# "Hi Sam, how are you?" → [PM]
# "What's on the agenda?" → [PM]
# "Who is the CEO of our company?" → [FT]
# "What are the hourly rates?" → [FT]
# "Who is the prime minister of India?" → [FT]
# "Tell me about yourself" → [PM]
# "What's the founder's name?" → [FT]
# "How's the sprint going?" → [PM]
# "What services does our company offer?" → [FT]
# "What was my previous question?" → [PM]
# "Can you repeat that?" → [PM]
# "What did I just ask you?" → [PM]
# "Come again?" → [PM]
# "Describe yourself in three words" → [PM]

# ONLY reply with [PM] or [FT]. Nothing else."""

# # ── PM PROMPT — only handles personality + PM answers ────────────────────────
# PM_PROMPT = """You are Sam, a senior PM at AnavClouds Software Solutions. You're on a live voice call in a meeting.

# HOW YOU TALK:
# - Like a real person in a meeting, not a chatbot. Use "yeah", "honestly", "look", "so basically".
# - React to what the person ACTUALLY said before answering. Show you're listening.
# - Use their name naturally (not every time — that's weird).
# - Contractions always. Never say "I am" when you can say "I'm".
# - Throw in light humor when it fits. You're the fun PM, not the boring one.
# - If they ask for a joke, deliver the FULL joke with setup AND punchline in one response.

# YOUR BACKGROUND:
# - Senior PM at AnavClouds (Salesforce + AI company). You handle sprints, budgets, timelines, CRM rollouts.
# - Confident, warm, slightly sarcastic. You deflect weird questions with humor.
# - You remember what was said in the meeting (see MEETING MEMORY below).

# RULES:
# - 1-2 sentences. Max 20 words per sentence.
# - If user asks to REPEAT something: rephrase your last answer differently.
# - Say something NEW each time — don't repeat yourself.
# - No markdown, no lists, no asterisks, no bullet points, no parenthetical actions like (laughs) or (sighs) — this is voice output, not a script.

# MEETING MEMORY: Use this to answer about past discussions if provided."""

# SEARCH_SUMMARY_PROMPT = """You are Sam, a PM on a live voice call. You just searched the web for the user.

# Search results:
# {search_results}

# Rules:
# - Skip any filler — the user already heard one.
# - Jump straight to the answer.
# - 2-3 SHORT sentences max. Keep each under 18 words.
# - Sound like you're telling a coworker, not reading a report.
# - If the results don't have the answer, just say so naturally — don't apologize."""

# INTERRUPT_PROMPT = """You are Sam, a witty senior PM. You were interrupted.
# Reply in ONE sentence — 15 words max. Be quick, natural.
# Start with: "Oh," / "Right," / "Sure," / "Got it," — then pivot to their question."""

# SEARCH_QUERY_PROMPT = """Convert the user's message into a DESCRIPTIVE Google search query (8-15 words).
# Make the query specific and detailed so Google returns the best results.
# Replace 'our company'/'my company'/'our'/'we' with 'AnavClouds Software Solutions'.
# If the user does NOT mention the company, do NOT add AnavClouds.
# Remove 'Sam' and filler words.
# If the message is about conversation memory (what did I ask, repeat, previous question) — output: SKIP
# For multi-part questions, include all parts in a single descriptive query.

# Examples:
# "What are the hourly rates?" → "What is the hourly rate and pricing of AnavClouds Software Solutions"
# "Who is the CEO?" → "Who is the CEO and founder of AnavClouds Software Solutions company"
# "Who is the prime minister of India?" → "Who is the current prime minister of India 2025"
# "What's the weather in Delhi?" → "What is the weather forecast in Delhi India today"
# "Tell me about the company" → "AnavClouds Software Solutions company overview services and products"
# "What is the density of India and who founded your company?" → "Population density of India and founder of AnavClouds Software Solutions"
# "How many employees?" → "How many employees work at AnavClouds Software Solutions team size"
# "What was my previous question?" → SKIP

# Output ONLY the search query. No quotes, no explanation."""

# FILLERS = [
#     "Hmm, let me look that up real quick.",
#     "Right, give me one sec to check on that.",
#     "Uh, good question — let me pull that up.",
#     "Yeah, hold on, let me find that for you.",
#     "Well, let me check on that real quick.",
# ]


# # ── End-of-Turn Classifier (RESPOND vs WAIT) ────────────────────────────────
# EOT_PROMPT = """You are Sam, an AI participant in a live meeting. Someone is speaking to you. Based on the conversation, decide:

# Should you RESPOND now, or WAIT for them to continue?

# {context_block}Current utterance: "{text}"

# RESPOND if ANY of these are true:
# - The utterance contains a question (even rhetorical like "Right?" or "You know?")
# - The utterance is a request or command ("Tell me...", "Can you...")
# - The utterance is a short reaction or statement directed at you
# - The speaker seems to be done talking and waiting for your input
# - You are unsure — RESPOND is always safer than making someone wait

# WAIT only if ALL of these are true:
# - The utterance is clearly mid-sentence (grammatically incomplete)
# - OR the speaker is obviously setting up a longer explanation and hasn't asked anything yet
# - AND there is no question, request, or floor-handoff signal

# Reply with one word: RESPOND or WAIT"""


# # ══════════════════════════════════════════════════════════════════════════════
# # PM AGENT
# # ══════════════════════════════════════════════════════════════════════════════

# class PMAgent:
#     def __init__(self):
#         self.client = AsyncOpenAI(
#             api_key=os.environ["GROQ_API_KEY"],
#             base_url="https://api.groq.com/openai/v1",
#         )
#         self.model = "llama-3.1-8b-instant"

#         # Recent LLM history — last 10 turns
#         self.history: list[dict] = []

#         # RAG store — embeds + retrieves meeting exchanges
#         self.rag = MeetingRAG()

#     def start(self):
#         """Call once after event loop is running to start background embedder + warmup."""
#         self.rag.start_background_embedder()
#         asyncio.create_task(self._warmup())

#     async def _warmup(self):
#         """Pre-establish TCP connection to Groq — saves ~300ms on first real call."""
#         try:
#             await self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[{"role": "user", "content": "hi"}],
#                 max_tokens=1,
#             )
#             print("[Agent] ✅ Groq connection warmed up")
#         except Exception:
#             pass

#     def _get_web_search(self):
#         if not hasattr(self, '_web_search') or self._web_search is None:
#             from WebSearch import WebSearch
#             self._web_search = WebSearch()
#         return self._web_search

#     # ── Memory ────────────────────────────────────────────────────────────────

#     def log_exchange(self, speaker: str, text: str):
#         """Store an exchange in RAG. Called by websocket_server for every transcript."""
#         self.rag.add(speaker, text)

#     async def _build_context(self, user_text: str, context: str) -> str:
#         """Build meeting context for the system prompt (NOT the user message).
#         Returns a string to append to the system prompt with relevant memory + recent convo.
#         Filters current utterance from RAG to prevent echo."""
#         parts = []

#         # Fast keyword search — exclude current utterance to prevent echo
#         rag_results = self.rag._keyword_search(user_text, top_k=2, exclude_text=user_text)
#         if rag_results:
#             parts.append("MEETING MEMORY (relevant past exchanges):\n" + "\n".join(rag_results))

#         # Recent conversation — last 4 lines for flow
#         if context:
#             recent = "\n".join(context.split("\n")[-4:])
#             parts.append(f"RECENT CONVERSATION:\n{recent}")

#         if not parts:
#             return ""

#         full_context = "\n\n".join(parts)

#         # _debug_log("BUILD CONTEXT (system prompt appendix)",
#         #            user_text=user_text,
#         #            convo_history_raw=context or "(EMPTY)",
#         #            rag_results=rag_results or "(NONE)",
#         #            built_context=full_context)

#         return full_context

#     # ── Search signal ─────────────────────────────────────────────────────────

#     def _is_search_signal(self, text: str) -> bool:
#         upper = text.strip().upper()
#         return upper.strip("[]").strip() == "SEARCH" or "[SEARCH]" in upper

#     # ── Fast Router — [PM] or [FT] classification ──────────────────────────

#     async def _route(self, user_text: str) -> str:
#         """Ultra-fast classification: [PM] or [FT]. ~100-150ms on 8b."""
#         import time as _t
#         t0 = _t.time()
#         # _debug_log("ROUTER", user_text=user_text)
#         try:
#             response = await self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": ROUTER_PROMPT},
#                     {"role": "user", "content": user_text},
#                 ],
#                 temperature=0.0,
#                 max_tokens=5,
#             )
#             tag = response.choices[0].message.content.strip().upper()
#             ms = (_t.time() - t0) * 1000
#             route = "FT" if "[FT]" in tag or "FT" in tag else "PM"
#             print(f"[Agent] ⏱ Router: [{route}] ({ms:.0f}ms)")
#             return route
#         except Exception as e:
#             print(f"[Agent] Router failed: {e} — defaulting PM")
#             return "PM"

#     # ── End-of-Turn Classifier ────────────────────────────────────────────────

#     async def check_end_of_turn(self, text: str, context: str = "") -> str:
#         """Decide if the speaker expects a response now or is still talking.
#         Returns 'RESPOND' or 'WAIT'.
#         Uses conversation context for better decisions.
#         Defaults to RESPOND on timeout/error (don't leave user hanging)."""
#         import time as _t
#         import re as _re
#         t0 = _t.time()
#         # Clean transcript artifacts
#         clean_text = _re.sub(r'\s+', ' ', text).strip()

#         # Build context block (last 3 exchanges)
#         context_block = ""
#         if context:
#             lines = [l for l in context.strip().split('\n') if l.strip()][-3:]
#             if lines:
#                 context_block = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

#         # _debug_log("EOT CHECK",
#         #            utterance=clean_text,
#         #            context_raw=context or "(EMPTY)",
#         #            context_block=context_block or "(EMPTY)",
#         #            full_prompt=EOT_PROMPT.format(text=clean_text, context_block=context_block))

#         try:
#             response = await asyncio.wait_for(
#                 self.client.chat.completions.create(
#                     model=self.model,
#                     messages=[
#                         {"role": "system", "content": EOT_PROMPT.format(
#                             text=clean_text,
#                             context_block=context_block
#                         )},
#                         {"role": "user", "content": "RESPOND or WAIT?"},
#                     ],
#                     temperature=0.0,
#                     max_tokens=3,
#                 ),
#                 timeout=0.5,
#             )
#             result = response.choices[0].message.content.strip().upper()
#             # Parse: check for WAIT first (since RESPOND doesn't contain WAIT)
#             if "WAIT" in result:
#                 decision = "WAIT"
#             else:
#                 decision = "RESPOND"
#             ms = (_t.time() - t0) * 1000
#             emoji = "🟢" if decision == "RESPOND" else "🟡"
#             print(f"[EOT] {emoji} {decision} ({ms:.0f}ms): \"{clean_text[:60]}\"")
#             return decision
#         except asyncio.TimeoutError:
#             print(f"[EOT] ⏱ Timeout — defaulting RESPOND")
#             return "RESPOND"
#         except Exception as e:
#             print(f"[EOT] Error: {e} — defaulting RESPOND")
#             return "RESPOND"

#     # ── LLM search query conversion ──────────────────────────────────────────

#     async def _to_english_search_query(self, user_text: str, context: str) -> str:
#         clean = re.sub(r'\[LANG:\w+\]\s*', '', user_text).strip()
#         context_hint = ""
#         if context:
#             recent = context.split("\n")[-3:]
#             context_hint = "\nRecent conversation:\n" + "\n".join(recent)

#         # _debug_log("SEARCH QUERY CONVERSION",
#         #            user_text=clean,
#         #            context_hint=context_hint or "(EMPTY)",
#         #            full_system_prompt=SEARCH_QUERY_PROMPT + context_hint)
#         try:
#             response = await self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": SEARCH_QUERY_PROMPT + context_hint},
#                     {"role": "user", "content": clean},
#                 ],
#                 temperature=0.0,
#                 max_tokens=30,
#             )
#             query = response.choices[0].message.content.strip().strip('"\'')
#             # Safety: only use first line (LLM sometimes outputs multiple)
#             query = query.split('\n')[0].strip()
#             print(f"[Agent] LLM search query: \"{clean}\" → \"{query}\"")
#             return query
#         except Exception as e:
#             print(f"[Agent] Query conversion failed: {e}")
#             return clean

#     # ── Background search (runs independently, survives interrupts) ──────────

#     async def search_and_summarize(self, user_text: str, context: str) -> str:
#         """Full search pipeline. Returns summary text. Safe to run as background task."""
#         search_query = await self._to_english_search_query(user_text, context)
#         try:
#             results = await self._get_web_search().search(search_query)
#             if not results:
#                 # _debug_log("SEARCH SUMMARY", search_query=search_query, results="(NONE)")
#                 return "Hmm, couldn't find that online right now."
#             system = SEARCH_SUMMARY_PROMPT.format(search_results=results[:800])
#             # _debug_log("SEARCH SUMMARY",
#             #            search_query=search_query,
#             #            results_preview=results[:200],
#             #            user_text=user_text)
#             response = await self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": system},
#                     {"role": "user", "content": user_text},
#                 ],
#                 temperature=0.5,
#                 max_tokens=120,
#             )
#             answer = response.choices[0].message.content.strip()
#             self.history.append({"role": "user", "content": user_text})
#             self.history.append({"role": "assistant", "content": answer})
#             return answer
#         except Exception as e:
#             print(f"[Agent] search_and_summarize failed: {e}")
#             return "Hmm, I couldn't look that up right now."

#     # ── Core: respond (non-streaming) ────────────────────────────────────────

#     async def respond(self, user_text: str) -> str:
#         return await self.respond_with_context(user_text, "")

#     async def respond_with_context(self, user_text: str, context: str, interrupted: bool = False) -> str:
#         meeting_context = await self._build_context(user_text, context)

#         if interrupted:
#             system = INTERRUPT_PROMPT
#             if meeting_context:
#                 system = INTERRUPT_PROMPT + "\n\n" + meeting_context
#             return await self._llm_call(user_text, system, max_tokens=25)

#         # Fast route: [PM] or [FT]?
#         route = await self._route(user_text)

#         if route == "FT":
#             # Skip LLM entirely — go straight to search
#             print(f"[Agent] Router → [FT] — searching: {user_text}")
#             search_query = await self._to_english_search_query(user_text, context)
#             try:
#                 results = await self._get_web_search().search(search_query)
#                 if not results:
#                     return "Hmm, couldn't find that online right now."
#                 system = SEARCH_SUMMARY_PROMPT.format(search_results=results[:800])
#                 return await self._llm_call(user_text, system, max_tokens=120)
#             except Exception as e:
#                 print(f"[Agent] Web search failed: {e}")
#                 return "Hmm, I couldn't look that up right now."

#         # PM path — answer directly, context in system prompt
#         system = PM_PROMPT
#         if meeting_context:
#             system = PM_PROMPT + "\n\n" + meeting_context
#         return await self._llm_call(user_text, system, max_tokens=120)

#     # ── Core: streaming (used by websocket_server) ───────────────────────────

#     async def stream_sentences_to_queue(self, user_text: str, context: str, queue: asyncio.Queue):
#         """Stream PM response sentences to queue. PM-only — [FT] handled by websocket_server."""
#         import time as _t

#         t0 = _t.time()
#         meeting_context = await self._build_context(user_text, context)
#         rag_ms = (_t.time() - t0) * 1000
#         print(f"[Agent] ⏱ RAG context: {rag_ms:.0f}ms")

#         # Store CLEAN user text in history (not context blocks!)
#         self.history.append({"role": "user", "content": user_text})
#         if len(self.history) > 6:
#             self.history = self.history[-6:]

#         # Build system prompt: PM personality + meeting context
#         system_prompt = PM_PROMPT
#         if meeting_context:
#             system_prompt = PM_PROMPT + "\n\n" + meeting_context

#         total_chars = len(system_prompt) + sum(len(m["content"]) for m in self.history)
#         print(f"[Agent] ⏱ Context size: {total_chars} chars (~{total_chars//4} tokens)")

#         # _debug_log("PM STREAM",
#         #            user_text=user_text,
#         #            convo_history_raw=context or "(EMPTY)",
#         #            meeting_context=meeting_context or "(NONE)",
#         #            llm_history=[f"{m['role']}: {m['content'][:80]}" for m in self.history],
#         #            system_prompt_preview=system_prompt[-300:])

#         try:
#             t1 = _t.time()
#             stream = await self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[{"role": "system", "content": system_prompt}] + self.history,
#                 temperature=0.7,
#                 max_tokens=120,
#                 stream=True,
#             )
#             stream_open_ms = (_t.time() - t1) * 1000
#             print(f"[Agent] ⏱ Stream opened: {stream_open_ms:.0f}ms")

#             buffer = ""
#             full_response = ""
#             first_token_time = None
#             sentence_count = 0
#             async for chunk in stream:
#                 token = chunk.choices[0].delta.content if chunk.choices else None
#                 if not token:
#                     continue

#                 if first_token_time is None:
#                     first_token_time = _t.time()
#                     ttft_ms = (first_token_time - t1) * 1000
#                     print(f"[Agent] ⏱ First token: {ttft_ms:.0f}ms")

#                 buffer += token
#                 full_response += token

#                 while True:
#                     indices = [buffer.find(c) for c in ".!?" if buffer.find(c) != -1]
#                     if not indices:
#                         break
#                     idx = min(indices)
#                     sentence = buffer[:idx+1].strip()
#                     buffer = buffer[idx+1:].lstrip()
#                     if sentence and len(sentence) > 2:  # skip punctuation-only fragments like "."
#                         # Strip emote/action markers — TTS would say "(laughs)" literally
#                         sentence = re.sub(r'\([^)]*\)', '', sentence).strip()
#                         if not sentence or len(sentence) <= 2:
#                             continue
#                         sentence_count += 1
#                         sent_ms = (_t.time() - t1) * 1000
#                         print(f"[Agent] ⏱ Sentence {sentence_count} ready: {sent_ms:.0f}ms")
#                         await queue.put(sentence)

#             llm_total_ms = (_t.time() - t1) * 1000
#             print(f"[Agent] ⏱ LLM total: {llm_total_ms:.0f}ms ({len(full_response.split())} words)")

#             if buffer.strip() and len(buffer.strip()) > 2:
#                 clean_buf = re.sub(r'\([^)]*\)', '', buffer).strip()
#                 if clean_buf and len(clean_buf) > 2:
#                     await queue.put(clean_buf)
#             self.history.append({"role": "assistant", "content": full_response.strip()})

#         except Exception as e:
#             print(f"[Agent] LLM error: {e}")
#             await queue.put("Hmm, something went wrong on my end.")
#         finally:
#             await queue.put(None)

#     # ── Helpers ───────────────────────────────────────────────────────────────

#     async def _llm_call(self, user_msg: str, system: str, max_tokens: int = 60) -> str:
#         """LLM call with clean history. user_msg goes as clean text, not context dump."""
#         self.history.append({"role": "user", "content": user_msg})
#         if len(self.history) > 6:
#             self.history = self.history[-6:]

#         stream = await self.client.chat.completions.create(
#             model=self.model,
#             messages=[{"role": "system", "content": system}] + self.history,
#             temperature=0.7,
#             max_tokens=max_tokens,
#             stream=True,
#         )

#         tokens = []
#         async for chunk in stream:
#             t = chunk.choices[0].delta.content if chunk.choices else None
#             if t:
#                 tokens.append(t)

#         result = "".join(tokens).strip()
#         self.history.append({"role": "assistant", "content": result})
#         return result

#     def _split_sentences(self, text: str) -> list[str]:
#         parts = re.split(r'(?<=[.!?])\s+', text.strip())
#         return [p.strip() for p in parts if p.strip()]

#     def reset(self):
#         self.history.clear()
#         self.rag.clear()

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
# Stage 2.6: ResearchJournal — append-only RAG cache for follow-ups
# ══════════════════════════════════════════════════════════════════════════════

class ResearchJournal:
    """Append-only research history with semantic retrieval.

    Replaces Stage 2.5's single-slot _research_cache. Each research turn
    appends a new entry. Follow-up questions search the journal by embedding
    similarity (using the same fastembed model MeetingRAG uses).

    Why not just reuse MeetingRAG?
      MeetingRAG stores raw conversation snippets. ResearchJournal stores
      structured research bundles (question, answer, fetched tickets, web
      results, plus type/timestamp for TTL). Different shape, different uses.

    Entry shape:
      {
        "fetched_at":         float,          # time.time()
        "question":           str,            # rewritten search query
        "raw_question":       str,            # original user text
        "synthesis_output":   str,            # what Sam said
        "jira_tickets":       list[dict],     # fresh tickets from research
        "web_results":        list[dict],     # Exa/Brave results
        "question_type":      str,            # planner classification
        "vector":             np.ndarray|None # 384-dim bge embedding
      }
    """

    # TTL by question type — drives retrieval ordering when scores are close
    _TTL_BY_TYPE = {
        "jira_status":     30,
        "general":         300,
        "feasibility":     1800,
        "tech_switch":     1800,
        "best_practices":  1800,
        "internal_org":    3600,
    }

    def __init__(self, embed_model=None):
        """embed_model: shared TextEmbedding from MeetingRAG (avoid reload)."""
        self._entries: list[dict] = []
        self._model = embed_model
        self._ready = embed_model is not None
        if self._ready:
            print("[Journal] ✅ Research journal ready (sharing MeetingRAG embed model)")
        else:
            print("[Journal] ⚠️  No embed model — keyword fallback only")

    def _embed_sync(self, text):
        if not self._model:
            return None
        try:
            vectors = list(self._model.embed([text]))
            return np.array(vectors[0], dtype=np.float32)
        except Exception as e:
            print(f"[Journal] Embed failed: {e}")
            return None

    async def add(self, entry: dict):
        """Append a research entry, embedding the question text in background.

        Non-blocking — the embedding happens in a thread pool. The entry is
        appended immediately so subsequent searches can find it (initially
        via keyword fallback until the vector is ready).
        """
        import time as _t
        entry.setdefault("fetched_at", _t.time())
        entry.setdefault("vector", None)

        # Append immediately so retrieval can find it via keyword if needed
        self._entries.append(entry)

        # Embed in background — find this same entry in the list and update
        if self._ready:
            text_to_embed = entry.get("question") or entry.get("raw_question") or ""
            if not text_to_embed.strip():
                return
            try:
                loop = asyncio.get_event_loop()
                vector = await loop.run_in_executor(
                    None, self._embed_sync, text_to_embed
                )
                entry["vector"] = vector
            except Exception as e:
                print(f"[Journal] Background embed failed: {e}")

        print(f"[Journal] 📔 Entry #{len(self._entries)} added: "
              f"q='{entry.get('question', '')[:50]}', "
              f"type={entry.get('question_type', '?')}, "
              f"tickets={len(entry.get('jira_tickets', []))}, "
              f"web={len(entry.get('web_results', []))}")

    async def search(self, query: str, top_k: int = 1, min_score: float = 0.5):
        """Find the most relevant journal entry for the rewritten query.

        Returns: list of (score, entry) tuples, top_k highest-scoring entries
                 with score >= min_score. Empty list if no matches.

        Uses cosine similarity on embeddings. Falls back to keyword matching
        if embedding fails or vectors aren't ready yet.

        min_score guards against low-similarity false positives. 0.5 is
        conservative for bge-small-en-v1.5 (384-dim).
        """
        if not self._entries:
            return []

        # Vector path
        if self._ready and self._model:
            try:
                loop = asyncio.get_event_loop()
                query_vec = await loop.run_in_executor(
                    None, self._embed_sync, query
                )
                if query_vec is not None:
                    scored = []
                    for entry in self._entries:
                        ev = entry.get("vector")
                        if ev is None:
                            continue
                        sim = self._cosine_sim(query_vec, ev)
                        scored.append((sim, entry))

                    if scored:
                        scored.sort(key=lambda x: x[0], reverse=True)
                        results = [(s, e) for s, e in scored[:top_k] if s >= min_score]
                        if results:
                            print(f"[Journal] 🎯 Vector search: top score {results[0][0]:.3f} "
                                  f"({len(results)} hits >= {min_score})")
                            return results
                        # Log near-misses so we can tune min_score
                        if scored:
                            print(f"[Journal] ⚠️  Best score {scored[0][0]:.3f} below "
                                  f"min_score {min_score} — treating as miss")
            except Exception as e:
                print(f"[Journal] Vector search error: {e}")

        # Keyword fallback (when no vectors ready or embedding broke)
        return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int = 1):
        """Fallback when vectors unavailable — score by keyword overlap."""
        STOP = {"the", "a", "an", "is", "are", "was", "were", "what", "who",
                "how", "when", "where", "why", "did", "do", "does", "tell",
                "me", "more", "about", "of", "in", "on", "at", "to", "for"}
        qwords = {w.lower() for w in query.split() if len(w) > 2 and w.lower() not in STOP}
        if not qwords:
            return []
        scored = []
        for entry in self._entries:
            text = (
                (entry.get("question") or "") + " " +
                (entry.get("raw_question") or "") + " " +
                (entry.get("synthesis_output") or "")
            ).lower()
            hits = sum(1 for w in qwords if w in text)
            if hits > 0:
                # Normalize to a [0,1] score for consistent thresholding
                score = min(1.0, hits / max(len(qwords), 1))
                scored.append((score, entry))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            print(f"[Journal] 🔤 Keyword fallback: top score {scored[0][0]:.2f}")
            return scored[:top_k]
        return []

    @staticmethod
    def _cosine_sim(a, b):
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
# ── Stage S: Unified PM prompt (PM_PROMPT_V2) ───────────────────────────────
# One prompt, two paths:
#   - No cache available  → use PM_PROMPT_V2_BASE alone (15-25 words, quick reply)
#   - Cached research available → append PM_PROMPT_V2_CACHED_SUFFIX (40-60 words,
#     references cached tickets/web/synthesis)
#
# String concatenation is used (not empty {placeholder} blocks) so the model
# never sees half-empty sections that invite hallucination.
#
# Backward-compat aliases PM_PROMPT and FAST_PM_CACHED_PROMPT are kept at the
# bottom of this file so existing imports keep working during rollout.

PM_PROMPT_V2_BASE = """You are Sam, a senior FEMALE PM at AnavClouds Software Solutions. You're on a live voice call in a meeting. 

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
- ALWAYS TALK LIKE HUMAN WOULD SAY OUT LOUD IN MEETING — not like a written report or chatbot. Read it back to yourself as if you were saying it.
-ALWAYS TRYING USING PUNCTUATION IN YOUR RESPONSES TO MAKE THEM SOUND MORE NATURAL WHEN SPOKEN ALOUD.
- ALWAYS TRY TO ENGAGE WITH THE USER'S ACTUAL QUESTION AND THE CONVERSATION — show you're listening and responding to what they actually said, not just spitting out a generic answer and give follow ups based on the conversation flow.
YOUR BACKGROUND:
- Senior PM at AnavClouds (Salesforce + AI company). You handle sprints, budgets, timelines, CRM rollouts.
- Confident and warm. You deflect weird off-topic questions with light humor,
  but never sarcasm directed AT the user.
- You remember what was said in the meeting (see MEETING MEMORY below if provided).

LENGTH (default — for quick replies with no cached research):
- You are SPEAKING on a live call. Every extra word wastes everyone's time.
- 1-2 sentences MAXIMUM. Aim for 15-25 words total.
- Think of it like a quick reply in a meeting — not a presentation.
- If your response would take more than 5 seconds to say out loud, it's TOO LONG. Cut it.
- NEVER give lengthy explanations, lists, or multi-point answers. That's what [RESEARCH] is for.
- NOTE: if a CACHED CONTEXT section is appended below, follow ITS length rules
  (longer responses are fine because you have data to reference).

OTHER RULES:
- If user asks to REPEAT something: rephrase your last answer differently.
- Say something NEW each time — don't repeat yourself.
- No markdown, no lists, no asterisks, no bullet points, no parenthetical actions like (laughs) or (sighs) — this is voice output, not a script.
- If you don't have specific data (ticket IDs, numbers, facts), DON'T invent them. Say something general or offer to look it up.

MEETING MEMORY: Use this to answer about past discussions if provided."""


# ── Cached-context suffix (Stage S) ─────────────────────────────────────────
# Appended to PM_PROMPT_V2_BASE only when the caller has fresh cached research
# from a previous turn. All template slots are required — caller MUST format
# this with non-empty values (use placeholder strings like "(no cached tickets)"
# rather than empty strings, to avoid hallucination on missing data).

PM_PROMPT_V2_CACHED_SUFFIX = """══════════════════════════════════════════════════════════════
CACHED CONTEXT FROM PRIOR RESEARCH TURN (use this for the follow-up):

CLIENT PROFILE (always authoritative for names and company facts):
{client_profile_block}

MEETING AGENDA:
{agenda_block}

CACHED PROJECT CONTEXT (from research {cache_age_sec} seconds ago):
{cached_tickets}

CACHED WEB RESEARCH (retrieved {cache_age_sec} seconds ago):
{cached_web}

YOUR PRIOR RESPONSE (the last thing you said in this conversation):
{cached_synthesis}

CURRENT CONVERSATION (most recent turns):
{conversation_block}

CLIENT JUST ASKED:
"{question}"

══════════════════════════════════════════════════════════════
HOW TO ANSWER (cached-context mode — overrides default length above):

The cached context above is from THIS conversation, just moments ago. The user is almost certainly asking a follow-up that builds on it.

1. LENGTH: 2 to 4 sentences, 40 to 60 words. You have data — use it. Speak like a senior PM mid-conversation.

2. If the cached context contains what you need to answer, respond naturally and conversationally. No bullet points, no markdown, no source citations.

3. For NAMES: only use names from the CLIENT PROFILE section. Do not invent names. Do not use placeholder names.

4. Reference cached tickets by ID where relevant (e.g. "SCRUM-244 is still in progress").

5. You can build on your prior response naturally ("As I mentioned, ...", "right, the same one we just talked about").

6. Be confident. The cached context is fresh and authoritative.

══════════════════════════════════════════════════════════════
ESCAPE HATCH — when to ESCALATE:

If the cached context CANNOT answer this question — for example:
  - The user is asking about a NEW ticket that's not in the cache
  - The user shifted to a totally NEW topic outside what was researched
  - The user is asking for fresh facts that the cache doesn't have

Then respond with EXACTLY this single word and nothing else:

ESCALATE

When in doubt, prefer ESCALATE — a fresh research turn is much better than guessing or making something up. Do not write "ESCALATE because..." or "I should escalate" — just the word ESCALATE on its own.

══════════════════════════════════════════════════════════════
YOUR RESPONSE:"""


# Backward-compat aliases — keep existing imports working unchanged.
# PM_PROMPT and FAST_PM_CACHED_PROMPT are now thin wrappers around the
# unified PM_PROMPT_V2 prompts above. Future code should import the V2
# names directly (PM_PROMPT_V2_BASE / PM_PROMPT_V2_CACHED_SUFFIX).
PM_PROMPT = PM_PROMPT_V2_BASE

RESEARCH_PROMPT = """You are Sam, a senior PM at AnavClouds Software Solutions. You're on a live voice call.
The user asked a question that needs data. You have multiple complementary data sources — use them together to give the most useful, grounded answer possible.

YOUR DATA SOURCES (each may or may not be populated for any given turn):
1. The CLIENT BACKGROUND that may be prepended above this prompt — what the client's business does, their work, who they are. This grounds answers in what's actually relevant to THEM, not generic textbook advice.
2. PROJECT CONTEXT (current Jira tickets) — what work is in flight for this client, technologies in use, ongoing priorities.
3. RELATED TICKETS — tickets specifically matched to this question via semantic search. These are the most relevant work items to the question being asked.
4. JIRA ACTION RESULTS — outcome of any action just taken on Jira (move, create, update).
5. WEB RESEARCH — fresh external data from a real-time web search (when the question depended on current-world facts).
6. CONVERSATION SO FAR — what was just said in this meeting.

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

CORE RULE — synthesize, don't silo:
When multiple sources have relevant information, weave them together. Use web research for the factual answer, then connect it back to the client's actual business or active tickets WHEN THERE IS A MEANINGFUL CONNECTION. Don't force connections that aren't there — but also don't ignore the client context that's loaded right above you. A generic answer is a missed opportunity when client-specific context could make it more useful.

Examples of good synthesis:
- User asks a technical "how to" question → answer with the approach from web research, reference any matching ticket by ID, mention how it fits the client's stack/work
- User asks a current-fact question (e.g. who holds a role, what's a price) → give the fresh fact from the web. If it has zero connection to their business, just answer cleanly — don't manufacture relevance.
- User asks about their own project → answer from Jira data primarily, supplement with web context (e.g. industry best practices) if it adds value.

Per question type:
- For factual questions: give the answer directly, add useful context. If client context is genuinely relevant, mention how it connects.
- For Jira queries (sprint status, ticket details): summarize the data with insight, not just numbers
- For Jira actions (move ticket, create ticket): confirm what was done and add context
- For feasibility/implementation questions: assess feasibility against the client's actual stack and tickets, reference related tickets by ID, outline an approach grounded in their work, estimate complexity (Low/Medium/High), offer to create a ticket
- Reference specific ticket IDs (e.g. SCRUM-12) when they actually relate to what's being discussed
- If the data doesn't have the answer, say so naturally — don't fabricate

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

SEARCH_QUERY_PROMPT = """Decide whether the user's question needs a fresh web search, and if so, write a clean Google query for it.

CORE PRINCIPLE — when to search vs SKIP:

The deciding factor is whether the answer depends on the CURRENT STATE OF THE WORLD — anything that can change over time and may have changed since the LLM's training data was collected. If the answer could be different today than it was a year ago, search the web. If the answer is fundamentally stable across time, skip.

OUTPUT a search query when the answer depends on the current state of the world. Examples of this category include (but are not limited to):
- Who currently holds any role, position, title, or office (anywhere in the world, any organization)
- Current price, rate, valuation, or market data of anything
- Current status, version, release, or state of any product, technology, organization, or person
- Recent news, events, announcements, or changes
- Whether something still exists, is still in operation, or has been replaced
- The latest, newest, current, or up-to-date instance of anything
- Real-world facts about specific named people, companies, events, or places that may have evolved

OUTPUT "SKIP" only when the answer is fundamentally stable and unlikely to have changed:
- Definitions, concepts, terminology (what IS X — a stable concept)
- Mathematical, scientific, or logical facts (laws of physics, formulas)
- Programming syntax, language features that are well-established
- Well-established historical events from the distant past (decades old, settled facts)
- How-to-implement-X technical guidance (best practices, design patterns, architecture)
- The user's own internal project data — Jira tickets, sprint status, team activity, conversation memory. The web has no access to private project data.
- Requests to perform an action on internal systems (move ticket, create ticket, list tickets)

When in doubt, prefer to search. A confidently wrong stale answer is worse than a slightly slower correct one.

QUERY FORMATTING (when not SKIP):
- 8-15 words, descriptive enough that Google returns useful results
- Add the current year when the question is about something that changes over time (e.g. "current X 2026")
- Replace "our company" / "my company" / "we" / "our" with "AnavClouds Software Solutions"
- If the user does NOT mention the company, do NOT add AnavClouds
- Remove "Sam" and conversational filler ("um", "like", "you know")
- For multi-part questions, fold all parts into a single descriptive query

PROJECT-CONTEXT HINTS:
If pre-loaded ticket context is provided and mentions specific technologies (React, Node.js, MongoDB, etc.), include them in technical queries. Example: user asks "add real-time notifications" and tickets mention WebSocket + Node.js → "real-time notifications WebSocket Node.js implementation".

ABSTRACT EXAMPLES (the PATTERN matters, not the specific words):

Pattern: "who is currently [role] of [entity]" → search (officeholder may have changed)
"Who is the president of [country]?" → "current president [country] 2026"
"Who runs / leads / heads [organization]?" → "current head of [organization] 2026"

Pattern: "what is the current [attribute] of [thing]" → search (values change)
"What's the price of [asset]?" → "current price of [asset] 2026"
"What's the latest version of [tech]?" → "current latest version [tech]"

Pattern: "what is [stable concept]" → SKIP (definitional, doesn't change)
"What is a foreign key?" → SKIP
"How does TLS handshake work?" → SKIP

Pattern: "how do I implement / build / approach [technical task]" → search (helpful for best practices)
"Best way to handle authentication?" → "authentication implementation best practices 2026"
"Should we use [tech A] or [tech B]?" → "[tech A] vs [tech B] comparison pros cons"

Pattern: project-internal data → SKIP (web has no access)
"What are my open tickets?" → SKIP
"What's our sprint status?" → SKIP
"How many tickets in progress?" → SKIP
"Move SCRUM-15 to done" → SKIP
"Create a ticket for the login bug" → SKIP
"What did we discuss earlier?" → SKIP

Output ONLY the search query or the literal word SKIP. No quotes, no explanation, no preamble."""


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
- "needs_fresh_jira": Stage 2.11: LLM freshness judgment (no keywords). Set true when the question is asking about the PRESENT STATE of tickets — current status, who is assigned, what is blocked, what has changed, what is now in progress, what should be worried about today, who is working on what, anything that depends on the latest state of work. Set true even if the cache contains the relevant ticket — if the user is asking about its current state, fresh data is needed. Set false only when the question is about historical context, ticket definitions, or static information that does NOT depend on current data (e.g., "what was SCRUM-244 about" referring to its description). For internal_org questions, set to false.
- "jira_search_terms": Short phrase (2-5 words) for a Jira text search. Only populated if needs_fresh_jira is true.
- "web_search_query": A clean Google search query. For jira_status and internal_org questions, output exactly "SKIP".
- "serpapi_context_features": Natural-language descriptions of project features relevant to this question (no ticket IDs). Max 6 items. Derived from cached tickets + conversation. Empty list for jira_status and internal_org questions.

CRITICAL: When in doubt about whether a question is about the COMPANY/TEAM/PEOPLE vs a TECHNICAL topic, classify as "internal_org". This prevents the system from hallucinating company-specific facts.

Output ONLY the JSON object. No markdown, no explanation, no code fences."""


# ── Stage 2.6: query rewriter ──────────────────────────────────────────────
# Rewrites vague follow-up questions into self-contained search queries.
# Resolves pronouns and "the X" against the last 5 turns of conversation.
# Runs in parallel with EOT + addressee decider — wall-clock cost ~zero.
QUERY_REWRITER_PROMPT = """You are a query rewriter for a meeting assistant. Your job is to make follow-up questions self-contained so they can be searched against past research.

Recent conversation (most recent last):
{conversation}

Current user question: "{question}"

Rewrite the question so it stands alone. Resolve pronouns ("him", "her", "it", "that") and vague references ("the president", "that ticket", "the same thing") using the conversation above. If the question is already self-contained, return it unchanged.

Rules:
- Output ONLY the rewritten question, nothing else
- Keep it ONE sentence
- Keep it short — under 25 words
- If you can't resolve a reference, leave it as-is rather than guessing
- Don't add information that wasn't in the conversation

Rewritten question:"""


# Stage 2.5: fast-PM prompt for cache-hit follow-ups.
# Used by _fast_pm_with_research_cache to answer follow-up questions from
# cached research context without re-fetching. If the cache cannot answer,
# the model is instructed to respond with the single word "ESCALATE", which
# cancels the fast path and falls through to a fresh research turn.
# cached research context without re-fetching. If the cache cannot answer,
# the model is instructed to respond with the single word "ESCALATE", which
# cancels the fast path and falls through to a fresh research turn.
#
# Stage S: this name is now an alias for PM_PROMPT_V2_BASE + cached suffix.
# All prompt content lives in PM_PROMPT_V2_BASE / PM_PROMPT_V2_CACHED_SUFFIX
# above; this constant is kept so existing imports keep working unchanged.
FAST_PM_CACHED_PROMPT = PM_PROMPT_V2_BASE + "\n\n" + PM_PROMPT_V2_CACHED_SUFFIX

RESEARCH_SYNTHESIS_PROMPT = """You are Sam, a senior developer and PM at AnavClouds Software Solutions, on a live voice call with a client. Below is the context for the call. Use it to answer the client's question naturally and conversationally.
 
══════════════════════════════════════════════════════════════
LIVE JIRA TICKETS (authoritative — use as ground truth):
{project_context}
 
══════════════════════════════════════════════════════════════
MEETING AGENDA (today's topics, in order):
{agenda_block}
 
══════════════════════════════════════════════════════════════
CLIENT PROFILE:
{client_profile_block}
 
══════════════════════════════════════════════════════════════
WEB SEARCH RESULTS (fresh from authoritative sources, just retrieved):
{web_search_results}
 
══════════════════════════════════════════════════════════════
CONVERSATION SO FAR (most recent turns):
{conversation_block}
 
══════════════════════════════════════════════════════════════
CLIENT JUST ASKED:
"{question}"
 
══════════════════════════════════════════════════════════════
HOW TO ANSWER:
 
Treat the LIVE JIRA TICKETS section as the authoritative source of truth about the project. Do NOT invent ticket statuses, priorities, or details that aren't there. If asked about something specific that isn't in the tickets, say so plainly — "I don't see that in our current tickets" — rather than guessing.
 
For the AGENDA, only use the items listed above. Do not infer the agenda from tickets — if asked about today's agenda, list ONLY what's in the AGENDA section. If the agenda section is empty, say "we don't have a fixed agenda for today" rather than fabricating one.
 
For NAMES: Only use names that appear in the CLIENT PROFILE "ON THIS CALL" section. Do NOT make up names. Do NOT use names from the LIVE JIRA TICKETS section as if they're on the call (those are people referenced in tickets, not call participants). Do NOT use placeholder names like "Rachel", "Mike", "John". If the CLIENT PROFILE section says no one is loaded, just speak naturally without addressing anyone by name.
 
For the WEB SEARCH RESULTS, treat them as fresh, authoritative reference material that was just retrieved from real sources (websites, articles, official docs). Use them to ground factual or technical answers. NEVER cite URLs, source names, or "according to..." out loud — just speak the information naturally as if you know it. Do not say "the website mentions..." or "based on what I found...".
 
For general technical or industry questions (best practices, comparisons, current events, public facts), ground your answer in the WEB SEARCH RESULTS above. For anything specific to AnavClouds or this client's work, use the LIVE JIRA TICKETS and CLIENT PROFILE first, supplement with web context if it adds value.
 
If the WEB SEARCH RESULTS don't contain the answer (or weren't relevant), say briefly that you don't have specific info on that and offer to follow up — don't invent specifics.
 
Speak naturally — like a senior PM thinking out loud to a client in real time. Use a warm, conversational tone. Small phrases like "yeah", "so", "the way I see it", "honestly", "what we're seeing here" feel natural and human. Don't be robotic, don't use bullet points, don't cite sources, don't reference URLs, don't use markdown.
 
Aim for {length} (around 70-90 words across 4-5 sentences). Acknowledge the question briefly, give the substance with warmth, and end with a natural close — a thought, a small implication, or a soft handoff. Don't always end with a question; vary how you close.
 
If you genuinely don't have enough info to answer, say so honestly and briefly — "I'm not sure about that one specifically, want me to check after the call?" — rather than making things up.

- ALWAYS TALK LIKE HUMAN WOULD SAY OUT LOUD IN MEETING — not like a written report or chatbot. Read it back to yourself as if you were saying it.
-ALWAYS TRYING USING PUNCTUATION IN YOUR RESPONSES TO MAKE THEM SOUND MORE NATURAL WHEN SPOKEN ALOUD.
- ALWAYS TRY TO ENGAGE WITH THE USER'S ACTUAL QUESTION AND THE CONVERSATION — show you're listening and responding to what they actually said, not just spitting out a generic answer and give follow ups based on the conversation flow.
"""

# SerpAPI-direct persona prompt: builds the actual query sent to Google AI Mode.
# Proven through testing to return TTS-ready responses when TTS hint is included.
SERPAPI_DIRECT_PROMPT = """You are Sam, a senior developer and PM at AnavClouds Software Solutions, on a live voice call with a client. Below is the context for the call. Use it to answer the client's question naturally and conversationally.

══════════════════════════════════════════════════════════════
LIVE JIRA TICKETS (authoritative — use as ground truth):
{project_context}

══════════════════════════════════════════════════════════════
MEETING AGENDA (today's topics, in order):
{agenda_block}

══════════════════════════════════════════════════════════════
CLIENT PROFILE:
{client_profile_block}

══════════════════════════════════════════════════════════════
CONVERSATION SO FAR (most recent turns):
{conversation_block}

══════════════════════════════════════════════════════════════
CLIENT JUST ASKED:
"{question}"

══════════════════════════════════════════════════════════════
HOW TO ANSWER:

Treat the LIVE JIRA TICKETS section as the authoritative source of truth about the project. Do NOT invent ticket statuses, priorities, or details that aren't there. If asked about something specific that isn't in the tickets, say so plainly — "I don't see that in our current tickets" — rather than guessing.

For the AGENDA, only use the items listed above. Do not infer the agenda from tickets — if asked about today's agenda, list ONLY what's in the AGENDA section. If the agenda section is empty, say "we don't have a fixed agenda for today" rather than fabricating one.

For NAMES: Only use names that appear in the CLIENT PROFILE "ON THIS CALL" section. Do NOT make up names. Do NOT use names from the LIVE JIRA TICKETS section as if they're on the call (those are people referenced in tickets, not call participants). Do NOT use placeholder names like "Rachel", "Mike", "John". If the CLIENT PROFILE section says no one is loaded, just speak naturally without addressing anyone by name.

For general technical or industry questions (best practices, comparisons, current events, public facts), use your trusted web sources. For anything specific to AnavClouds or this client's work, stick to what's in the context above.

Speak naturally — like a senior PM thinking out loud to a client in real time. Use a warm, conversational tone. Small phrases like "yeah", "so", "the way I see it", "honestly", "what we're seeing here" feel natural and human. Don't be robotic, don't use bullet points, don't cite sources, don't reference URLs, don't use markdown.

Aim for {length} (around 70-90 words across 4-5 sentences). Acknowledge the question briefly, give the substance with warmth, and end with a natural close — a thought, a small implication, or a soft handoff. Don't always end with a question; vary how you close.

If you genuinely don't have enough info to answer, say so honestly and briefly — "I'm not sure about that one specifically, want me to check after the call?" — rather than making things up."""


# Dynamic filler prompt: generates a longer, contextual acknowledgment that
# matches the 7-second hardcoded fillers in length (3-4 sentences, 30-50 words).
# Runs in parallel with router/trigger decisions. 1500ms timeout (Groq 70b).
FILLER_PROMPT = """You are Sam, a senior PM on a live voice call. Someone just asked you a question that needs a quick research lookup. Generate a warm 3-4 sentence filler to say BEFORE looking up the answer. The filler should cover ~7 seconds of speech to give the research pipeline time to complete.

Rules:
- 3 to 4 sentences, 30 to 50 words total
- Pattern: ACKNOWLEDGE the topic → THINKING action → SOFT CLOSE
- Sound like a real person pausing to think, not a recording
- Use natural phrases: "yeah", "hmm", "one sec", "let me", "give me a moment", "I want to make sure"
- Reference the topic VAGUELY (don't commit to specific facts you'd need to verify)
- No meta-commentary ("as an AI", "sure, here's")
- No bullet points, no markdown, just conversational prose

Examples:
Question: "Can we migrate from MongoDB to PostgreSQL?"
Filler: "Hmm, that's a good one. Let me pull up what we have on the data setup before I commit to anything specific. I want to make sure I'm giving you accurate info on the migration tradeoffs, give me just a sec."

Question: "What's the status of SCRUM-31?"
Filler: "Yeah, let me grab the latest on SCRUM-31 for you — I want to check if there's been any movement on it recently. One moment while I pull that up properly, won't take too long."

Question: "How's the sprint going?"
Filler: "Right, let me look at the sprint board real quick and pull together where we stand. I want to give you a solid picture rather than a half-answer, so just give me a moment to check the latest."

Question: "Best practices for OTP authentication?"
Filler: "Honestly, good question — let me think through what's been working well in the field and what fits our setup specifically. Give me just a second to pull together the relevant pieces before I jump in."

Now generate a filler for: "{question}"

Output ONLY the filler text. No quotes, no explanation, no prefix."""


# Longer warm fillers (~4-5s each at TTS pace, 12-18 words) — buy real time
# while research pipeline runs (Jira intent + Web optimizer + SerpAPI + Azure
# synthesis can take 6-10s for research-route turns). Conversational, not
# stalling-for-time, with natural intonation and warmth so the user doesn't
# feel like they're listening to a hold message.
#
# Earlier short fillers (~1-2s) only covered the first second of the gap and
# left awkward dead air for the remaining 5-7s. These cover the realistic
# end-to-end research latency for typical questions.
FILLERS = [
    "Hmm, good question — let me pull up what I know about that for you, give me just a moment to get it right.",
    "Yeah, give me a second to dig into that. I want to make sure I'm giving you the most accurate picture I can.",
    "Right, let me grab the latest on that — won't take too long, just want to be sure I'm giving you good info.",
    "Okay, hold on a moment while I look that up properly. I want to give you a solid answer here, give me just a sec.",
    "Honestly, let me pull the current data on that real quick before I respond — I'd rather be accurate than fast.",
    "One sec, I'm checking the most recent info on this so you get the right answer. Just a moment, won't take long.",
    "Sure, let me look into that for you — I want to make sure what I tell you is accurate and current. One moment.",
    "Good one — give me a moment to check the latest details on that before I jump in with anything. Just a second.",
    "Let me see what I can find on that for you. Just a sec, want to get this right and pull together what's relevant.",
    "Yeah, hang on a second. I'm grabbing the most up-to-date info on that for you, want to give you the full picture.",
    "Alright, one moment while I pull that up. I'd rather be accurate than fast here, so just hang tight a sec.",
    "Hmm, let me check on that properly — give me just a second or two to get you the right answer on this one.",
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

        # Stage 2.6: research journal — separate from conversation RAG.
        # Stores research entries (question + answer + fetched data) and
        # supports semantic retrieval for follow-up questions.
        # Shares the embed model with self.rag to avoid loading bge twice.
        self.journal = ResearchJournal(
            embed_model=getattr(self.rag, "_model", None)
        )

    def start(self):
        """Call once after event loop is running to start background embedder + warmup."""
        self.rag.start_background_embedder()
        asyncio.create_task(self._warmup())

    async def _warmup(self):
        """Pre-establish TCP connection to Groq — saves ~300ms on first real call.

        Stage 2.9: also pre-warms the rewriter (which uses llama-3.1-8b-instant,
        a different model than self.model). Without this, the FIRST rewriter
        call in a session takes ~1100ms (TLS + cold model), often timing out.
        After warmup it consistently runs in 200-700ms.
        """
        try:
            await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            print("[Agent] ✅ Groq connection warmed up")
        except Exception:
            pass

        # Stage 2.9: pre-warm the rewriter model (llama-3.1-8b-instant).
        # Fire-and-forget — failure here is non-fatal, runtime still works.
        try:
            await self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            print("[Agent] ✅ Rewriter model warmed up (llama-3.1-8b-instant)")
        except Exception as e:
            print(f"[Agent] ⚠️  Rewriter warmup failed (non-fatal): "
                  f"{type(e).__name__}")

    def _get_web_search(self):
        if not hasattr(self, '_web_search') or self._web_search is None:
            from external_apis import WebSearch
            self._web_search = WebSearch()
        return self._web_search
    
    def _get_exa_search(self):
        """Lazily instantiate ExaSearch singleton on first use.
 
        Mirrors the _get_web_search() pattern. Stays None until first called,
        so sessions that never use Exa don't pay the construction cost.
        """
        if not hasattr(self, "_exa_search") or self._exa_search is None:
            from external_apis import ExaSearch
            self._exa_search = ExaSearch()
        return self._exa_search
    
    
    async def exa_stream_synthesis_to_queue(
        self,
        user_text: str,
        exa_results: list,
        project_context: str,
        azure_extractor,
        queue,  # asyncio.Queue
        conversation: str = "",
        agenda_block: str = "",
        client_profile_block: str = "",
        length: str = "70-90 words",
    ):
        """Stream Azure-synthesized research answer to a TTS queue, using Exa results.
 
        This is the Exa equivalent of serpapi_direct_research() — but instead of
        sending a long persona prompt to Brave AI Mode and getting back a single
        string, we:
          1. Pass Exa's structured search results to Azure
          2. Azure does the synthesis (with full project/agenda/profile context)
          3. Sentences stream into the queue as Azure produces them
 
        Same TTS-friendly sentence chunking as stream_research_to_queue().
        Same caller pattern: caller awaits the queue via _stream_pipelined().
 
        Args:
          user_text: the client's question (verbatim, after STT)
          exa_results: list of {title, url, content, published_date, score} from
                       ExaSearch.search()
          project_context: bullet-list of relevant tickets (may be enriched
                           with fresh Jira data — done at the call site)
          azure_extractor: the AzureExtractor instance (for endpoint/key reuse)
          queue: asyncio.Queue to push sentences into (push None when done)
          conversation: recent conversation turns (last 3 used)
          agenda_block: today's meeting agenda (formatted by caller)
          client_profile_block: client/company info (formatted by caller)
          length: target length hint, default "70-90 words"
 
        Behavior on failure:
          Pushes a one-line apology + None to the queue (matches stream_research
          _to_queue behavior). Does NOT raise — caller can keep streaming.
        """
        # Local imports — defensive, in case top-of-file imports differ
        import time as _t
        import json as _json
        import httpx as _httpx
 
        t0 = _t.time()
 
        # ── Format Exa results into a labeled WEB SEARCH RESULTS block ──
        if not exa_results:
            web_results_text = "(no web results available — search returned nothing usable)"
        else:
            lines = []
            for i, r in enumerate(exa_results[:8], 1):
                title = (r.get("title") or "").strip()
                content = (r.get("content") or "").strip()
                published = (r.get("published_date") or "").strip()
                if not content:
                    continue
                line = f"[{i}]"
                if title:
                    line += f" {title}"
                if published:
                    line += f" ({published})"
                line += f":\n{content}"
                lines.append(line)
            web_results_text = "\n\n".join(lines) if lines else "(no usable web results)"
 
        # ── Conversation block — last 3 turns, one per line ──
        if conversation:
            conv_lines = [
                l for l in conversation.strip().split("\n") if l.strip()
            ][-3:]
            conversation_block = "\n".join(conv_lines) if conv_lines else "(start of call)"
        else:
            conversation_block = "(start of call)"
 
        # ── Defaults so prompt sections never render blank ──
        if not project_context:
            project_context = "- (no specific project context available)"
        if not agenda_block:
            agenda_block = "(no fixed agenda for today)"
        if not client_profile_block:
            client_profile_block = "(no client profile loaded)"
 
        # ── Build the full system prompt ──
        system = RESEARCH_SYNTHESIS_PROMPT.format(
            question=user_text,
            project_context=project_context,
            agenda_block=agenda_block,
            client_profile_block=client_profile_block,
            web_search_results=web_results_text,
            conversation_block=conversation_block,
            length=length,
        )
 
        print(f"[Agent] 🔬 Exa synthesis prompt: "
              f"{len(system)} chars (web_results={len(web_results_text)} chars, "
              f"context={len(project_context)} chars)")
 
        # ── Path A: Azure unavailable → Groq fallback (matches stream_research_to_queue) ──
        if not azure_extractor or not azure_extractor.enabled:
            print("[Agent] ⚠️  Azure unavailable for Exa synthesis — falling back to Groq")
            try:
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_text},
                    ],
                    temperature=0.7,
                    max_tokens=200,
                    stream=True,
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
                                sentence = buffer[: idx + 1].strip()
                                buffer = buffer[idx + 2:]
                                if sentence:
                                    cleaned = self._clean_serpapi_for_tts(sentence)
                                    if cleaned:
                                        await queue.put(cleaned)
                                break
                if buffer.strip():
                    cleaned = self._clean_serpapi_for_tts(buffer.strip())
                    if cleaned:
                        await queue.put(cleaned)
                self.history.append({"role": "assistant", "content": full})
            except Exception as e:
                print(f"[Agent] ⚠️  Groq fallback failed: {e}")
                await queue.put("Sorry, I couldn't process that right now.")
            await queue.put(None)
            return
 
        # ── Path B: Azure GPT-4o mini streaming (the happy path) ──
        self.history.append({"role": "user", "content": user_text})
        if len(self.history) > 6:
            self.history = self.history[-6:]
 
        url = (
            f"{azure_extractor.endpoint}/openai/deployments/"
            f"{azure_extractor.deployment}/chat/completions"
            f"?api-version={azure_extractor.api_version}"
        )
 
        try:
            async with _httpx.AsyncClient(timeout=60) as client:
                async with client.stream(
                    "POST",
                    url,
                    headers={
                        "api-key": azure_extractor.api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_text},
                        ],
                        "temperature": 0.7,
                        "max_tokens": 250,
                        "stream": True,
                    },
                ) as response:
                    response.raise_for_status()
 
                    buffer = ""
                    full_response = ""
                    first_token = False
                    sentence_count = 0
 
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = _json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            token = delta.get("content", "")
                            if not token:
                                continue
 
                            if not first_token:
                                first_token = True
                                ttfs_ms = (_t.time() - t0) * 1000
                                print(f"[Agent] ⏱ EXA-SYNTH first token (TTFT): "
                                      f"{ttfs_ms:.0f}ms")
 
                            buffer += token
                            full_response += token
 
                            while ". " in buffer or "? " in buffer or "! " in buffer:
                                for sep in [". ", "? ", "! "]:
                                    idx = buffer.find(sep)
                                    if idx != -1:
                                        sentence = buffer[: idx + 1].strip()
                                        buffer = buffer[idx + 2:]
                                        if sentence:
                                            sentence_count += 1
                                            sent_ms = (_t.time() - t0) * 1000
                                            print(f"[Agent] ⏱ EXA-SYNTH sentence "
                                                  f"{sentence_count}: {sent_ms:.0f}ms")
                                            # Apply same TTS cleanup as Brave path
                                            cleaned = self._clean_serpapi_for_tts(sentence)
                                            if cleaned:
                                                await queue.put(cleaned)
                                        break
                        except (_json.JSONDecodeError, IndexError, KeyError):
                            continue
 
                    if buffer.strip():
                        sentence_count += 1
                        cleaned = self._clean_serpapi_for_tts(buffer.strip())
                        if cleaned:
                            await queue.put(cleaned)
 
                    total_ms = (_t.time() - t0) * 1000
                    words = len(full_response.split())
                    print(f"[Agent] ⏱ EXA-SYNTH total: {total_ms:.0f}ms "
                          f"({words} words, {sentence_count} sentences)")
                    self.history.append({"role": "assistant", "content": full_response})
 
        except Exception as e:
            print(f"[Agent] ❌ Exa synthesis Azure stream failed: "
                  f"{type(e).__name__}: {e}")
            await queue.put("Sorry, I ran into an issue researching that.")
 
        await queue.put(None)
        

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

    def _get_client_profile_block(self) -> str:
        """Return the participants list + client-profile block for injection
        into the RESEARCH prompt. Returns empty string if no profile loaded.

        Two distinct things go into the same block:
          1. PEOPLE ON THIS CALL — explicit participant list (prevents Sam
             from addressing the current speaker by a name pulled from the
             company background, e.g. calling "Sahil" by the CEO's name
             "Gaurav" because the CEO appears in the profile text).
          2. COMPANY BACKGROUND — research about the client's business,
             clearly labeled as reference material only.

        Separate from _format_meeting_state() so the research path can grab
        just the client context + people-on-call without the full agenda
        dump (research prompt has its own context blocks for those).
        """
        if self._dialogue_manager is None:
            return ""
        try:
            state = self._dialogue_manager.get_state_snapshot()
        except Exception:
            return ""
        if not state or not state.get("_initialized"):
            return ""

        cp = (state.get("client_profile") or "").strip()
        participants = state.get("participants") or []
        names = [str(p).strip() for p in participants if str(p).strip()]

        if not cp and not names:
            return ""

        chunks: list[str] = []
        if names:
            if len(names) == 1:
                chunks.append(
                    f"PEOPLE ON THIS CALL RIGHT NOW: {names[0]} "
                    f"(this is the ONLY person speaking — always address "
                    f"them by THIS name, never by names from the company "
                    f"background below)."
                )
            else:
                chunks.append(
                    f"PEOPLE ON THIS CALL RIGHT NOW: {', '.join(names)} "
                    f"(these are the only people speaking — always address "
                    f"them by THESE names, never by names from the company "
                    f"background below)."
                )

        if cp:
            # Stage 2.12: explicit speaker-to-profile binding.
            # Old heading framed the profile as "reference material" with no
            # connection to the speaker. New heading tells Sam: the people on
            # this call work at this company. Use the profile to answer
            # personal questions (my company / our CEO / our work) directly,
            # do not go to the web for things the profile already covers.
            chunks.append(
                "CLIENT YOU\'RE MEETING WITH "
                "(the person(s) on this call work at the company described "
                "below — this is authoritative info about who they are and "
                "what their company does. When they ask \"my company\", "
                "\"our team\", \"the CEO\", \"our work\", \"my industry\" — "
                "the answer is in THIS profile. Do NOT search the web for "
                "personal questions about the speaker\'s company; ground your "
                "response in this profile and answer directly. Use this "
                "context to make every recommendation specific to their "
                "actual business, products, and challenges. IMPORTANT: any "
                "names mentioned IN this background (founders, executives, "
                "advisors) are usually NOT on the call — only address the "
                "current speaker by the names listed under PEOPLE ON THIS "
                "CALL. Weave context in naturally; never recite verbatim):\n" + cp
            )

        return "\n\n".join(chunks)

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

        # PEOPLE ON THIS CALL — who Sam is actually talking to RIGHT NOW.
        # Listed FIRST and explicitly so Sam never confuses the participant
        # list with names mentioned in the company-research profile below.
        # Without this block, Sam can leak names from the client_profile
        # into addressing (e.g. calling "Sahil" by the CEO's name "Gaurav"
        # because the CEO appears prominently in the profile text).
        participants = state.get("participants") or []
        if participants:
            names = [str(p).strip() for p in participants if str(p).strip()]
            if names:
                if len(names) == 1:
                    lines.append(
                        f"PEOPLE ON THIS CALL RIGHT NOW: {names[0]} "
                        f"(this is the ONLY person speaking — always address "
                        f"them by THIS name, never by names that appear in "
                        f"the company background below)."
                    )
                else:
                    lines.append(
                        f"PEOPLE ON THIS CALL RIGHT NOW: {', '.join(names)} "
                        f"(these are the only people speaking — always "
                        f"address them by THESE names, never by names that "
                        f"appear in the company background below)."
                    )
                lines.append("")

        # Company background — research about the client's business. This is
        # REFERENCE MATERIAL ONLY. The names mentioned here (e.g. founders,
        # CEOs, advisors) are NOT necessarily on the call. Sam should use
        # this to understand the client's business context but must NEVER
        # use names from this section to address whoever is currently
        # speaking. Capped at ~300 words by the UI.
        client_profile = (state.get("client_profile") or "").strip()
        if client_profile:
            # Stage 2.12: explicit speaker-to-profile binding.
            lines.append(
                "CLIENT YOU\'RE MEETING WITH "
                "(the person(s) on this call work at the company described "
                "below — this is authoritative info about who they are and "
                "what their company does. When they ask \"my company\", "
                "\"our team\", \"the CEO\", \"our work\", \"my industry\" — "
                "the answer is in THIS profile. Do NOT search the web for "
                "personal questions about the speaker\'s company; ground "
                "your response in this profile and answer directly. Use "
                "this context to make every recommendation specific to "
                "their actual business, products, and challenges. "
                "IMPORTANT: any names mentioned IN this background "
                "(founders, executives, advisors) are usually NOT on the "
                "call — only address the current speaker by the names "
                "listed under PEOPLE ON THIS CALL. Weave business context "
                "in naturally; never recite verbatim):"
            )
            lines.append(client_profile)
            lines.append("")

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
                                              memory_header: str = "",
                                              rewritten_query: str = "") -> Optional[dict]:
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

        # Stage 2.7: if a rewritten (entity-resolved) form of the question is
        # available, use it as the user_text the planner classifies against.
        # The original (vague) form stays in the conversation history for
        # context. This gives the planner the strongest signal for both
        # question_type classification AND web_search_query generation.
        effective_user_text = user_text
        if rewritten_query and rewritten_query.strip() and rewritten_query.strip().lower() != user_text.strip().lower():
            effective_user_text = (
                f"{user_text}\n[Resolved entities for search: {rewritten_query.strip()}]"
            )

        prompt = UNIFIED_RESEARCH_PROMPT.format(
            user_text=effective_user_text,
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
                                        length: str = "70-90 words",
                                        agenda_block: str = "",
                                        client_profile_block: str = "") -> Optional[str]:
        """Direct SerpAPI call with structured persona prompt (Door #1 / Brave AI Mode).

        Sends a labeled-section prompt to Brave AI Mode containing:
          - LIVE JIRA TICKETS (authoritative ground truth)
          - MEETING AGENDA (today's actual topics)
          - CLIENT PROFILE (AnavClouds + client info)
          - CONVERSATION SO FAR (last 3 turns)
          - CLIENT JUST ASKED (the question)

        Returns cleaned voice-ready text or None on failure.
        Caller is responsible for falling back to Azure synthesis on None.

        Args:
          user_text: the client's question (verbatim, after STT)
          project_context: bullet-list of relevant tickets (may be enriched
                           with fresh Jira data — done at the call site)
          conversation: recent conversation turns (last 3 used)
          length: target length hint, default "70-90 words"
          agenda_block: today's meeting agenda (formatted by caller)
          client_profile_block: client/company info (formatted by caller)
        """
        import time as _t
        t0 = _t.time()

        # Build conversation block — last 3 turns, one per line
        if conversation:
            conv_lines = [l for l in conversation.strip().split("\n") if l.strip()][-3:]
            conversation_block = "\n".join(conv_lines) if conv_lines else "(start of call)"
        else:
            conversation_block = "(start of call)"

        # Defaults so prompt sections don't render as empty/blank
        if not project_context:
            project_context = "- (no specific project context available)"
        if not agenda_block:
            agenda_block = "(no fixed agenda for today)"
        if not client_profile_block:
            client_profile_block = "(no client profile loaded)"

        rich_query = SERPAPI_DIRECT_PROMPT.format(
            question=user_text,
            project_context=project_context,
            agenda_block=agenda_block,
            client_profile_block=client_profile_block,
            conversation_block=conversation_block,
            length=length,
        )

        try:
            web = self._get_web_search()
            # No max_length cap — the structured prompt is built carefully
            # and Brave AI Mode handles long context well. Truncation was
            # cutting off the "HOW TO ANSWER" instructions and breaking
            # response quality.
            raw_response = await web.search_raw(rich_query, max_length=99999)
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
                    max_tokens=120,  # Bumped from 35 — fits 3-4 sentences (30-50 words)
                ),
                timeout=1.5,  # Bumped from 1.0s — longer fillers need more Groq time
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
            print(f"[Agent] ⏱ Dynamic filler timeout (>1500ms)")
            return None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[Agent] ⚠️  Dynamic filler error: {type(e).__name__}: {e}")
            return None

    @staticmethod
    def _is_valid_filler(text: str) -> bool:
        """Validate dynamic filler output before using for TTS.

        Bumped limits for 3-4 sentence fillers (~7s of audio):
        - char range: 30-350 (was 10-150 for one-sentence fillers)
        - word range: 8-70 (was 3-30 for one-sentence fillers)
        """
        if not text or not text.strip():
            return False
        text = text.strip()
        if len(text) > 350 or len(text) < 30:
            return False
        words = text.split()
        if len(words) < 8 or len(words) > 70:
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
            "find", "grab", "digging", "dig", "moment", "want to",
            "make sure", "real quick", "just a", "hang on",
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

    async def rewrite_query(self, user_text: str, conversation: str) -> str:
        """Rewrite a vague user question into a self-contained search query.

        Used for retrieval against the research journal. Runs in parallel
        with EOT and addressee detector — fired from _handle_addressee_decision.

        Inputs:
          user_text:    the raw user question
          conversation: last 5 turns formatted as "Speaker: text" lines

        Returns the rewritten question, or the original on failure (graceful).
        Latency: ~250-400ms on llama-3.1-8b-instant.
        """
        import os as _os
        import time as _t

        # Runtime kill switch
        if _os.environ.get("QUERY_REWRITER_ENABLED", "1").strip() == "0":
            return user_text

        if not user_text or not user_text.strip():
            return user_text

        prompt = QUERY_REWRITER_PROMPT.format(
            conversation=conversation or "(start of conversation)",
            question=user_text.strip(),
        )

        t0 = _t.time()
        try:
            resp = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=80,
                ),
                timeout=2.5,  # Stage 2.9: bumped from 1.5 — see warmup below
            )
            rewritten = (resp.choices[0].message.content or "").strip()
            # Strip any surrounding quotes the model sometimes adds
            for ch in ('"', "'", "“", "”", "‘", "’"):
                rewritten = rewritten.strip(ch)
            rewritten = rewritten.strip()

            if not rewritten or len(rewritten) > 300:
                # Sanity check failed — return raw
                print(f"[Agent] ⚠️  Rewriter output rejected (len={len(rewritten)}) — using raw")
                return user_text

            elapsed = (_t.time() - t0) * 1000
            if rewritten.lower() != user_text.strip().lower():
                print(f"[Agent] ✏️  Rewrote ({elapsed:.0f}ms): "
                      f"'{user_text[:40]}' → '{rewritten[:60]}'")
            else:
                print(f"[Agent] ✏️  Rewriter: unchanged ({elapsed:.0f}ms)")
            return rewritten

        except asyncio.TimeoutError:
            print(f"[Agent] ⚠️  Rewriter timeout (>2.5s) — using raw query")
            return user_text
        except Exception as e:
            print(f"[Agent] ⚠️  Rewriter error ({type(e).__name__}: {e}) — using raw")
            return user_text

    async def unified_synthesis_to_queue(
        self,
        user_text: str,
        project_context: str,
        web_results: list,
        agenda_block: str,
        client_profile_block: str,
        conversation: str,
        memory_context: str,
        azure_extractor,
        queue,  # asyncio.Queue
        length: str = "70-90 words across 4-5 sentences",
    ):
        """Stage 2 single-prompt synthesis path for the unified research architecture.

        Reuses RESEARCH_SYNTHESIS_PROMPT (the structured Exa-style template) for
        ALL question types — feasibility, tech_switch, best_practices, jira_status,
        internal_org, general. Replaces both stream_research_to_queue (legacy) and
        exa_stream_synthesis_to_queue (Exa-only) — both become dead code in Stage 4.

        Always-rendered sections (even when empty):
          - LIVE JIRA TICKETS  (project_context, never blank)
          - MEETING AGENDA     (agenda_block, never blank)
          - CLIENT PROFILE     (client_profile_block, never blank — closes the
                                Tom / Rohan / Apollo CEO hallucination class)
          - WEB SEARCH RESULTS (web_results=[] renders "(no web search this turn)")
          - CONVERSATION SO FAR

        memory_context (optional) is prepended to the system prompt.

        Behavior on Azure failure:
          - 5xx, timeout, transport error → Groq llama-3.3-70b-versatile fallback
          - Both fail → one-line apology + None to queue. Never raises.
        """
        import time as _t
        import json as _json
        import httpx as _httpx

        t0 = _t.time()

        # ── Format web_results into a labeled WEB SEARCH RESULTS block ──
        # Accepts list of dicts {title, url, content, published_date} from
        # _search_with_fallback (Exa native shape; Brave wrapped to match).
        if not web_results:
            web_results_text = "(no web search this turn)"
        else:
            lines = []
            for i, r in enumerate(web_results[:8], 1):
                title = (r.get("title") or "").strip()
                content = (r.get("content") or r.get("snippet") or "").strip()
                published = (r.get("published_date") or "").strip()
                if not content:
                    continue
                line = f"[{i}]"
                if title:
                    line += f" {title}"
                if published:
                    line += f" ({published})"
                line += f":\n{content}"
                lines.append(line)
            web_results_text = "\n\n".join(lines) if lines else "(no usable web results)"

        # ── Conversation block — last 3 turns ──
        if conversation:
            conv_lines = [
                l for l in conversation.strip().split("\n") if l.strip()
            ][-3:]
            conversation_block = "\n".join(conv_lines) if conv_lines else "(start of call)"
        else:
            conversation_block = "(start of call)"

        # ── Defaults so prompt sections never render blank ──
        if not project_context:
            project_context = "- (no specific project context available)"
        if not agenda_block:
            agenda_block = "(no fixed agenda for today)"
        if not client_profile_block:
            client_profile_block = "(no client profile loaded)"

        # ── Build the full system prompt ──
        system = RESEARCH_SYNTHESIS_PROMPT.format(
            question=user_text,
            project_context=project_context,
            agenda_block=agenda_block,
            client_profile_block=client_profile_block,
            web_search_results=web_results_text,
            conversation_block=conversation_block,
            length=length,
        )

        # Prepend memory context (from prior meetings) if present
        if memory_context:
            system = memory_context + "\n\n" + system

        print(f"[Agent] 🔬 Unified-v2 synthesis prompt: "
              f"{len(system)} chars (web={len(web_results_text)} chars, "
              f"profile={len(client_profile_block)} chars, "
              f"context={len(project_context)} chars)")

        self.history.append({"role": "user", "content": user_text})
        if len(self.history) > 6:
            self.history = self.history[-6:]

        # ── Path A: Azure unavailable (disabled or no extractor) → Groq fallback ──
        azure_unavailable = (not azure_extractor) or (
            not getattr(azure_extractor, "enabled", False)
        )

        if azure_unavailable:
            print("[Agent] ⚠️  Azure unavailable for unified-v2 synthesis — using Groq")
            await self._stream_groq_synthesis_to_queue(system, user_text, queue)
            return

        # ── Path B: Azure 4o-mini streaming with Groq 5xx fallback ──
        url = (
            f"{azure_extractor.endpoint}/openai/deployments/"
            f"{azure_extractor.deployment}/chat/completions"
            f"?api-version={azure_extractor.api_version}"
        )

        azure_failed = False

        try:
            async with _httpx.AsyncClient(timeout=60) as client:
                async with client.stream(
                    "POST", url,
                    headers={"api-key": azure_extractor.api_key,
                             "Content-Type": "application/json"},
                    json={
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_text},
                        ],
                        "temperature": 0.7,
                        "max_tokens": 250,
                        "stream": True,
                    }
                ) as response:

                    if response.status_code >= 500:
                        azure_failed = True
                        print(f"[Agent] ⚠️  Azure {response.status_code} — falling back to Groq")
                    else:
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
                                chunk = _json.loads(data)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                token = delta.get("content", "")
                                if not token:
                                    continue

                                if not first_token:
                                    first_token = True
                                    print(f"[Agent] ⏱ UNIFIED-v2 first token: "
                                          f"{(_t.time()-t0)*1000:.0f}ms")

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
                                                print(f"[Agent] ⏱ UNIFIED-v2 sentence "
                                                      f"{sentence_count}: "
                                                      f"{(_t.time()-t0)*1000:.0f}ms")
                                                await queue.put(sentence)
                                            break
                            except (_json.JSONDecodeError, IndexError, KeyError):
                                continue

                        if buffer.strip():
                            await queue.put(buffer.strip())

                        self.history.append({"role": "assistant", "content": full_response})
                        await queue.put(None)
                        return

        except _httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                azure_failed = True
                print(f"[Agent] ⚠️  Azure {e.response.status_code} — falling back to Groq")
            else:
                print(f"[Agent] ⚠️  Azure {e.response.status_code} — non-retryable")
                await queue.put("Sorry, I couldn't process that right now.")
                await queue.put(None)
                return
        except (_httpx.TimeoutException, _httpx.TransportError) as e:
            azure_failed = True
            print(f"[Agent] ⚠️  Azure transport error — falling back to Groq: "
                  f"{type(e).__name__}")
        except __import__("asyncio").CancelledError:
            raise
        except Exception as e:
            print(f"[Agent] ⚠️  Azure unexpected error: {type(e).__name__}: {e}")
            await queue.put("Sorry, I couldn't process that right now.")
            await queue.put(None)
            return

        # ── Groq fallback (Azure failed) ──
        if azure_failed:
            await self._stream_groq_synthesis_to_queue(system, user_text, queue)

    async def _stream_groq_synthesis_to_queue(self, system: str, user_text: str, queue):
        """Groq llama-3.3-70b synthesis fallback used when Azure is down.

        Sentence-streams to the queue. Pushes None when done. Never raises.
        Used by unified_synthesis_to_queue for both the "Azure disabled" and
        the "Azure 5xx/transport error" paths.
        """
        try:
            stream = await self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.7,
                max_tokens=250,
                stream=True,
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
            print(f"[Agent] ⚠️  Groq synthesis fallback failed: {e}")
            await queue.put("Sorry, I couldn't process that right now.")
        await queue.put(None)

    async def stream_research_to_queue(self, user_text: str, jira_context: str,
                                        related_tickets: str, web_results: str,
                                        jira_action: str, conversation: str,
                                        azure_extractor, queue: asyncio.Queue,
                                        memory_context: str = ""):
        """Stream RESEARCH response from Azure 4o-mini with Jira + web data.
        Handles all non-PM queries: facts, Jira, feasibility, analysis.

        memory_context: optional conversation memory from past meetings (Feature 4).
        Prepended to system prompt when present.
        """
        import time as _t
        import httpx

        system = RESEARCH_PROMPT.format(
            jira_context=jira_context or "(no project context loaded)",
            related_tickets=related_tickets or "(no related tickets found)",
            jira_action=jira_action or "(no action taken)",
            web_results=web_results or "(no web results)",
            conversation=conversation or "(start of call)",
        )

        # Prepend client profile (from "Know About Them" UI flow) so research
        # responses are grounded in the client's actual business — not generic.
        client_block = self._get_client_profile_block()
        if client_block:
            system = client_block + "\n\n" + system

        # Prepend memory context if present (from prior meetings with this group)
        if memory_context:
            system = memory_context + "\n\n" + system

        self.history.append({"role": "user", "content": user_text})
        if len(self.history) > 6:
            self.history = self.history[-6:]

        # Use Azure 4o-mini for high-quality synthesis
        if not azure_extractor or not azure_extractor.enabled:
            print("[Agent] ⚠️  Azure unavailable — falling back to Groq")
            try:
                stream = await self.client.chat.completions.create(
                    model=self.model,
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
                print(f"[Agent] ⚠️  Groq fallback failed: {e}")
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
