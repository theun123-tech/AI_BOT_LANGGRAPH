"""
server.py — Entry point with JWT auth + Output Media audio + multi-session
"""

import asyncio
import os
import time
import json
import hashlib
import hmac
import base64
import uuid
from aiohttp import web
from dotenv import load_dotenv

load_dotenv()

import storage as session_store
_saved = session_store.load_settings()
_env_map = {"jira_url": "JIRA_BASE_URL", "jira_email": "JIRA_EMAIL", "jira_token": "JIRA_API_TOKEN", "jira_project": "JIRA_DEFAULT_PROJECT", "azure_endpoint": "AZURE_ENDPOINT", "azure_key": "AZURE_API_KEY", "azure_deployment": "AZURE_DEPLOYMENT", "simli_api_key": "SIMLI_API_KEY", "simli_face_id": "SIMLI_FACE_ID"}
for _k, _env in _env_map.items():
    _v = _saved.get(_k, "")
    if _v and not os.environ.get(_env):
        os.environ[_env] = _v

from websocket_server import WebSocketServer
from external_apis import RecallBot

PORT = int(os.environ.get("PORT", 8000))
USE_OUTPUT_MEDIA = os.environ.get("USE_OUTPUT_MEDIA", "true").lower() in ("1", "true", "yes")
SIMLI_API_KEY = os.environ.get("SIMLI_API_KEY", "").strip()
SIMLI_FACE_ID = os.environ.get("SIMLI_FACE_ID", "").strip()

JWT_SECRET = os.environ.get("JWT_SECRET", "change-me-in-production-please")
JWT_EXPIRY = 24 * 3600

USERS = {}
admin_user = os.environ.get("ADMIN_USERNAME", "admin")
admin_pass = os.environ.get("ADMIN_PASSWORD", "admin123")
USERS[admin_user] = admin_pass
for i in range(1, 11):
    name = os.environ.get(f"USER_{i}_NAME", "").strip()
    pwd = os.environ.get(f"USER_{i}_PASS", "").strip()
    if name and pwd:
        USERS[name] = pwd
print(f"[Auth] {len(USERS)} user(s) configured")

def _b64url_encode(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def _b64url_decode(s):
    s += "=" * (4 - len(s) % 4) if len(s) % 4 else ""
    return base64.urlsafe_b64decode(s)

def jwt_encode(payload):
    h = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    p = _b64url_encode(json.dumps(payload).encode())
    sig = hmac.new(JWT_SECRET.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
    return f"{h}.{p}.{_b64url_encode(sig)}"

def jwt_decode(token):
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        h, p, s = parts
        expected = hmac.new(JWT_SECRET.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64url_decode(s)):
            return None
        payload = json.loads(_b64url_decode(p))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None

def _get_user(request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    return jwt_decode(auth[7:])

active_bots = {}
active_server = None
_start_time = time.time()


async def handle_login(request):
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    username = data.get("username", "").strip()
    password = data.get("password", "")
    if username not in USERS or USERS[username] != password:
        return web.json_response({"error": "Invalid credentials"}, status=401)
    token = jwt_encode({"sub": username, "iat": int(time.time()), "exp": int(time.time()) + JWT_EXPIRY})
    return web.json_response({"token": token, "username": username})


async def handle_start(request):
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    meeting_url = data.get("meeting_url", "").strip()
    if not meeting_url:
        return web.json_response({"error": "meeting_url required"}, status=400)

    mode = data.get("mode", "client_call")
    username = user["sub"]

    # Phase 2+3: optional setup data (agenda/tickets/scope) from UI
    setup_data = data.get("setup") or {}

    if username in active_bots:
        try:
            old = active_bots[username]
            await old["bot"].leave()
            await active_server.remove_session(old["session_id"])
        except Exception:
            pass
        active_bots.pop(username, None)

    domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "") or os.environ.get("RENDER_EXTERNAL_HOSTNAME", "")
    tunnel = os.environ.get("TUNNEL_URL", "").strip().rstrip("/")
    session_id = str(uuid.uuid4())

    if domain:
        base_url = f"https://{domain}"
        ws_base = f"wss://{domain}"
    elif tunnel:
        base_url = tunnel
        ws_base = tunnel.replace("https://", "wss://").replace("http://", "ws://")
    else:
        return web.json_response({"error": "No public URL configured."}, status=400)

    ws_url = f"{ws_base}/ws/{session_id}"
    audio_page_url = None
    if USE_OUTPUT_MEDIA:
        audio_page_url = f"{base_url}/audio-page?session={session_id}"
        # Simli avatar: check saved settings first, then env vars
        saved = session_store.load_settings()
        simli_key = saved.get("simli_api_key", "") or SIMLI_API_KEY
        simli_face = saved.get("simli_face_id", "") or SIMLI_FACE_ID
        simli_on = saved.get("simli_enabled", False)
        print(f"[Server] 🎭 Simli check: enabled={simli_on} ({type(simli_on).__name__}), key={'✅' if simli_key else '❌'}, face={'✅' if simli_face else '❌'}")
        if simli_on and simli_key and simli_face:
            audio_page_url += f"&simli_key={simli_key}&face_id={simli_face}"

    print(f"[Server] {username} → deploying Sam to {meeting_url}")
    print(f"[Server] Session: {session_id[:12]}, WS: {ws_url}")
    if audio_page_url:
        print(f"[Server] Audio page: {audio_page_url}")

    try:
        session = active_server.create_session(session_id, bot_id="pending")
        session.username = username
        session.meeting_url = meeting_url
        session.mode = mode
        session.started_at = time.time()
        # Phase 2+3: stash setup BEFORE setup() — DialogueManager reads it there
        session._meeting_setup = setup_data if isinstance(setup_data, dict) else {}
        if session._meeting_setup:
            try:
                session_store.save_meeting_setup(session_id, session._meeting_setup)
            except Exception as e:
                print(f"[Server] ⚠️  save_meeting_setup failed (non-fatal): {e}")
        await session.setup()

        bot = RecallBot(session_id=session_id)
        bot_id = await bot.join(
            meeting_url, ws_url,
            audio_page_url=audio_page_url,
            use_output_media=USE_OUTPUT_MEDIA,
            mode=mode,
        )
        session.bot_id = bot_id
        session.speaker.bot_id = bot_id

        active_bots[username] = {
            "bot": bot, "bot_id": bot_id, "session_id": session_id,
            "meeting_url": meeting_url, "started_at": time.time(),
        }
        return web.json_response({"status": "joined", "bot_id": bot_id, "session_id": session_id,
                                   "streaming": USE_OUTPUT_MEDIA and audio_page_url is not None})
    except Exception as e:
        await active_server.remove_session(session_id)
        print(f"[Server] Join failed: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_stop(request):
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    username = user["sub"]
    if username not in active_bots:
        return web.json_response({"status": "no active bot"})
    info = active_bots[username]

    # Phase 6 step 2: deliver end-of-meeting recap before leaving if not yet done
    session_id = info.get("session_id", "")
    session = active_server.sessions.get(session_id) if active_server else None
    if session is not None:
        dm = getattr(session, "_dialogue_manager", None)
        speak_recap = getattr(session, "_speak_recap", None)
        if dm is not None and speak_recap is not None:
            try:
                snap = dm.get_state_snapshot()
                not_delivered = not snap.get("recap_delivered", False)
                no_error = "error" not in snap
                has_content = bool(
                    snap.get("commitments_open")
                    or snap.get("topics_resolved")
                    or snap.get("open_questions")
                )
                if not_delivered and no_error and has_content:
                    print(f"[Stop] Delivering recap before leave...")
                    try:
                        await asyncio.wait_for(speak_recap(), timeout=45.0)
                    except asyncio.TimeoutError:
                        print(f"[Stop] Recap timed out (45s), leaving anyway")
                    except Exception as e:
                        print(f"[Stop] Recap error: {e}")
            except Exception as e:
                print(f"[Stop] Pre-leave state check failed: {e}")

    try:
        await info["bot"].leave()
    except Exception:
        pass
    try:
        await active_server.remove_session(info["session_id"])
    except Exception:
        pass
    active_bots.pop(username, None)
    return web.json_response({"status": "left"})


async def handle_status(request):
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    info = active_bots.get(user["sub"])
    if info:
        session = active_server.sessions.get(info["session_id"])
        streaming = session._streaming_mode if session else False
        return web.json_response({
            "active": True, "bot_id": info["bot_id"], "session_id": info["session_id"],
            "meeting_url": info["meeting_url"], "uptime_seconds": int(time.time() - info["started_at"]),
            "streaming": streaming,
        })
    return web.json_response({"active": False})


async def handle_health(request):
    return web.json_response({"status": "ok", "active_bots": len(active_bots), "uptime": int(time.time() - _start_time)})


async def handle_sessions(request):
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    return web.json_response({"sessions": session_store.get_sessions(limit=50, user=user["sub"])})


async def handle_session_detail(request):
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    detail = session_store.get_session_detail(request.match_info.get("session_id", ""))
    if not detail or detail.get("user") != user["sub"]:
        return web.json_response({"error": "Not found"}, status=404)
    return web.json_response(detail)


async def handle_settings_get(request):
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    s = session_store.load_settings()
    return web.json_response({
        "jira": {"configured": bool(s.get("jira_url") and s.get("jira_email")), "url": s.get("jira_url", ""), "email": s.get("jira_email", ""), "project": s.get("jira_project", ""), "sprint": s.get("jira_sprint", "")},
        "azure": {"configured": bool(s.get("azure_endpoint")), "endpoint": s.get("azure_endpoint", ""), "deployment": s.get("azure_deployment", "")},
        "simli": {"enabled": s.get("simli_enabled", False), "face_id": s.get("simli_face_id", "")},
        "output_media": USE_OUTPUT_MEDIA,
    })


async def handle_settings_save(request):
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    current = session_store.load_settings()
    for key in ["jira_url", "jira_email", "jira_token", "jira_project", "jira_sprint",
                 "azure_endpoint", "azure_key", "azure_deployment",
                 "simli_api_key", "simli_face_id"]:
        if key in data and data[key]:
            current[key] = data[key].strip()
    # Handle boolean toggle for simli_enabled
    if "simli_enabled" in data:
        current["simli_enabled"] = bool(data["simli_enabled"])
    print(f"[Settings] 🎭 Saving simli_enabled={current.get('simli_enabled')} (from_request={'simli_enabled' in data}, raw={data.get('simli_enabled')})")
    session_store.save_settings(current)
    return web.json_response({"ok": True})


async def handle_jira_test(request):
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    try:
        from external_apis import JiraClient
        # Read credentials from POST body (UI form). If body is empty or
        # missing fields, JiraClient falls back to env vars per field.
        try:
            body = await request.json()
        except Exception:
            body = {}
        jira = JiraClient(
            base_url=body.get("base_url"),
            email=body.get("email"),
            token=body.get("token"),
            project=body.get("project"),
        )
        if not jira.enabled:
            return web.json_response({"ok": False, "error": "Not configured (need base_url, email, and token)"})
        ok = await jira.test_connection()
        await jira.close()
        return web.json_response({"ok": ok, "message": f"Connected to {jira.base_url}" if ok else "Failed"})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)})


async def handle_jira_projects(request):
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    try:
        from external_apis import JiraClient
        # If the UI POSTs credentials, use them. Otherwise (legacy GET, or
        # POST with empty body) fall through to env vars.
        try:
            body = await request.json() if request.method == "POST" else {}
        except Exception:
            body = {}
        jira = JiraClient(
            base_url=body.get("base_url"),
            email=body.get("email"),
            token=body.get("token"),
            project=body.get("project"),
        )
        projects = await jira.get_projects() if jira.enabled else []
        await jira.close()
        return web.json_response({"projects": projects})
    except Exception as e:
        return web.json_response({"projects": [], "error": str(e)})


async def handle_jira_sprints(request):
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    try:
        from external_apis import JiraClient
        jira = JiraClient()
        sprints = await jira.get_sprints(project_key=request.query.get("project", jira.project)) if jira.enabled else []
        await jira.close()
        return web.json_response({"sprints": sprints})
    except Exception as e:
        return web.json_response({"sprints": [], "error": str(e)})


async def handle_pending_get(request):
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    pending = session_store.get_pending_tickets()
    return web.json_response({"pending": pending, "count": len(pending)})


async def handle_pending_sync(request):
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    pending = session_store.get_pending_tickets()
    if not pending:
        return web.json_response({"synced": 0})
    from external_apis import JiraClient
    jira = JiraClient()
    synced = 0
    for item in pending:
        try:
            await jira.create_ticket(summary=item.get("summary", ""), issue_type=item.get("type", "Task"), priority=item.get("priority", "Medium"), description=item.get("description", ""), labels=item.get("labels", []))
            synced += 1
        except Exception:
            break
    await jira.close()
    if synced > 0:
        session_store.clear_pending_tickets()
    return web.json_response({"synced": synced})


async def handle_audio_page(request):
    """Serve the Output Media audio page."""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_page.html")
    if os.path.exists(html_path):
        resp = web.FileResponse(html_path)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp
    return web.Response(text="audio_page.html not found", status=404)


async def handle_standups_today(request):
    """Get all team standups for today (PM dashboard)."""
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    date = request.query.get("date", None)
    standups = session_store.get_team_standups(date=date)
    blocker_count = sum(s.get("blocker_count", 0) for s in standups)
    completed = sum(1 for s in standups if s.get("completed"))
    return web.json_response({
        "date": date or time.strftime("%Y-%m-%d", time.gmtime()),
        "standups": standups,
        "total": len(standups),
        "completed": completed,
        "blocker_count": blocker_count,
    })


async def handle_standup_detail(request):
    """Get full standup detail for a specific developer."""
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    developer = request.match_info.get("developer", "")
    date = request.query.get("date", None)
    detail = session_store.get_standup_detail(developer, date)
    if not detail:
        return web.json_response({"error": "Not found"}, status=404)
    return web.json_response(detail)


async def handle_index(request):
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if os.path.exists(html_path):
        resp = web.FileResponse(html_path)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp
    return web.Response(text="index.html not found", status=404)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2+3: Setup / Templates / Jira search / Debug endpoints
# ══════════════════════════════════════════════════════════════════════════════

async def handle_meeting_setup_save(request):
    """POST /api/meeting_setup/save — save setup for an upcoming session."""
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    session_id = (data.get("session_id") or "").strip()
    setup = data.get("setup") or {}
    if not session_id:
        return web.json_response({"error": "session_id required"}, status=400)
    try:
        session_store.save_meeting_setup(session_id, setup)
        return web.json_response({"ok": True})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_meeting_setup_get(request):
    """GET /api/meeting_setup/{session_id} — fetch saved setup."""
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    session_id = request.match_info.get("session_id", "")
    setup = session_store.get_meeting_setup(session_id)
    if setup is None:
        return web.json_response({"setup": None})
    return web.json_response({"setup": setup})


async def handle_agenda_templates_list(request):
    """GET /api/agenda_templates — list caller's templates."""
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    try:
        templates = session_store.get_agenda_templates(user["sub"])
        return web.json_response({"templates": templates})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_agenda_template_save(request):
    """POST /api/agenda_templates/save — save/update a template."""
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    name = (data.get("name") or "").strip()
    if not name:
        return web.json_response({"error": "name required"}, status=400)
    template = {
        "name": name,
        "agenda": data.get("agenda") or [],
        "scope_in": data.get("scope_in") or [],
        "scope_out": data.get("scope_out") or [],
        "ticket_keys": data.get("ticket_keys") or [],
    }
    try:
        session_store.save_agenda_template(user["sub"], name, template)
        return web.json_response({"ok": True, "template": template})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_agenda_template_delete(request):
    """DELETE /api/agenda_templates/{name}"""
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    name = request.match_info.get("name", "")
    if not name:
        return web.json_response({"error": "name required"}, status=400)
    try:
        ok = session_store.delete_agenda_template(user["sub"], name)
        return web.json_response({"ok": ok})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_jira_search(request):
    """GET /api/jira/search?q=... — live autocomplete search."""
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    query = request.query.get("q", "").strip()
    if not query:
        return web.json_response({"results": []})
    from external_apis import JiraClient
    jira = JiraClient()
    if not jira.enabled:
        return web.json_response({"results": [], "error": "Jira not configured"})
    try:
        # JiraClient.__init__ already creates _client when enabled — no warmup needed
        results = await jira.search_tickets(query, max_results=10)
        return web.json_response({"results": results})
    except Exception as e:
        print(f"[Server] \u26a0\ufe0f  jira_search failed: {type(e).__name__}: {e}")
        return web.json_response({"results": [], "error": str(e)})
    finally:
        try:
            await jira.close()
        except Exception:
            pass


async def handle_prior_context(request):
    """GET /api/prior_context?participants=a,b,c — fetch Feature 4 Memory summaries."""
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    participants_raw = request.query.get("participants", "")
    participants = [p.strip() for p in participants_raw.split(",") if p.strip()]
    if not participants:
        return web.json_response({"summaries": [], "conversation_id": None})
    import hashlib
    names = sorted(a.strip().lower() for a in participants if a.strip())
    canonical = "|".join(names)
    conv_id = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    try:
        summaries = session_store.get_conversation_summaries(conv_id, limit=3)
        return web.json_response({
            "summaries": summaries,
            "conversation_id": conv_id,
            "participants_normalized": names,
        })
    except Exception as e:
        return web.json_response({"summaries": [], "error": str(e)})


async def handle_dialogue_state(request):
    """GET /api/dialogue_state/{session_id} — snapshot of live state (debug).

    Returns DialogueManager state if USE_DIALOGUE_MANAGER=1 and session has one,
    else {"error": "not active"}.
    """
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    session_id = request.match_info.get("session_id", "")
    session = active_server.sessions.get(session_id) if active_server else None
    if not session:
        return web.json_response({"error": "session not found"}, status=404)
    dm = getattr(session, "_dialogue_manager", None)
    if dm is None:
        return web.json_response({"error": "DialogueManager not active for this session"})
    try:
        snapshot = dm.get_state_snapshot()
        return web.json_response({"state": snapshot})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_commitments(request):
    """GET /api/commitments/{session_id} — Phase 6 step 1.

    Returns the list of commitments captured by DialogueManager
    for this session. Each entry includes owner, action, deadline,
    and the turn where it was detected.

    Response shape:
      {"commitments": [{"owner": "Sahil", "action": "review SCRUM-244",
                        "deadline": "Friday", "confidence": 0.95,
                        "turn_number": 12, "status": "open"}, ...]}
    """
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    session_id = request.match_info.get("session_id", "")
    session = active_server.sessions.get(session_id) if active_server else None
    if not session:
        return web.json_response({"commitments": [], "error": "session not found"})
    dm = getattr(session, "_dialogue_manager", None)
    if dm is None:
        return web.json_response({"commitments": [], "error": "DialogueManager not active"})
    try:
        snapshot = dm.get_state_snapshot()
        # Phase 6 step 1 hotfix 2: state uses "commitments_open" / "commitments_resolved"
        # / "commitments_inherited" (see meeting_state.py). Flatten all of them
        # with a status tag so the frontend can filter/group.
        items = []
        for field_name, default_status in [
            ("commitments_open", "open"),
            ("commitments_resolved", "resolved"),
            ("commitments_inherited", "inherited"),
        ]:
            raw_list = snapshot.get(field_name) or []
            for c in raw_list:
                if not isinstance(c, dict):
                    continue
                items.append({
                    "owner": str(c.get("owner", "") or ""),
                    "action": str(c.get("action", "") or ""),
                    "deadline": c.get("deadline"),
                    "confidence": c.get("confidence"),
                    "turn_number": c.get("turn_number"),
                    "status": str(c.get("status") or default_status),
                })
        return web.json_response({"commitments": items})
    except Exception as e:
        return web.json_response({"commitments": [], "error": str(e)})


def _clean_profile_markdown(text: str) -> str:
    """Strip markdown formatting and inline citations from a profile string.

    Google AI Mode often returns reconstructed_markdown with:
      - Inline links: [text](url)
      - Headers: ### Heading (sometimes mid-line)
      - Bullets: - item (sometimes mid-line)
      - Citation markers: [0], [1], [^1]
      - A trailing "### References" section
      - Backslash-escaped chars: \\-, \\(, \\)

    The text often arrives as a single long line with `###` and `-` markers
    embedded mid-string, so we first split on those markers to give the
    line-based regexes something to work with.

    Voice/UI use needs plain prose — no markdown, no URLs. This converts to
    natural readable text suitable for both display in the textarea and
    injection into Sam's system prompt.

    Idempotent: running twice is safe (does nothing on the second pass).
    """
    import re as _re_local

    if not text:
        return ""
    s = text

    # 1. Drop trailing References section (everything from "### References"
    #    or "References:" on, regardless of whether it's at line start)
    s = _re_local.sub(
        r"\s*#{1,6}\s*References\b.*$",
        "",
        s,
        flags=_re_local.IGNORECASE | _re_local.DOTALL,
    )
    s = _re_local.sub(
        r"\s*\bReferences\s*:.*$",
        "",
        s,
        flags=_re_local.IGNORECASE | _re_local.DOTALL,
    )

    # 2. Convert inline markdown links [text](url) → just "text"
    s = _re_local.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)

    # 3. Drop bare citation markers like [0], [12], [^3]
    s = _re_local.sub(r"\[\^?\d+\]", "", s)

    # 4. Insert paragraph break BEFORE every "### Heading" marker, then
    #    strip the marker. This handles mid-line headers gracefully.
    s = _re_local.sub(r"\s*#{1,6}\s+", "\n\n", s)

    # 5. Insert sentence break BEFORE every " - Bullet" marker (mid-line
    #    bullets common in Google's reconstructed_markdown), then strip.
    #    Only matches when preceded by space/period (not start of word).
    s = _re_local.sub(r"(?:^|\s)[-*•]\s+", " ", s)

    # 6. Strip remaining bold/italic markers
    s = _re_local.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = _re_local.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", s)

    # 7. Drop "Field name:" labels that came from bulleted lists
    #    "Role & Expertise: He is..." → "He is..."
    #    Carefully limited to short capitalized labels at sentence boundaries
    s = _re_local.sub(
        r"(^|\.\s+|\n\n)([A-Z][A-Za-z0-9 &/()+\-]{2,40}):\s+",
        r"\1",
        s,
    )

    # 8. Unescape common backslash-escaped punctuation
    s = s.replace("\\(", "(").replace("\\)", ")")
    s = s.replace("\\-", "-").replace("\\+", "+")
    s = s.replace("\\&", "&").replace("\\.", ".")
    s = s.replace("\\,", ",").replace("\\:", ":")
    s = s.replace("\\|", "|")

    # 9. Collapse excess whitespace, normalize line breaks
    s = "\n".join(line.strip() for line in s.split("\n"))
    s = _re_local.sub(r"\n{2,}", "\n\n", s)

    # 10. Within paragraphs, single newline → space (so flowing prose isn't
    #     awkwardly split). Preserve double newlines as paragraph breaks.
    paragraphs = s.split("\n\n")
    paragraphs = [_re_local.sub(r"\s*\n\s*", " ", p).strip() for p in paragraphs]
    paragraphs = [p for p in paragraphs if p]
    s = "\n\n".join(paragraphs)

    # 11. Final whitespace cleanup
    s = _re_local.sub(r"[ \t]{2,}", " ", s).strip()

    return s


async def handle_clients_research(request):
    """POST /api/clients/research — fetch a client/company profile via SerpAPI.

    Body: {"client_names": "alice, bob", "company_names": "Acme, Foo Inc"}

    Strategy (3 layers, in order):

    1. SerpAPI Google AI Mode WITH GEO GROUNDING (gl/hl/location). This is
       the critical fix: Google AI Mode behaves differently per region.
       Without these params, SerpAPI's US servers hit a variant of AI Mode
       that often skips synthesis for non-US entities (e.g. AnavClouds, an
       India-based company) and returns only citations. With "in" + "India"
       defaults, Google treats it as a local search and synthesizes properly.

       Configurable via env: SERPAPI_GL, SERPAPI_HL, SERPAPI_LOCATION.
       Uses a SHORT natural query — AI Mode synthesizes more reliably for
       natural questions than for multi-paragraph structured instructions.
       Retries up to 3 times because AI Mode is non-deterministic.

    2. If all 3 retries return references-only, hand the citation snippets
       to Azure gpt-4o-mini and have it write a coherent profile FROM those
       grounded sources. This is true LLM synthesis, not stitching, and
       it's grounded in real Google citations so factuality is preserved.

    3. If even Azure fails, return an honest warning + diagnostic so the
       user can write the profile manually.
    """
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    client_names = (data.get("client_names") or "").strip()
    company_names = (data.get("company_names") or "").strip()

    if not client_names and not company_names:
        return web.json_response(
            {"error": "Please provide at least one client name or company name"},
            status=400,
        )
    if len(client_names) > 500 or len(company_names) > 500:
        return web.json_response(
            {"error": "Input too long (500 char max per field)"},
            status=400,
        )

    # SHORT natural query — Google AI Mode synthesizes reliably for natural
    # questions. The detailed structure (2-3 sentences each, factual, etc.)
    # is moved to the Azure synthesis prompt at step 2 instead.
    parts = ["Tell me about"]
    if client_names:
        parts.append(client_names)
        if company_names:
            parts.append("from")
            parts.append(company_names)
    elif company_names:
        parts.append(company_names)
    parts.append("— who they are, what they do, and notable work.")
    short_query = " ".join(parts)

    # Detailed instructions reused for the Azure synthesis fallback step
    detailed_instructions = (
        "For each person, write 2-3 sentences covering their professional "
        "role, their connection to the company, and any notable public work "
        "or projects. For each company, write 2-3 sentences covering what "
        "the company does, founding year/location/scale, and notable "
        "initiatives. Keep it factual and concise. Plain readable prose only "
        "— no URLs, no citations, no markdown bullets, no headers. Maximum "
        "250 words total. If a specific person or company cannot be reliably "
        "identified from the source material below, say so explicitly — do "
        "not guess or invent details."
    )

    try:
        import httpx
        import json as _json
        import time as _time

        # Pick a SerpAPI key via the shared rotator. Handles all three env
        # var formats: SERPAPI_KEYS (plural, comma-separated — Stage R),
        # SERPAPI_KEY (singular), and legacy SERPAPI_KEY_1..N.
        from key_rotator import key_for_request
        serp_key = key_for_request("SERPAPI") or ""
        if not serp_key:
            return web.json_response({
                "error": "SerpAPI not configured (set SERPAPI_KEYS in your .env)"
            }, status=500)

        # Geo grounding params — control where Google AI Mode "thinks it is".
        # Defaults to India because most current clients are India-based.
        # Override via env vars if your typical clients are elsewhere.
        gl_param = (os.environ.get("SERPAPI_GL") or "in").strip()
        hl_param = (os.environ.get("SERPAPI_HL") or "en").strip()
        location_param = (os.environ.get("SERPAPI_LOCATION") or "India").strip()

        print(f"[ClientsResearch] === Starting research ===")
        print(f"[ClientsResearch] Query: \"{short_query}\"")
        print(f"[ClientsResearch] Geo: gl={gl_param}, hl={hl_param}, location={location_param}")

        # ── STAGE 1: Try SerpAPI Google AI Mode up to 3 times ──
        attempts = []
        synthesized_text = ""
        extraction_strategy = "none"
        latest_references = []
        latest_top_keys = []
        debug_files = []
        total_ms = 0

        for attempt_num in range(1, 4):
            t0 = _time.time()
            params = {
                "engine": "google_ai_mode",
                "q": short_query,
                "api_key": serp_key,
                # Location/language params help AI Mode ground properly.
                # Without these, SerpAPI's US servers can hit a variant of
                # AI Mode that skips grounding and just returns citations.
                "gl": gl_param,
                "hl": hl_param,
                "location": location_param,
            }

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(
                        "https://serpapi.com/search.json", params=params)
            except httpx.TimeoutException:
                attempts.append({"attempt": attempt_num, "outcome": "timeout"})
                print(f"[ClientsResearch] Attempt {attempt_num}: TIMEOUT")
                continue

            ms = (_time.time() - t0) * 1000
            total_ms += ms

            if resp.status_code != 200:
                attempts.append({
                    "attempt": attempt_num,
                    "outcome": f"http_{resp.status_code}",
                    "ms": int(ms),
                })
                print(f"[ClientsResearch] Attempt {attempt_num}: "
                      f"HTTP {resp.status_code} after {ms:.0f}ms")
                continue

            try:
                raw_data = resp.json()
            except Exception:
                attempts.append({"attempt": attempt_num, "outcome": "invalid_json"})
                continue

            # Save full response per attempt for inspection
            try:
                ts_str = _time.strftime("%Y%m%d-%H%M%S")
                debug_dir = "serpapi_debug"
                os.makedirs(debug_dir, exist_ok=True)
                debug_file = os.path.join(
                    debug_dir, f"clients_{ts_str}_a{attempt_num}.json")
                with open(debug_file, "w", encoding="utf-8") as f:
                    _json.dump({
                        "query": short_query,
                        "geo": {"gl": gl_param, "hl": hl_param,
                                "location": location_param},
                        "response": raw_data,
                    }, f, indent=2, ensure_ascii=False)
                debug_files.append(debug_file)
                print(f"[ClientsResearch] Attempt {attempt_num}: 💾 {debug_file}")
            except Exception:
                pass

            meta = raw_data.get("search_metadata", {}) or {}
            status = meta.get("status", "?")
            top_keys = [k for k in raw_data.keys()
                        if k not in ("search_metadata", "search_parameters")]
            latest_top_keys = top_keys
            latest_references = raw_data.get("references", []) or []

            print(f"[ClientsResearch] Attempt {attempt_num}: "
                  f"status={status}, fields={top_keys}, ms={ms:.0f}")

            if status != "Success":
                attempts.append({
                    "attempt": attempt_num,
                    "outcome": f"status_{status}",
                    "ms": int(ms),
                })
                continue

            # Try to extract synthesis — markdown first, then text_blocks
            recon = raw_data.get("reconstructed_markdown", "") or ""
            if recon and recon.strip():
                synthesized_text = recon.strip()
                extraction_strategy = "reconstructed_markdown"
                attempts.append({
                    "attempt": attempt_num,
                    "outcome": "synthesized_via_markdown",
                    "ms": int(ms),
                })
                print(f"[ClientsResearch] ✅ Attempt {attempt_num} synthesized "
                      f"({len(synthesized_text)} chars via markdown)")
                break

            text_blocks = raw_data.get("text_blocks", []) or []
            if text_blocks:
                tb_parts = []
                for b in text_blocks:
                    if not isinstance(b, dict):
                        continue
                    snip = (b.get("snippet") or b.get("text") or "").strip()
                    if snip:
                        tb_parts.append(snip)
                    for it in (b.get("list") or []):
                        if isinstance(it, dict):
                            it_snip = (it.get("snippet") or it.get("text") or "").strip()
                            if it_snip:
                                tb_parts.append(it_snip)
                if tb_parts:
                    synthesized_text = " ".join(tb_parts).strip()
                    extraction_strategy = "text_blocks"
                    attempts.append({
                        "attempt": attempt_num,
                        "outcome": "synthesized_via_text_blocks",
                        "ms": int(ms),
                    })
                    print(f"[ClientsResearch] ✅ Attempt {attempt_num} synthesized "
                          f"({len(synthesized_text)} chars via text_blocks)")
                    break

            # No synthesis on this attempt — try again
            attempts.append({
                "attempt": attempt_num,
                "outcome": "references_only",
                "ms": int(ms),
                "ref_count": len(latest_references),
            })
            print(f"[ClientsResearch] Attempt {attempt_num}: only references "
                  f"({len(latest_references)} citations) — retrying")

        # ── STAGE 2: Azure synthesis from reference snippets ──
        # Real LLM synthesis (NOT stitching) using Google's citations as
        # source material. Output is grounded in real citations so factuality
        # is preserved.
        used_azure_fallback = False
        if not synthesized_text and latest_references:
            print(f"[ClientsResearch] No synthesis after 3 attempts — "
                  f"falling back to Azure synthesis from "
                  f"{len(latest_references)} references")

            ref_lines = []
            for r in latest_references[:9]:
                if not isinstance(r, dict):
                    continue
                title = (r.get("title") or "").strip()
                src = (r.get("source") or "").strip()
                snip = (r.get("snippet") or "").strip()
                if snip:
                    if title:
                        ref_lines.append(f"- {title} ({src}): {snip}")
                    else:
                        ref_lines.append(f"- {snip}")
            ref_block = "\n".join(ref_lines)

            azure_endpoint = (os.environ.get("AZURE_ENDPOINT", "") or "").strip().rstrip("/")
            azure_key = (os.environ.get("AZURE_API_KEY", "") or "").strip()
            azure_deployment = (os.environ.get("AZURE_DEPLOYMENT", "gpt-4o-mini") or "").strip()
            azure_api_version = (os.environ.get("AZURE_API_VERSION", "2024-02-15-preview") or "").strip()

            if azure_endpoint and azure_key and ref_block:
                synth_system = (
                    "You write factual profile summaries. The user gives you "
                    "raw search citations from Google. You write a clean "
                    "natural-prose summary based ONLY on what those citations "
                    "say — never invent details, never hallucinate dates or "
                    "numbers. If the citations don't cover something, leave "
                    "it out. Output plain prose only — no bullets, no "
                    "headers, no URLs, no citation markers like [1]."
                )
                synth_user = (
                    f"{detailed_instructions}\n\n"
                    f"Source material (Google search citations):\n{ref_block}\n\n"
                    f"Now write the profile."
                )

                url = (f"{azure_endpoint}/openai/deployments/{azure_deployment}"
                       f"/chat/completions?api-version={azure_api_version}")
                try:
                    t_az = _time.time()
                    async with httpx.AsyncClient(timeout=30.0) as ac:
                        ar = await ac.post(
                            url,
                            headers={"api-key": azure_key,
                                     "Content-Type": "application/json"},
                            json={
                                "messages": [
                                    {"role": "system", "content": synth_system},
                                    {"role": "user", "content": synth_user},
                                ],
                                "temperature": 0.3,
                                "max_tokens": 500,
                            },
                        )
                    az_ms = (_time.time() - t_az) * 1000
                    if ar.status_code == 200:
                        adata = ar.json()
                        synth = ((adata.get("choices") or [{}])[0]
                                 .get("message", {}).get("content", "") or "").strip()
                        if synth:
                            synthesized_text = synth
                            extraction_strategy = "azure_synthesis_from_references"
                            used_azure_fallback = True
                            print(f"[ClientsResearch] ✅ Azure synthesis "
                                  f"({len(synth)} chars in {az_ms:.0f}ms)")
                    else:
                        print(f"[ClientsResearch] ⚠️ Azure HTTP {ar.status_code}")
                except Exception as e:
                    print(f"[ClientsResearch] ⚠️ Azure call failed: "
                          f"{type(e).__name__}: {e}")
            else:
                print(f"[ClientsResearch] ⚠️ Azure not configured "
                      f"(endpoint={bool(azure_endpoint)}, "
                      f"key={bool(azure_key)}, refs={bool(ref_block)})")

        # ── STAGE 3: Honest failure with diagnostics ──
        if not synthesized_text:
            ref_count = len(latest_references)
            return web.json_response({
                "profile_text": "",
                "word_count": 0,
                "warning": (
                    f"Google AI Mode did not synthesize an answer "
                    f"(returned only {ref_count} citation references "
                    f"across {len(attempts)} attempts) and Azure "
                    f"fallback also failed. Try a more specific query "
                    f"(add location, sector) or write the profile manually."
                ),
                "diagnostic": {
                    "top_keys": latest_top_keys,
                    "extraction_strategy": "none",
                    "latency_ms": int(total_ms),
                    "reference_count": ref_count,
                    "attempts": attempts,
                    "debug_files": debug_files,
                },
            })

        # Clean markdown/links/headers/citations regardless of which path
        # produced the text (markdown / text_blocks / Azure synthesis).
        # Sam needs plain prose — markdown URLs and reference markers would
        # confuse both the TTS and the Agent.
        synthesized_text = _clean_profile_markdown(synthesized_text)

        # Stage 2.13 + 2.14: structured profile header.
        # Stage 2.14: drop misleading client-on-call line.
        # The "Client names" field in the UI is a research pivot — the user
        # types a public-facing person at the company (e.g. an executive,
        # founder, or named figure) so SerpAPI can find the RIGHT company
        # among many with similar names. That person is NOT necessarily on
        # the call. The actual speaker is whoever joins the meeting in
        # Recall.ai — a different layer entirely.
        #
        # Stage 2.13 wrongly assumed client_names = attendees and added a
        # "CLIENT(S) ON THE CALL: <client_names>" line. That made Sam
        # confidently mis-identify the speaker. We removed that line.
        #
        # What remains: just the COMPANY: line (clean string Sam can extract)
        # and a clarifying KEY FACTS line saying the prose is ABOUT THE
        # COMPANY, not the speaker. Names mentioned IN the prose (founders,
        # executives, public figures used as research anchors) are explicitly
        # not assumed to be on the call.
        header_lines = []
        if company_names:
            header_lines.append(f"COMPANY: {company_names}")
        if header_lines:
            header_lines.append(
                "KEY FACTS BELOW (this describes the speaker's company — "
                "use it to ground your answers about their business. Names "
                "mentioned in the description, e.g. founders or executives, "
                "are research references and are NOT necessarily on the call):"
            )
            structured_header = "\n".join(header_lines)
            synthesized_text = structured_header + "\n\n" + synthesized_text

        # Cap at 300 words (after header prepend so total length is bounded)
        words = synthesized_text.split()
        if len(words) > 300:
            synthesized_text = " ".join(words[:300]) + "..."

        return web.json_response({
            "profile_text": synthesized_text,
            "word_count": min(len(words), 300),
            "diagnostic": {
                "top_keys": latest_top_keys,
                "extraction_strategy": extraction_strategy,
                "latency_ms": int(total_ms),
                "attempts": attempts,
                "debug_files": debug_files,
                "used_azure_fallback": used_azure_fallback,
                "reference_count": len(latest_references),
            },
        })

    except httpx.TimeoutException:
        return web.json_response({"error": "Request timed out"}, status=504)
    except Exception as e:
        print(f"[ClientsResearch] ⚠️ Failed: {type(e).__name__}: {e}")
        return web.json_response({"error": f"Research failed: {e}"}, status=500)


async def main():
    global active_server
    server = WebSocketServer(port=PORT)
    active_server = server

    # When a session is removed (bot left, kicked, disconnected), clean up active_bots
    def on_session_removed(session):
        username = session.username
        if username and username in active_bots:
            active_bots.pop(username, None)
            print(f"[Server] 🔄 Bot status reset for {username} (session ended)")

    server.on_session_removed = on_session_removed

    routes = [
        ("POST", "/auth/login", handle_login),
        ("POST", "/start", handle_start),
        ("POST", "/stop", handle_stop),
        ("GET", "/status", handle_status),
        ("GET", "/api/health", handle_health),
        ("GET", "/api/sessions", handle_sessions),
        ("GET", "/api/sessions/{session_id}", handle_session_detail),
        ("GET", "/api/settings", handle_settings_get),
        ("POST", "/api/settings/save", handle_settings_save),
        ("POST", "/api/settings/jira/test", handle_jira_test),
        ("GET",  "/api/jira/projects", handle_jira_projects),
        ("POST", "/api/jira/projects", handle_jira_projects),
        ("GET", "/api/jira/sprints", handle_jira_sprints),
        ("GET", "/api/pending", handle_pending_get),
        ("POST", "/api/pending/sync", handle_pending_sync),
        ("GET", "/api/standups", handle_standups_today),
        ("GET", "/api/standups/{developer}", handle_standup_detail),
        # Phase 2+3: setup UI + templates + debug
        ("POST", "/api/meeting_setup/save", handle_meeting_setup_save),
        ("GET", "/api/meeting_setup/{session_id}", handle_meeting_setup_get),
        ("GET", "/api/agenda_templates", handle_agenda_templates_list),
        ("POST", "/api/agenda_templates/save", handle_agenda_template_save),
        ("DELETE", "/api/agenda_templates/{name}", handle_agenda_template_delete),
        ("GET", "/api/jira/search", handle_jira_search),
        ("GET", "/api/prior_context", handle_prior_context),
        ("GET", "/api/dialogue_state/{session_id}", handle_dialogue_state),
        # Phase 6 step 1: commitments visibility
        ("GET", "/api/commitments/{session_id}", handle_commitments),
        # Client research (Know About Them) — SerpAPI Google AI Mode profile fetch
        ("POST", "/api/clients/research", handle_clients_research),
        ("GET", "/audio-page", handle_audio_page),
        ("GET", "/", handle_index),
    ]
    for method, path, handler in routes:
        if method == "GET":
            server.app.router.add_get(path, handler)
        elif method == "POST":
            server.app.router.add_post(path, handler)
        elif method == "DELETE":
            server.app.router.add_delete(path, handler)

    await server.start()

    mode = "Output Media (streaming)" if USE_OUTPUT_MEDIA else "output_audio API (fallback)"
    print(f"[Server] Running on port {PORT}")
    print(f"[Server] Frontend: http://localhost:{PORT}/")
    print(f"[Server] Audio mode: {mode}")
    simli_status = f"✅ Enabled (face: {SIMLI_FACE_ID})" if SIMLI_API_KEY and SIMLI_FACE_ID else "⚠️  Disabled (no SIMLI_API_KEY or SIMLI_FACE_ID)"
    print(f"[Server] Simli avatar: {simli_status}")
    print(f"[Server] Credentials: {admin_user} / {'*' * len(admin_pass)}")

    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass

if __name__ == "__main__":
    asyncio.run(main())
