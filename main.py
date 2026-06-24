import os
import json
import base64
import secrets
import smtplib
import threading
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from email.utils import formataddr

from google import genai
from google.genai import types
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from tools import TOOLS, dispatch_tool
import db

# ─────────────────────────────────────────────
# Gemini
# ─────────────────────────────────────────────
def _get_gemini_key():
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    with open("api.json", "r") as f:
        return json.load(f)["gemini_api_key"]

client = genai.Client(api_key=_get_gemini_key())
MODEL  = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
ADMIN_PASSWORD    = os.environ.get("ADMIN_PASSWORD", "grinta123")
ZOHO_USER         = os.environ.get("ZOHO_USER", "")
ZOHO_APP_PASSWORD = os.environ.get("ZOHO_APP_PASSWORD", "")
NOTIFY_EMAIL      = os.environ.get("NOTIFY_EMAIL", "")
ADMIN_URL         = os.environ.get("ADMIN_URL", "https://grinta-agent.onrender.com/admin")

# ─────────────────────────────────────────────
# Email notification
# ─────────────────────────────────────────────
def _send_email(subject: str, body: str) -> None:
    if not (ZOHO_USER and ZOHO_APP_PASSWORD and NOTIFY_EMAIL):
        print("[email] SMTP not configured — skipping")
        return
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"]    = formataddr(("Grinta Agent", ZOHO_USER))
        msg["To"]      = NOTIFY_EMAIL
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.starttls()
            server.login(ZOHO_USER, ZOHO_APP_PASSWORD)
            server.sendmail(ZOHO_USER, [NOTIFY_EMAIL], msg.as_string())
        print("[email] sent")
    except Exception as e:
        print(f"[email] failed: {e}")


def notify_escalation(session_id: str, reason: str, summary: str) -> None:
    subject = "🔴 פנייה חדשה דורשת טיפול — Grinta"
    body = (
        "התקבלה פנייה חדשה שהבוט העביר לטיפול אנושי.\n\n"
        f"סיבה: {reason}\n"
        f"סיכום: {summary}\n\n"
        f"מזהה שיחה: {session_id}\n\n"
        f"לטיפול בפנייה: {ADMIN_URL}\n"
    )
    threading.Thread(target=_send_email, args=(subject, body), daemon=True).start()

# ─────────────────────────────────────────────
# Knowledge base
# ─────────────────────────────────────────────
def load_knowledge() -> str:
    kb = ""
    for fname in ["faq.md", "policies.md"]:
        path = Path(fname)
        if path.exists():
            kb += f"\n\n---\n\n{path.read_text(encoding='utf-8')}"
    return kb.strip()

KNOWLEDGE = load_knowledge()

SYSTEM_PROMPT = f"""You are a customer service agent for Grinta (גרינטה), an Israeli online store selling licensed football jerseys.

## Your personality
- Friendly, helpful, and professional
- Default to responding in Hebrew. If the customer writes in Arabic or English, respond in that same language instead. For any other language, respond in Hebrew.
- Be concise — no unnecessary filler text
- Never make up information you don't have
- Do NOT volunteer prices or costs unless the customer specifically asks about price or cost. For example, if asked "can I add a name and number?" answer that yes, they can add any name and number they like — do NOT mention the price. Only mention the price if they ask how much it costs. This applies to all features (printing, adding pants, adding socks, player version, etc.).

## Your knowledge
{KNOWLEDGE}

## Tool usage rules
- If a customer asks about their order, ALWAYS call get_order_by_email or get_order_by_number before responding
- Never invent order information — only report what the tools return
- If a customer provides an order number, use get_order_by_number
- If a customer provides an email, use get_order_by_email
- If neither is provided, ask the customer for their email or order number first
- For product/team/jersey/availability questions: the full product catalog is provided below under "Product catalog". Answer ONLY from that list. If a product appears in the catalog, it exists and is available. If a team/product is NOT in the catalog, we don't currently carry it — offer to check if we can source it (ask for club, season, and kit type). NEVER invent products or claim something is out of stock if it appears in the catalog. For availability of a specific SIZE, refer the customer to the size guide / product page, since the catalog lists the size range we offer, not live per-size stock.
- For cancellations: check the order, then tell the customer whether cancellation is possible based on the 24-hour policy, but explain that a human representative finalizes it. Add a note with add_order_note saying "CANCELLATION REQUESTED BY CUSTOMER"
- For returns: tell the customer whether they meet the return conditions, but explain the final approval is done by the team

## Escalation rules
ONLY escalate (call escalate_to_human) when you genuinely cannot answer or resolve something.
As long as you have an answer and are managing fine, there is NO reason to escalate.
Escalate when:
- The customer is angry or uses aggressive language
- The customer explicitly asks to speak to a human
- You truly don't have the information to help
- The request is unusual and outside everything in your knowledge

## Image handling
- If a customer sends an image of a jersey, assess if there is visible damage or defect
- If there is a clear defect, apologize and escalate to human

## Important
- Never discuss competitors
- Never promise things not in the policies
- Never share other customers' information
"""

# ─────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────
app = FastAPI(title="Grinta CS Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str = ""
    image_base64: str | None = None
    image_mime: str | None = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    escalated: bool = False
    reply_id: int = 0


@app.get("/")
def root():
    return {"status": "Grinta CS Agent is running"}


def build_history(session_id: str, skip_last_user: bool):
    msgs = db.get_messages(session_id)
    if skip_last_user and msgs and msgs[-1]["role"] == "user":
        msgs = msgs[:-1]
    history = []
    for m in msgs:
        role = "user" if m["role"] == "user" else "model"
        history.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    return history


def build_system_instruction() -> str:
    """System prompt + the live product catalog (cached, refreshed periodically)."""
    catalog = tools.get_catalog_text()
    if not catalog:
        return SYSTEM_PROMPT
    return (
        SYSTEM_PROMPT
        + "\n\n## Product catalog (Hebrew — this is the full list of products we offer)\n"
        + "כל מוצר שמופיע כאן קיים וזמין להזמנה, עם המידות והאופציות שלו. "
        + "מוצר שלא מופיע ברשימה — איננו מציעים אותו כרגע.\n\n"
        + catalog
    )


def gemini_generate(history):
    """Call Gemini with up to 3 retries on 503."""
    system_instruction = build_system_instruction()
    for attempt in range(3):
        try:
            return client.models.generate_content(
                model=MODEL,
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=TOOLS,
                )
            )
        except Exception as e:
            if "503" in str(e) and attempt < 2:
                time.sleep(3)
                continue
            raise


def run_loop(session_id: str, history):
    """Run the tool-use loop over a prepared history. Returns (text, escalated)."""
    escalated = False
    for _ in range(5):
        response = gemini_generate(history)
        content = response.candidates[0].content
        history.append(content)

        tool_calls = [p for p in content.parts if p.function_call is not None]

        if not tool_calls:
            text = "".join(p.text for p in content.parts if p.text).strip()
            return text, escalated

        tool_response_parts = []
        for part in tool_calls:
            fc   = part.function_call
            name = fc.name
            args = dict(fc.args)
            if name == "escalate_to_human":
                escalated = True
                db.set_status(session_id, "escalated", args.get("reason", ""))
                notify_escalation(session_id, args.get("reason", ""), args.get("summary", ""))
            print(f"[Tool call] {name}({args})")
            result = dispatch_tool(name, args)
            tool_response_parts.append(types.Part(
                function_response=types.FunctionResponse(
                    name=name, response={"result": result}
                )
            ))
        history.append(types.Content(role="user", parts=tool_response_parts))

    return "מצטערים, נתקלנו בבעיה טכנית. נשמח לעזור לך בדרך אחרת.", escalated


def run_bot_turn(session_id: str):
    """Generate a bot answer for the current conversation state (used after handback).
    Runs in a background thread. Saves the assistant reply so the widget can poll it."""
    try:
        history = build_history(session_id, skip_last_user=False)
        text, escalated = run_loop(session_id, history)
        # If a human took over again while we were generating, discard
        fresh = db.get_session(session_id)
        if fresh and fresh.get("status") == "escalated" and not escalated:
            return
        db.add_message(session_id, "assistant", text)
    except Exception as e:
        print(f"[run_bot_turn] error: {e}")


def _is_stale(updated_at_str: str, hours: int = 1) -> bool:
    """True if the given timestamp is older than `hours` hours."""
    if not updated_at_str:
        return False
    try:
        ts = datetime.fromisoformat(updated_at_str.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ts) > timedelta(hours=hours)
    except Exception as e:
        print(f"[stale check] {e}")
        return False


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    db.ensure_session(req.session_id)

    # Read status + last activity BEFORE the new message resets the timer
    session_before = db.get_session(req.session_id)
    was_escalated = bool(session_before and session_before.get("status") == "escalated")

    # Auto-handback: escalated but quiet for over 1 hour -> return to bot
    auto_handback = False
    if was_escalated and _is_stale(session_before.get("updated_at"), 1):
        db.set_status(req.session_id, "bot")
        auto_handback = True

    # Decode + upload image (if any) so it can be shown later in the inbox
    img_bytes = None
    image_url = None
    if req.image_base64:
        try:
            img_bytes = base64.b64decode(req.image_base64)
            image_url = db.upload_image(req.session_id, img_bytes, req.image_mime or "image/jpeg")
        except Exception as e:
            print(f"[Image error] {e}")

    stored_text = req.message or ("[התקבלה תמונה]" if req.image_base64 else "")
    db.add_message(req.session_id, "user", stored_text, image_url)

    # If a human is handling it (and it wasn't just auto-handed-back), bot stays silent
    if was_escalated and not auto_handback:
        return ChatResponse(reply="", session_id=req.session_id, escalated=True)

    history = build_history(req.session_id, skip_last_user=True)

    user_parts = []
    if img_bytes:
        user_parts.append(types.Part.from_bytes(
            data=img_bytes,
            mime_type=req.image_mime or "image/jpeg",
        ))
    user_parts.append(types.Part(text=req.message or "הלקוח שלח תמונה. בדוק אותה ועזור בהתאם."))
    history.append(types.Content(role="user", parts=user_parts))

    text, escalated = run_loop(req.session_id, history)

    # Re-check: if a human took over DURING generation, discard the bot's answer
    fresh = db.get_session(req.session_id)
    if fresh and fresh.get("status") == "escalated" and not escalated:
        return ChatResponse(reply="", session_id=req.session_id, escalated=True)

    row = db.add_message(req.session_id, "assistant", text)
    if escalated:
        db.set_status(req.session_id, "escalated")
    reply_id = (row or {}).get("id") or 0
    return ChatResponse(reply=text, session_id=req.session_id, escalated=escalated, reply_id=reply_id)


@app.get("/poll")
def poll(session_id: str, after_id: int = 0):
    """Return new assistant + human messages after a cursor (for the widget)."""
    msgs = db.get_new_messages_after(session_id, after_id)
    return {"messages": msgs}


@app.get("/history")
def history(session_id: str):
    """Return the full visible conversation for a session (widget on load)."""
    msgs = db.get_messages(session_id)
    out = []
    for m in msgs:
        if m["role"] in ("user", "assistant", "human"):
            out.append({
                "id": m["id"],
                "role": m["role"],
                "content": m["content"],
                "image_data": m.get("image_data"),
            })
    return {"messages": out}


# ─────────────────────────────────────────────
# Admin inbox
# ─────────────────────────────────────────────
security = HTTPBasic()


def check_admin(creds: HTTPBasicCredentials = Depends(security)):
    user_ok = secrets.compare_digest(creds.username, "admin")
    pass_ok = secrets.compare_digest(creds.password, ADMIN_PASSWORD)
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True


class ReplyRequest(BaseModel):
    session_id: str
    content: str


@app.get("/admin/api/sessions")
def admin_sessions(only_escalated: bool = False, _: bool = Depends(check_admin)):
    return {"sessions": db.list_sessions(only_escalated)}


@app.get("/admin/api/messages")
def admin_messages(session_id: str, _: bool = Depends(check_admin)):
    return {"messages": db.get_messages(session_id)}


@app.post("/admin/api/reply")
def admin_reply(req: ReplyRequest, _: bool = Depends(check_admin)):
    db.add_message(req.session_id, "human", req.content)
    return {"ok": True}


@app.post("/admin/api/handback")
def admin_handback(req: ReplyRequest, _: bool = Depends(check_admin)):
    db.set_status(req.session_id, "bot")
    # If the customer's last message is unanswered, let the bot answer it now
    msgs = db.get_messages(req.session_id)
    if msgs and msgs[-1]["role"] == "user":
        threading.Thread(target=run_bot_turn, args=(req.session_id,), daemon=True).start()
    return {"ok": True}


@app.post("/admin/api/takeover")
def admin_takeover(req: ReplyRequest, _: bool = Depends(check_admin)):
    db.set_status(req.session_id, "escalated", "נציג השתלט על השיחה")
    return {"ok": True}


@app.get("/admin", response_class=HTMLResponse)
def admin_page(_: bool = Depends(check_admin)):
    return ADMIN_HTML


ADMIN_HTML = """
<!doctype html>
<html lang="he" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grinta — תיבת פניות</title>
<style>
  :root { --black:#0a0a0a; --gold:#c9a227; --gray:#f4f4f4; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:'Heebo',Arial,sans-serif; background:var(--gray); color:#1a1a1a; }
  header { background:var(--black); color:#fff; padding:14px 20px; border-bottom:2px solid var(--gold); }
  header b { color:var(--gold); }
  .wrap { display:flex; height:calc(100vh - 52px); }
  .list { width:320px; background:#fff; border-left:1px solid #e2e2e2; overflow-y:auto; }
  .filter { padding:10px; border-bottom:1px solid #eee; display:flex; gap:8px; }
  .filter button { flex:1; padding:8px; border:1px solid #ddd; background:#fff; border-radius:8px; cursor:pointer; font-family:inherit; }
  .filter button.active { background:var(--black); color:#fff; border-color:var(--black); }
  .sess { padding:12px 14px; border-bottom:1px solid #f0f0f0; cursor:pointer; }
  .sess:hover { background:#fafafa; }
  .sess.active { background:#fff7e0; border-right:3px solid var(--gold); }
  .sess .top { display:flex; justify-content:space-between; align-items:center; }
  .badge { font-size:11px; padding:2px 8px; border-radius:10px; }
  .badge.escalated { background:#ffe0e0; color:#b00; }
  .badge.bot { background:#e0f0ff; color:#06c; }
  .badge.closed { background:#eee; color:#888; }
  .sess .preview { font-size:13px; color:#666; margin-top:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .conv { flex:1; display:flex; flex-direction:column; }
  .msgs { flex:1; overflow-y:auto; padding:18px; display:flex; flex-direction:column; gap:10px; }
  .m { max-width:70%; padding:10px 14px; border-radius:14px; font-size:14px; line-height:1.5; white-space:pre-wrap; }
  .m.user { align-self:flex-start; background:#fff; border:1px solid #e2e2e2; }
  .m.assistant { align-self:flex-end; background:var(--black); color:#fff; }
  .m.human { align-self:flex-end; background:var(--gold); color:#0a0a0a; }
  .m .who { font-size:10px; opacity:.6; margin-bottom:3px; }
  .composer { display:flex; gap:8px; padding:12px; background:#fff; border-top:1px solid #e2e2e2; }
  .composer input { flex:1; padding:11px 14px; border:1px solid #d8d8d8; border-radius:22px; font-family:inherit; font-size:14px; }
  .composer button { padding:0 18px; border:none; border-radius:22px; background:var(--gold); cursor:pointer; font-weight:700; }
  .composer .hb { background:#eee; }
  .empty { margin:auto; color:#999; }
</style>
</head>
<body>
<header>Grinta — <b>תיבת פניות</b></header>
<div class="wrap">
  <div class="list">
    <div class="filter">
      <button id="f-esc" class="active" onclick="setFilter(true)">דורש טיפול</button>
      <button id="f-all" onclick="setFilter(false)">הכל</button>
    </div>
    <div id="sessions"></div>
  </div>
  <div class="conv">
    <div class="msgs" id="msgs"><div class="empty">בחר שיחה מהרשימה</div></div>
    <div class="composer">
      <input id="reply" placeholder="כתוב תשובה ללקוח..." onkeydown="if(event.key==='Enter')sendReply()">
      <button onclick="sendReply()">שלח</button>
      <button class="hb" id="toggleBtn" style="display:none">✋ קח שליטה</button>
    </div>
  </div>
</div>
<script>
  let onlyEsc = true;
  let current = null;
  let sessionsData = [];

  function setFilter(esc){
    onlyEsc = esc;
    document.getElementById('f-esc').classList.toggle('active', esc);
    document.getElementById('f-all').classList.toggle('active', !esc);
    loadSessions();
  }

  function loadSessions(){
    fetch('/admin/api/sessions?only_escalated=' + onlyEsc)
      .then(r=>r.json()).then(d=>{
        sessionsData = d.sessions || [];
        const box = document.getElementById('sessions');
        box.innerHTML = '';
        sessionsData.forEach(s=>{
          const div = document.createElement('div');
          div.className = 'sess' + (s.session_id===current ? ' active':'');
          div.onclick = ()=>openConv(s.session_id);
          div.innerHTML =
            '<div class="top"><span class="badge '+s.status+'">'+s.status+'</span>'+
            '<span style="font-size:11px;color:#aaa">'+(s.message_count||0)+' הודעות</span></div>'+
            '<div class="preview">'+(s.last_message||'')+'</div>';
          box.appendChild(div);
        });
        updateToggle();
      });
  }

  function updateToggle(){
    const btn = document.getElementById('toggleBtn');
    if(!btn) return;
    const s = sessionsData.find(x=>x.session_id===current);
    if(!s){ btn.style.display='none'; return; }
    btn.style.display='inline-block';
    if(s.status==='escalated'){
      btn.textContent = '↩︎ החזר לבוט';
      btn.onclick = handback;
    } else {
      btn.textContent = '✋ קח שליטה';
      btn.onclick = takeover;
    }
  }

  function openConv(id){
    current = id;
    loadSessions();
    loadMsgs();
    updateToggle();
  }

  function loadMsgs(){
    if(!current) return;
    fetch('/admin/api/messages?session_id=' + encodeURIComponent(current))
      .then(r=>r.json()).then(d=>{
        const box = document.getElementById('msgs');
        box.innerHTML = '';
        (d.messages||[]).forEach(m=>{
          const who = m.role==='user'?'לקוח':(m.role==='human'?'אתה':'בוט');
          const div = document.createElement('div');
          div.className = 'm ' + m.role;
          let html = '<div class="who">'+who+'</div>';
          if (m.content) html += escapeHtml(m.content);
          if (m.image_data) html += '<br><img src="'+m.image_data+'" style="max-width:200px;border-radius:8px;margin-top:6px">';
          div.innerHTML = html;
          box.appendChild(div);
        });
        box.scrollTop = box.scrollHeight;
      });
  }

  function sendReply(){
    const inp = document.getElementById('reply');
    const text = inp.value.trim();
    if(!text || !current) return;
    inp.value='';
    fetch('/admin/api/reply', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({session_id: current, content: text})})
      .then(()=>loadMsgs());
  }

  function handback(){
    if(!current) return;
    fetch('/admin/api/handback', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({session_id: current, content: ''})})
      .then(()=>{ loadSessions(); });
  }

  function takeover(){
    if(!current) return;
    fetch('/admin/api/takeover', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({session_id: current, content: ''})})
      .then(()=>{ loadSessions(); });
  }

  function escapeHtml(s){
    return s.replace(/[&<>]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
  }

  loadSessions();
  setInterval(loadSessions, 5000);
  setInterval(loadMsgs, 4000);
</script>
</body>
</html>
"""