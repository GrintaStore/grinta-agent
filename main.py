import os
import json
import base64
import secrets
import threading
import time
import tools
import requests
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta

from google import genai
from google.genai import types
from fastapi import FastAPI, Depends, HTTPException, status, Request
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

# Models are tried in order: if one fails (e.g. 503 overloaded), fall to the next
MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
]

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
ADMIN_PASSWORD    = os.environ.get("ADMIN_PASSWORD", "grinta123")
NOTIFY_EMAIL      = os.environ.get("NOTIFY_EMAIL", "")
ADMIN_URL         = os.environ.get("ADMIN_URL", "https://grinta-agent.onrender.com/admin")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
EMAIL_FROM     = os.environ.get("EMAIL_FROM", "Grinta <contact@grinta.co.il>")
RESEND_WEBHOOK_TOKEN = os.environ.get("RESEND_WEBHOOK_TOKEN", "")

# ─────────────────────────────────────────────
# Email notification
# ─────────────────────────────────────────────
def _send_email(subject: str, body: str) -> None:
    if not (RESEND_API_KEY and NOTIFY_EMAIL):
        print("[email] Resend not configured — skipping")
        return
    try:
        res = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "from": EMAIL_FROM,
                "to": [NOTIFY_EMAIL],
                "subject": subject,
                "text": body,
            },
            timeout=20,
        )
        if res.status_code in (200, 201):
            print("[email] sent")
        else:
            print(f"[email] failed {res.status_code}: {res.text[:200]}")
    except Exception as e:
        print(f"[email] failed: {e}")


# Branded HTML wrapper for customer replies. The reply text is inserted at
# __GR_BODY__ (already HTML-escaped, newlines -> <br>). No order/tracking parts.
EMAIL_TEMPLATE = """<!DOCTYPE html>
<html lang="he" dir="rtl" xmlns="http://www.w3.org/1999/xhtml">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>הודעה מ-Grinta</title>
  <!--[if mso]>
  <style type="text/css">
    table, td, div, p, a, span { font-family: Tahoma, Arial, sans-serif !important; }
  </style>
  <![endif]-->
  <style>
    body, table, td, a { -webkit-text-size-adjust:100%; -ms-text-size-adjust:100%; }
    table, td { mso-table-lspace:0pt; mso-table-rspace:0pt; }
    img { border:0; outline:none; text-decoration:none; }
    body { margin:0; padding:0; width:100% !important; height:100% !important; background-color:#f3eee3; direction:rtl; }
    .gr-btn:hover { background-color:#b58b3e !important; }
    @media only screen and (max-width:600px) {
      .gr-container { width:100% !important; }
      .gr-px { padding-left:22px !important; padding-right:22px !important; }
      .gr-h1 { font-size:24px !important; line-height:30px !important; }
    }
  </style>
</head>
<body>
  <div style="display:none; max-height:0; overflow:hidden; opacity:0;">הודעה חדשה מ-Grinta</div>
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f3eee3;">
    <tr>
      <td align="center" style="padding:24px 12px;">
        <table class="gr-container" width="600" cellpadding="0" cellspacing="0" style="max-width:600px; background:#ffffff; border-radius:18px; overflow:hidden; box-shadow:0 8px 30px rgba(13,13,13,0.10);">

          <!-- Header -->
          <tr>
            <td align="center" style="padding:32px 32px 16px;">
              <a href="https://grinta.co.il">
                <img src="https://cdn.shopify.com/s/files/1/0809/9633/5859/files/logo_transparent_cut_b4935059-4a03-4dd7-9a3a-122897ef8959.png?v=1780164862" height="80" style="display:block;">
              </a>
            </td>
          </tr>

          <!-- Black bar -->
          <tr>
            <td align="center" style="background:#0d0d0d; padding:14px;">
              <span style="color:#c6a15b; font-family:Tahoma,sans-serif; font-size:15px; font-weight:700;">שירות לקוחות · Grinta</span>
            </td>
          </tr>

          <!-- Gold stripe -->
          <tr>
            <td style="height:4px; background:linear-gradient(90deg,#a87f3c,#e3c987,#a87f3c);"></td>
          </tr>

          <!-- Title -->
          <tr>
            <td class="gr-px" style="padding:36px 40px 12px; text-align:right;">
              <h1 class="gr-h1" style="margin:0; font-family:Tahoma,sans-serif; font-size:28px; line-height:36px; color:#0d0d0d;">💬 יש לנו הודעה עבורך</h1>
            </td>
          </tr>

          <!-- Message box -->
          <tr>
            <td class="gr-px" style="padding:12px 40px 24px;">
              <table width="100%" cellpadding="0" cellspacing="0" style="background:#fdfbf6; border:1px solid #ece4d3; border-radius:12px;">
                <tr>
                  <td style="padding:18px 20px; text-align:right; font-family:Tahoma,sans-serif; font-size:15px; line-height:26px; color:#0d0d0d;">__GR_BODY__</td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- Reply button -->
          <tr>
            <td align="center" style="padding:0 40px 30px;">
              <a href="mailto:contact@grinta.co.il" class="gr-btn"
                 style="display:inline-block; background:#c6a15b; color:#0d0d0d; font-family:Tahoma,sans-serif; font-size:16px; font-weight:700; padding:14px 44px; border-radius:50px; text-decoration:none;">השב למייל</a>
            </td>
          </tr>

          <!-- Instagram -->
          <tr>
            <td style="padding:0 40px 24px;">
              <table width="100%" cellpadding="0" cellspacing="0" style="background:#fdfbf6; border:1px solid #ece4d3; border-radius:14px;">
                <tr>
                  <td align="center" style="padding:24px;">
                    <div style="font-size:28px;">📸</div>
                    <div style="font-family:Tahoma,sans-serif; font-size:18px; font-weight:700; margin-top:10px;">עקבו אחרינו באינסטגרם</div>
                    <div style="font-family:Tahoma,sans-serif; font-size:14px; margin-top:8px; color:#4a4339;">חדשות, מבצעים והחולצות הכי חמות</div>
                    <a href="https://www.instagram.com/grinta.co.il/"
                       style="display:inline-block; margin-top:16px; background:#0d0d0d; color:#c6a15b; padding:12px 32px; border-radius:50px; text-decoration:none; font-weight:700;">@grinta.co.il ←</a>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="background:#0d0d0d; border-top:3px solid #c6a15b; padding:28px 40px 30px;" align="center">
              <div style="font-family:Tahoma,sans-serif; font-size:18px; font-weight:700; letter-spacing:3px; color:#ffffff;">GRINTA</div>
              <div style="margin-top:14px; text-align:center; font-family:Tahoma,sans-serif;">
                <a href="https://www.grinta.co.il" style="color:#c6a15b; text-decoration:none; font-size:13px; padding:0 8px;">החנות</a>
                <span style="color:#4a4339;">|</span>
                <a href="https://www.instagram.com/grinta.co.il/" style="color:#c6a15b; text-decoration:none; font-size:13px; padding:0 8px;">אינסטגרם</a>
                <span style="color:#4a4339;">|</span>
                <a href="https://grinta.co.il/pages/contact-us" style="color:#c6a15b; text-decoration:none; font-size:13px; padding:0 8px;">צור קשר</a>
              </div>
              <p style="margin:16px 0 0; font-family:Tahoma,sans-serif; font-size:11px; line-height:18px; color:#8a8170;">© 2026 Grinta · כל הזכויות שמורות</p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""


def render_email_html(body_text: str) -> str:
    """Wrap a plain-text reply in the branded Grinta HTML shell."""
    safe = (body_text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe = safe.replace("\n", "<br>")
    return EMAIL_TEMPLATE.replace("__GR_BODY__", safe)


def send_customer_email(to_email: str, subject: str, body: str) -> bool:
    """Send a reply to a customer from contact@grinta.co.il via Resend."""
    if not RESEND_API_KEY:
        print("[customer email] Resend not configured")
        return False
    reply_subject = subject if subject.lower().startswith("re:") else f"Re: {subject}"
    try:
        res = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "from": EMAIL_FROM,
                "to": [to_email],
                "subject": reply_subject,
                "html": render_email_html(body),
                "text": body,
            },
            timeout=20,
        )
        if res.status_code in (200, 201):
            print(f"[customer email] sent to {to_email}")
            return True
        print(f"[customer email] failed {res.status_code}: {res.text[:200]}")
    except Exception as e:
        print(f"[customer email] error: {e}")
    return False


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
- Do NOT volunteer delivery time or shipping details unless the customer specifically asks about them.

## Your knowledge
{KNOWLEDGE}

## Tool usage rules
- If a customer asks about their order, ALWAYS call get_order_by_email or get_order_by_number before responding
- Never invent order information — only report what the tools return
- If a customer provides an order number, use get_order_by_number
- If a customer provides an email, use get_order_by_email
- If neither is provided, ask the customer for their email or order number first
- For product/team/jersey/availability questions: the full product catalog is provided below under "Product catalog". Answer ONLY from that list. If a product appears in the catalog, it exists and is available. If a team/product is NOT in the catalog, we don't currently carry it — offer to check if we can source it (ask for club, season, and kit type). NEVER invent products or claim something is out of stock if it appears in the catalog. For availability of a specific SIZE, refer the customer to the size guide / product page, since the catalog lists the size range we offer, not live per-size stock. Each catalog entry includes a link (קישור) — when a customer asks for a product link or where to buy it, give them that exact link. Never invent or guess a link.
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
    current_page: str | None = None


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


def build_system_instruction(current_page: str | None = None) -> str:
    """System prompt + the live product catalog (cached, refreshed periodically)."""
    catalog = tools.get_catalog_text()
    instruction = SYSTEM_PROMPT
    if catalog:
        instruction += (
            "\n\n## Product catalog (Hebrew — this is the full list of products we offer)\n"
            "כל מוצר שמופיע כאן קיים וזמין להזמנה, עם המידות והאופציות שלו. "
            "מוצר שלא מופיע ברשימה — איננו מציעים אותו כרגע.\n\n"
            + catalog
        )
    if current_page:
        instruction += (
            "\n\n## Current page\n"
            f"The customer is currently viewing this page on the site: {current_page}\n"
            "Use this to understand context. If it is a product page (/products/{handle}), "
            "match the handle to the catalog and treat that product as what the customer is "
            'referring to when they say "this jersey", "it", "this", etc. '
            "Only use this when relevant — for general questions, ignore it."
        )
    return instruction


def gemini_generate(history, current_page: str | None = None):
    """Try each model in order; fall to the next if one fails (503) or returns empty content."""
    system_instruction = build_system_instruction(current_page)
    last_error = None
    for model_name in MODELS:
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=TOOLS,
                )
            )
            # A response can come back "successful" but empty (safety block,
            # max tokens, etc.) with content.parts = None. Treat that as a
            # failure so we fall through to the next model instead of crashing.
            cand = (resp.candidates or [None])[0]
            if cand is None or cand.content is None or cand.content.parts is None:
                fr = getattr(cand, "finish_reason", None) if cand else None
                print(f"[model] {model_name} returned no content (finish_reason={fr}) — trying next")
                last_error = RuntimeError(f"empty response from {model_name} (finish_reason={fr})")
                continue
            return resp
        except Exception as e:
            last_error = e
            print(f"[model] {model_name} failed: {str(e)[:120]} — trying next")
            continue
    # All models failed
    raise last_error


def run_loop(session_id: str, history, current_page: str | None = None):
    """Run the tool-use loop over a prepared history. Returns (text, escalated)."""
    escalated = False
    for _ in range(5):
        response = gemini_generate(history, current_page)
        content = response.candidates[0].content
        history.append(content)

        parts = content.parts or []
        tool_calls = [p for p in parts if p.function_call is not None]

        if not tool_calls:
            text = "".join(p.text for p in parts if p.text).strip()
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

    # Record the customer's current page (for the admin inbox)
    if req.current_page:
        db.set_last_page(req.session_id, req.current_page)

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

    text, escalated = run_loop(req.session_id, history, req.current_page)

    # Re-check: if a human took over DURING generation, discard the bot's answer
    fresh = db.get_session(req.session_id)
    if fresh and fresh.get("status") == "escalated" and not escalated:
        return ChatResponse(reply="", session_id=req.session_id, escalated=True)

    row = db.add_message(req.session_id, "assistant", text)
    if escalated:
        db.set_status(req.session_id, "escalated")
    reply_id = (row or {}).get("id") or 0
    return ChatResponse(reply=text, session_id=req.session_id, escalated=escalated, reply_id=reply_id)


def fetch_inbound_email(email_id: str) -> dict:
    """Resend's email.received webhook is metadata-only; fetch the full body here."""
    try:
        res = requests.get(
            f"https://api.resend.com/emails/receiving/{email_id}",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
            timeout=20,
        )
        if res.status_code == 200:
            return res.json()
        print(f"[inbound fetch] failed {res.status_code}: {res.text[:200]}")
    except Exception as e:
        print(f"[inbound fetch] error: {e}")
    return {}


@app.post("/email/inbound")
async def email_inbound(req: Request, token: str = ""):
    # simple shared-secret check (token is in the webhook URL)
    if RESEND_WEBHOOK_TOKEN and token != RESEND_WEBHOOK_TOKEN:
        raise HTTPException(status_code=401, detail="bad token")

    payload = await req.json()
    if payload.get("type") != "email.received":
        return {"ok": True}  # ignore non-inbound events

    data = payload.get("data", {})
    from_raw = data.get("from", "") or ""
    m = re.search(r"<([^>]+)>", from_raw)
    from_email = (m.group(1) if m else from_raw).strip().lower()
    if not from_email:
        return {"ok": True}

    email_id = data.get("email_id") or data.get("id") or ""
    full     = fetch_inbound_email(email_id) if email_id else {}

    subject = (full.get("subject") or data.get("subject") or "").strip() or "(ללא נושא)"
    body    = (full.get("text") or full.get("html") or "").strip()
    stored  = f"נושא: {subject}\n\n{body}"

    session_id = f"email-{from_email}"
    db.ensure_session(session_id)
    db.set_email_meta(session_id, from_email, subject)
    db.add_message(session_id, "user", stored)
    db.set_status(session_id, "escalated", "פנייה במייל")
    return {"ok": True}


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
                "created_at": m.get("created_at"),
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

    # If this is an email conversation, actually email the reply to the customer.
    sess = db.get_session(req.session_id)
    if sess and sess.get("channel") == "email" and sess.get("customer_email"):
        ok = send_customer_email(
            sess["customer_email"],
            sess.get("email_subject") or "פנייתך ל-Grinta",
            req.content,
        )
        return {"ok": True, "emailed": ok}

    return {"ok": True}


@app.post("/admin/api/generate")
def admin_generate(req: ReplyRequest, _: bool = Depends(check_admin)):
    # Build history from the conversation, then append an explicit instruction
    # so the model always has a concrete task — otherwise, if the conversation
    # already ends with a reply, the model returns an empty STOP completion.
    history = build_history(req.session_id, skip_last_user=False)
    if not history:
        return {"draft": ""}
    history.append(types.Content(
        role="user",
        parts=[types.Part(text=(
            "כתוב טיוטת תשובה ללקוח על סמך השיחה עד כה. "
            "החזר רק את נוסח התשובה ללקוח, ללא הקדמות או הסברים."
        ))]
    ))
    try:
        text, _ = run_loop(req.session_id, history)
    except Exception as e:
        print(f"[generate] error: {e}")
        return {"draft": "", "error": str(e)}
    return {"draft": text or ""}


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
  :root { --black:#0a0a0a; --gold:#E0B252; --gray:#f4f4f4; }
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
  .composer { display:flex; flex-wrap:wrap; gap:8px; padding:12px; background:#fff; border-top:1px solid #e2e2e2; }
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
    <div id="pagebar" style="display:none;padding:8px 14px;background:#fff;border-bottom:1px solid #eee;font-size:13px;color:#555;"></div>
    <div class="msgs" id="msgs"><div class="empty">בחר שיחה מהרשימה</div></div>
    <div class="composer">
      <input id="reply" placeholder="כתוב תשובה ללקוח..." onkeydown="if(event.key==='Enter')sendReply()">
        <button id="genBtn" onclick="generateDraft()" style="background:#0a0a0a;color:var(--gold)">✨ צור טיוטה</button>
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
        var chan = s.channel==='email'
          ? '<span class="badge" style="background:#e7f6e7;color:#1a7a3a;margin-left:4px">📧 מייל</span>'
          : '';
        div.innerHTML =
          '<div class="top"><span>'+chan+'<span class="badge '+s.status+'">'+s.status+'</span></span>'+
          '<span style="font-size:11px;color:#aaa">'+fmtTime(s.updated_at)+'</span></div>'+
          '<div class="preview">'+(s.customer_email ? s.customer_email+' — ' : '')+(s.last_message||'')+'</div>';
          box.appendChild(div);
        });
        updateToggle();
        updatePageBar();
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

  function updatePageBar(){
    const bar = document.getElementById('pagebar');
    if(!bar) return;
    const s = sessionsData.find(x=>x.session_id===current);
    if(!s || !s.last_page){ bar.style.display='none'; return; }
    bar.style.display='block';
    bar.innerHTML = '📍 דף אחרון: <a href="'+s.last_page+'" target="_blank" style="color:#06c;word-break:break-all">'+s.last_page+'</a>';
  }

  function openConv(id){
    current = id;
    loadSessions();
    loadMsgs();
    updateToggle();
    updatePageBar();
  }

  function loadMsgs(){
    if(!current) return;
    fetch('/admin/api/messages?session_id=' + encodeURIComponent(current))
      .then(r=>r.json()).then(d=>{
        const box = document.getElementById('msgs');
        const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 80;
        box.innerHTML = '';
        let lastDay = null;
        (d.messages||[]).forEach(m=>{
          if (m.created_at) {
            const k = dateKey(m.created_at);
            if (k !== lastDay) {
              lastDay = k;
              const sep = document.createElement('div');
              sep.style.cssText = 'align-self:center;background:#e8e8e8;color:#666;font-size:11px;padding:3px 10px;border-radius:10px;margin:6px 0;';
              sep.textContent = dateLabel(m.created_at);
              box.appendChild(sep);
            }
          }
          const who = m.role==='user'?'לקוח':(m.role==='human'?'אתה':'בוט');
          const div = document.createElement('div');
          div.className = 'm ' + m.role;
          let html = '<div class="who">'+who+'</div>';
          if (m.content) html += escapeHtml(m.content);
          if (m.image_data) html += '<br><img src="'+m.image_data+'" style="max-width:200px;border-radius:8px;margin-top:6px">';
          if (m.created_at) html += '<div style="font-size:10px;opacity:.55;margin-top:4px;text-align:left">'+fmtTime(m.created_at)+'</div>';
          div.innerHTML = html;
          box.appendChild(div);
        });
        if (nearBottom) box.scrollTop = box.scrollHeight;
      });
  }

  function sendReply(){
    const inp = document.getElementById('reply');
    const text = inp.value.trim();
    if(!text || !current) return;
    inp.value='';
    fetch('/admin/api/reply', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({session_id: current, content: text})})
      .then(r=>r.json())
      .then(d=>{
        if(d.emailed === false){ alert('נשמר, אך שליחת המייל ללקוח נכשלה — בדוק את הלוגים'); }
        loadMsgs();
      })
      .catch(()=>{ alert('שגיאה בשליחה'); loadMsgs(); });
  }

    function generateDraft(){
      if(!current) return;
      var btn = document.getElementById('genBtn');
      var old = btn.textContent;
      btn.textContent = '...חושב';
      btn.disabled = true;
      fetch('/admin/api/generate', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({session_id: current, content: ''})})
        .then(r=>r.json())
        .then(d=>{
          if(d.draft){ document.getElementById('reply').value = d.draft; }
          else { alert('לא הצלחתי לייצר טיוטה' + (d.error ? ': '+d.error : '')); }
        })
        .catch(()=>alert('שגיאה בחיבור'))
        .finally(()=>{ btn.textContent = old; btn.disabled = false; });
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

  function fmtTime(ts){
    if(!ts) return '';
    try { return new Date(ts).toLocaleTimeString('he-IL',{hour:'2-digit',minute:'2-digit',timeZone:'Asia/Jerusalem'}); }
    catch(e){ return ''; }
  }
  function dateKey(ts){
    return new Date(ts).toLocaleDateString('en-CA',{timeZone:'Asia/Jerusalem'});
  }
  function dateLabel(ts){
    const k = dateKey(ts), today = dateKey(Date.now()), yest = dateKey(Date.now()-86400000);
    if(k===today) return 'היום';
    if(k===yest) return 'אתמול';
    return new Date(ts).toLocaleDateString('he-IL',{day:'numeric',month:'numeric',year:'numeric',timeZone:'Asia/Jerusalem'});
  }

  loadSessions();
  setInterval(loadSessions, 5000);
  setInterval(loadMsgs, 4000);
</script>
</body>
</html>
"""
