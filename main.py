import os
import json
import base64
import secrets
import threading
import time
import tools
import requests
import re
import hmac
import hashlib
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
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
]

# Stronger models used for email draft generation (better answers + tool use).
# The fast/cheap "lite" models stay for live chat; email drafts get these.
GENERATE_MODELS = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
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

# Zernio (Instagram + WhatsApp inbox). API key + webhook signing secret.
ZERNIO_API_KEY       = os.environ.get("ZERNIO_API_KEY", "")
ZERNIO_WEBHOOK_SECRET = os.environ.get("ZERNIO_WEBHOOK_SECRET", "")
ZERNIO_API_BASE      = "https://zernio.com/api/v1"

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
    body { margin:0; padding:0; width:100% !important; height:100% !important; background-color:#e8e3d8; direction:rtl; }
    @media only screen and (max-width:600px) {
      .gr-container { width:100% !important; }
      .gr-px { padding-left:24px !important; padding-right:24px !important; }
    }
  </style>
</head>
<body>
  <div style="display:none; max-height:0; overflow:hidden; opacity:0;">הודעה חדשה מ-Grinta</div>
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#e8e3d8;">
    <tr>
      <td align="center" style="padding:28px 12px;">
        <table class="gr-container" width="520" cellpadding="0" cellspacing="0" style="max-width:520px; background:#ffffff; border-radius:14px; overflow:hidden;">

          <!-- Logo -->
          <tr>
            <td align="center" style="padding:30px 32px 18px;">
              <a href="https://www.grinta.co.il">
                <img src="https://cdn.shopify.com/s/files/1/0809/9633/5859/files/logo_transparent_cut_b4935059-4a03-4dd7-9a3a-122897ef8959.png?v=1780164862" height="64" style="display:block;">
              </a>
            </td>
          </tr>

          <!-- Thin gold line -->
          <tr>
            <td style="padding:0 32px;">
              <div style="height:3px; background:#c6a15b;"></div>
            </td>
          </tr>

__GR_QUOTE_BLOCK__
          <!-- Message -->
          <tr>
            <td class="gr-px" dir="rtl" style="padding:26px 34px 24px; direction:rtl; unicode-bidi:plaintext; text-align:right; font-family:Tahoma,sans-serif; font-size:15px; line-height:27px; color:#1a1a1a;">__GR_BODY__</td>
          </tr>

          <!-- Instagram -->
          <tr>
            <td style="padding:0 32px 26px;">
              <table width="100%" cellpadding="0" cellspacing="0" style="background:#fdfbf6; border:1px solid #ece4d3; border-radius:14px;">
                <tr>
                  <td align="center" style="padding:22px;">
                    <div style="font-size:26px;">📸</div>
                    <div style="font-family:Tahoma,sans-serif; font-size:17px; font-weight:700; margin-top:8px; color:#1a1a1a;">עקבו אחרינו באינסטגרם</div>
                    <div style="font-family:Tahoma,sans-serif; font-size:13px; margin-top:6px; color:#4a4339;">חדשות, מבצעים והחולצות הכי חמות</div>
                    <a href="https://www.instagram.com/grinta.co.il/"
                       style="display:inline-block; margin-top:14px; background:#0d0d0d; color:#c6a15b; padding:11px 30px; border-radius:50px; text-decoration:none; font-weight:700; font-size:14px;">@grinta.co.il ←</a>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="background:#0d0d0d; padding:20px 32px;" align="center">
              <div style="font-family:Tahoma,sans-serif; font-size:16px; font-weight:700; letter-spacing:3px; color:#c6a15b;">GRINTA</div>
              <div style="margin-top:10px; font-family:Tahoma,sans-serif; font-size:12px;">
                <a href="https://www.grinta.co.il" style="color:#c6a15b; text-decoration:none; padding:0 7px;">החנות</a>
                <span style="color:#555555;">|</span>
                <a href="https://www.instagram.com/grinta.co.il/" style="color:#c6a15b; text-decoration:none; padding:0 7px;">אינסטגרם</a>
                <span style="color:#555555;">|</span>
                <a href="https://grinta.co.il/pages/contact-us" style="color:#c6a15b; text-decoration:none; padding:0 7px;">צור קשר</a>
              </div>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""


# Sub-template for the quoted customer message (inserted at __GR_QUOTE_BLOCK__).
QUOTE_BLOCK = """          <!-- Quoted customer message -->
          <tr>
            <td class="gr-px" dir="rtl" style="padding:22px 34px 0; direction:rtl; unicode-bidi:plaintext; text-align:right;">
              <div style="font-family:Tahoma,sans-serif; font-size:11px; font-weight:700; color:#a87f3c; margin-bottom:6px;">בתגובה להודעתך</div>
              <div style="background:#f7f3ea; border-right:3px solid #c6a15b; border-radius:8px; padding:12px 14px; font-family:Tahoma,sans-serif; font-size:13.5px; line-height:24px; color:#6b6357;">__GR_QUOTE_TEXT__</div>
            </td>
          </tr>
"""


def _escape_html(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")


def _clean_quote(text: str, max_len: int = 600) -> str:
    """Reduce a stored inbound message to just the customer's latest text:
    drop the leading 'נושא:' line we prepend, and cut old quoted reply history."""
    if not text:
        return ""
    t = text.strip()
    # strip the leading "נושא: ..." line added on inbound
    t = re.sub(r"^\s*נושא:.*?(?:\n|$)", "", t, count=1).strip()
    # cut at the first quoted-history marker (Gmail/Outlook, Hebrew + English)
    markers = [
        r"\nOn .{0,300}?wrote:",
        r"\nבתאריך .{0,300}?(?:כתב|כתבה|כתב/ה).{0,80}?:",
        r"\n-{2,}\s*Original Message\s*-{2,}",
        r"\n_{5,}",
        r"\nFrom:\s.+\nSent:\s",
        r"\n>",
    ]
    cut = len(t)
    for mk in markers:
        mm = re.search(mk, t, flags=re.IGNORECASE | re.DOTALL)
        if mm and mm.start() < cut:
            cut = mm.start()
    t = t[:cut].strip()
    # drop any leftover lines that are quoted (start with >)
    t = "\n".join(ln for ln in t.split("\n") if not ln.lstrip().startswith(">")).strip()
    t = re.sub(r"\n{3,}", "\n\n", t)
    if len(t) > max_len:
        t = t[:max_len].rstrip() + "…"
    return t


def render_email_html(body_text: str, quote_text: str | None = None) -> str:
    """Wrap a plain-text reply in the branded Grinta HTML shell.
    If quote_text is given, the customer's message is shown above the reply."""
    body_html = _escape_html(body_text)
    if quote_text:
        quote_block = QUOTE_BLOCK.replace("__GR_QUOTE_TEXT__", _escape_html(quote_text))
    else:
        quote_block = ""
    return (EMAIL_TEMPLATE
            .replace("__GR_QUOTE_BLOCK__", quote_block)
            .replace("__GR_BODY__", body_html))


# Filename to give an attached image, by mime type.
_IMAGE_EXT = {"image/jpeg": "jpg", "image/jpg": "jpg", "image/png": "png",
              "image/webp": "webp", "image/gif": "gif"}


def send_customer_email(to_email: str, subject: str, body: str, quote: str | None = None,
                        images: list | None = None) -> bool:
    """Send a reply to a customer from contact@grinta.co.il via Resend.
    Images, if given, are sent as real file attachments on a single email (never
    inline), so they are always delivered and never blocked by the mail client.
    Each image is an object with .base64 and .mime."""
    if not RESEND_API_KEY:
        print("[customer email] Resend not configured")
        return False
    reply_subject = subject if subject.lower().startswith("re:") else f"Re: {subject}"
    payload = {
        "from": EMAIL_FROM,
        "to": [to_email],
        "subject": reply_subject,
        "html": render_email_html(body, quote),
        "text": body,
    }
    if images:
        atts = []
        for i, img in enumerate(images, 1):
            name = (img.name or "").strip()
            if not name:
                ext = _IMAGE_EXT.get((img.mime or "").lower(), "jpg")
                suffix = "" if len(images) == 1 else f"-{i}"
                name = f"grinta{suffix}.{ext}"
            atts.append({"filename": name, "content": img.base64})
        payload["attachments"] = atts
    try:
        res = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
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
# Zernio (Instagram + WhatsApp)
# ─────────────────────────────────────────────
# Map a Zernio platform value to our internal channel name. Only these two are
# wired for now; other platforms are ignored on inbound.
ZERNIO_PLATFORMS = {"instagram": "instagram", "whatsapp": "whatsapp"}

# In-memory de-dup of webhook event ids. Deliveries are at-least-once, so the
# same message can arrive twice; we skip ids we've already handled. Best-effort:
# resets on restart / isn't shared across workers, which is fine at this volume.
_zernio_seen_events: set[str] = set()


def _zernio_already_seen(event_id: str) -> bool:
    if not event_id:
        return False
    if event_id in _zernio_seen_events:
        return True
    _zernio_seen_events.add(event_id)
    if len(_zernio_seen_events) > 2000:
        _zernio_seen_events.clear()
    return False


def _zernio_verify(raw_body: bytes, signature: str) -> bool:
    """Verify the X-Zernio-Signature header: lowercase hex HMAC-SHA256 of the raw
    body keyed by the webhook secret. If no secret is configured, allow (dev)."""
    if not ZERNIO_WEBHOOK_SECRET:
        return True
    computed = hmac.new(
        ZERNIO_WEBHOOK_SECRET.encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(computed, (signature or "").strip())


def send_zernio_message(conversation_id: str, account_id: str, text: str,
                        image_url: str | None = None, mime: str | None = None) -> bool:
    """Send a reply into an existing Zernio conversation (IG/WhatsApp).
    Only works inside the platform's 24h customer-service window.
    image_url attaches a file (must be a public URL — we pass the Supabase one);
    mime decides the attachmentType (image / video / audio / document)."""
    if not ZERNIO_API_KEY:
        print("[zernio] API key not configured")
        return False
    if not (conversation_id and account_id):
        print("[zernio] missing conversation_id/account_id — cannot send")
        return False
    if not (text or image_url):
        return False
    url = f"{ZERNIO_API_BASE}/inbox/conversations/{conversation_id}/messages"
    payload = {"accountId": account_id, "message": text or ""}
    if image_url:
        payload["attachmentUrl"] = image_url
        payload["attachmentType"] = _zernio_attachment_type(mime)
    try:
        res = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {ZERNIO_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        # Log the full response so we can see what Zernio actually did — a 2xx can
        # still be a no-op or carry an error body.
        print(f"[zernio] POST {url} account={account_id} att={payload.get('attachmentType')} -> {res.status_code}: {res.text[:500]}")
        if res.status_code in (200, 201):
            return True
    except Exception as e:
        print(f"[zernio] send error: {e}")
    return False


def _recent_outgoing_echo(session_id: str, text: str, within_seconds: int = 180) -> bool:
    """True if an identical outgoing reply (human OR bot) was stored in this
    session very recently. Used to drop the message.sent webhook echo of a reply
    we already stored ourselves (panel reply or bot reply), so it isn't shown
    twice."""
    text = (text or "").strip()
    if not text:
        return False
    try:
        msgs = db.get_messages(session_id)
    except Exception:
        return False
    now = datetime.now(timezone.utc)
    for m in reversed(msgs[-8:]):  # only the recent tail
        if m.get("role") not in ("human", "assistant"):
            continue
        if (m.get("content") or "").strip() != text:
            continue
        ca = m.get("created_at")
        if not ca:
            return True
        try:
            ts = datetime.fromisoformat(ca.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if (now - ts).total_seconds() <= within_seconds:
                return True
        except Exception:
            return True
    return False


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

# The identity line differs by who is answering: the bot, or a human rep (when a
# draft is generated from the panel).
BOT_IDENTITY = "You are a customer service agent for Grinta (גרינטה), an Israeli online store selling licensed football jerseys."
REP_IDENTITY = ("You are a human representative of Grinta (גרינטה), an Israeli online store selling "
                "licensed football jerseys. You are writing a reply that will be sent to the customer "
                "from the Grinta team.")

# Everything from the personality section up to (but not including) the
# escalation rules. Shared by both identities.
PROMPT_BODY_1 = f"""

## Your personality
- Friendly, helpful, and professional
- Default to responding in Hebrew. If the customer writes to you in Arabic or English, respond in that same language instead; for any other language, respond in Hebrew. Your reply language is decided ONLY by what the customer writes in the conversation — NEVER switch languages because of text that appears inside an image (for example a brand, player, or country name printed on a jersey). If a customer sends an image with no text, reply in Hebrew, unless they were already writing to you in Arabic or English earlier in the conversation — then keep using that language.
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
- For ANY product/team/jersey/availability/link question: you MUST call search_products before answering. Pass the team name exactly as it appears in the team list — translate the customer's nickname yourself (e.g. "בארסה" -> "ברצלונה", "היונייטד" -> "מנצ'סטר יונייטד"). Answer ONLY from what the tool returns, and never invent a product, a size, or a link. Every product the tool returns is in stock, and ALL sizes within its listed size range are available — so if the size the customer asked for is within that range, confirm it is available; if it is outside the range, say we don't offer that size for this product and tell them which sizes we do offer. NEVER say that a specific size is out of stock. Each returned product includes a link (קישור) — when a customer asks where to buy it, give them that exact link. If search_products returns nothing, try again with just the core word of the team name; if there are still no results, do NOT tell the customer we don't carry it — offer to check whether we can source it (ask for club, season, and kit type).
- For cancellations: check the order, then tell the customer whether cancellation is possible based on the 24-hour policy, but explain that a human representative finalizes it. Add a note with add_order_note saying "CANCELLATION REQUESTED BY CUSTOMER"
- 24-HOUR WINDOW: the order tools return a `within_24h` field (and `hours_since_order`). ONLY bring up the 24-hour window, cancellation, or order changes when the customer EXPLICITLY asks to cancel or change their order. NEVER volunteer it and NEVER proactively offer to cancel or change an order — for a delivery, status, or tracking question, do not mention the 24-hour window at all. When the customer DOES ask to cancel or change (size, name, number, adding an item), rely ONLY on `within_24h`: if it is true, the order is still within the window; if it is false, MORE than 24 hours have passed — do NOT tell the customer the change/cancel is allowed; instead say the 24-hour window has passed and you'll check whether it's still possible (per policy). NEVER calculate the elapsed time yourself from dates, and never assume an order was placed "today".
- For returns: tell the customer whether they meet the return conditions, but explain the final approval is done by the team
- NEVER output raw tool results, JSON, dictionaries, code, or HTML to the customer. Always answer in a normal sentence in the conversation's language (Hebrew by default). If a tool returns an error or a lookup fails, explain it plainly to the customer (for example, that you couldn't find an order with that number) — never paste the raw error text.
- NEVER call add_order_note (or any tool) with an order id, order number, or details you invented. Only use an order_id that a get_order_by_number lookup returned in THIS conversation. If you don't have a real order, do a lookup first or ask the customer — do not make up ids or notes.
- TRACKING LINK: to give a customer a tracking link, use ONLY the exact `tracking_url` returned by the order tool. NEVER build, guess, or modify a tracking URL yourself (do not invent domains like tracking.hfd.co.il or append the tracking number to a made-up link). If `tracking_url` is empty/missing, do NOT provide a link at all — give the tracking number if there is one and tell the customer the order was shipped with HFD, or that the link isn't available yet; when there's a genuine delivery problem, escalate to a human.
"""

# Escalation + contact-email sections. Only the BOT gets these — a human
# representative has nobody to escalate to.
PROMPT_ESCALATION = """
## Escalation rules
ONLY escalate (call escalate_to_human) when you genuinely cannot answer or resolve something.
As long as the request is within your knowledge, you can and should handle it yourself — insist on helping the customer and answer them directly. There is NO reason to escalate while you are managing fine.
Escalate only when:
- The request is outside your knowledge and you truly don't have the information to help

Before you escalate: if you do NOT already have the customer's email address, ask for it first (and their name, if you don't have it) so the team can reply by email — unless the customer already gave these earlier in the conversation (e.g. when checking an order). Once you have the email, call collect_contact_email to save it, then escalate. If you already have the email, just escalate.

## Contact email
- Whenever the customer provides their own email address — for ANY reason, including to check an order — call collect_contact_email to save them as a contact, so the team can follow up by email later. Pass their name too if you know it. Do this naturally in the background; no need to announce it.
- Saving a NEW contact requires a name. If collect_contact_email replies that a name is needed (need_name), ask the customer for their full name, then call collect_contact_email again with both the email and the name. (If the customer already exists in our system, the name isn't required.)
"""

# Everything after the escalation/contact sections. Shared by both identities.
PROMPT_BODY_2 = """
## Image handling
- Always reply about an image in the conversation's language (Hebrew by default). The language of any text visible inside the image is irrelevant to your reply language — a jersey with English writing on it does NOT mean you should answer in English.

## Important
- Never discuss competitors
- Never promise things not in the policies
- Never share other customers' information
"""

# The bot answers live chats; the representative prompt is used for panel drafts.
SYSTEM_PROMPT     = BOT_IDENTITY + PROMPT_BODY_1 + PROMPT_ESCALATION + PROMPT_BODY_2
REP_SYSTEM_PROMPT = REP_IDENTITY + PROMPT_BODY_1 + PROMPT_BODY_2

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


def _fetch_image_bytes(url: str):
    """Download an image (Supabase/widget or Zernio attachment URL) so the model
    can see it. Returns (bytes, mime_type) or (None, None) on any failure."""
    if not url:
        return None, None
    try:
        headers = {}
        if "zernio.com" in url and ZERNIO_API_KEY:
            headers["Authorization"] = f"Bearer {ZERNIO_API_KEY}"
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code != 200:
            print(f"[image fetch] {res.status_code} for {url[:80]}")
            return None, None
        mime = (res.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if not mime.startswith("image/"):
            low = url.lower()
            if low.endswith(".png"):
                mime = "image/png"
            elif low.endswith(".webp"):
                mime = "image/webp"
            elif low.endswith(".gif"):
                mime = "image/gif"
            else:
                mime = "image/jpeg"
        return res.content, mime
    except Exception as e:
        print(f"[image fetch] error: {e}")
        return None, None


def _rehost_zernio_image(message_id, session_id: str, url: str) -> None:
    """Download a Zernio attachment (WhatsApp media is auth-protected, so the
    browser can't load it directly) and re-host it on Supabase, then repoint the
    stored message at the public Supabase URL so the panel's <img> can show it.
    Runs in a background thread — keeps the webhook fast."""
    try:
        data, mime = _fetch_image_bytes(url)
        if not data:
            return
        hosted = db.upload_image(session_id, data, mime)
        if hosted:
            db.update_message_image(message_id, hosted)
    except Exception as e:
        print(f"[rehost image] {e}")


def _store_inbound_image(row, session_id: str, image_url: str) -> None:
    """If a just-stored message carries a Zernio image, re-host it in the
    background so the panel can display it."""
    if image_url and row and row.get("id"):
        threading.Thread(
            target=_rehost_zernio_image,
            args=(row["id"], session_id, image_url),
            daemon=True,
        ).start()


def build_history(session_id: str, skip_last_user: bool):
    msgs = db.get_messages(session_id)
    if skip_last_user and msgs and msgs[-1]["role"] == "user":
        msgs = msgs[:-1]

    # Find the most recent customer message that carries an image; we attach the
    # actual image only for that one (so the model can see it) and keep older
    # ones as their text placeholder — cheap, and mirrors the widget which only
    # passes the current turn's image.
    last_img_idx = -1
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user" and msgs[i].get("image_data"):
            last_img_idx = i
            break

    history = []
    for i, m in enumerate(msgs):
        role = "user" if m["role"] == "user" else "model"
        parts = []
        if i == last_img_idx:
            data, mime = _fetch_image_bytes(m.get("image_data"))
            if data:
                parts.append(types.Part.from_bytes(data=data, mime_type=mime))
        text = m.get("content") or ""
        if text:
            parts.append(types.Part(text=text))
        if not parts:
            parts.append(types.Part(text=" "))
        history.append(types.Content(role=role, parts=parts))
    return history


def build_system_instruction(current_page: str | None = None,
                             customer_name: str | None = None,
                             rep_direction: str | None = None,
                             as_rep: bool = False) -> str:
    """System prompt + the team index. as_rep swaps the bot identity for the
    human-representative one (used for panel drafts) and drops the escalation
    and contact-email sections."""
    teams = tools.get_team_index_text()
    instruction = REP_SYSTEM_PROMPT if as_rep else SYSTEM_PROMPT
    if teams:
        instruction += (
            "\n\n## Teams we carry (names exactly as they appear in our catalog)\n"
            "אלה הקבוצות שאנחנו מוכרים עבורן מוצרים. כשלקוח שואל על קבוצה, תרגם בעצמך "
            "את הכינוי לשם המדויק מהרשימה (למשל \"בארסה\" -> \"ברצלונה\") וקרא ל-search_products "
            "עם השם המדויק.\n\n"
            + teams
        )
    if customer_name:
        first = customer_name.split()[0]
        instruction += (
            "\n\n## Customer\n"
            f"The customer's name appears to be: {customer_name}\n"
            f"If this is clearly a real personal name, you MAY address the customer by "
            f"their FIRST name ({first}) naturally where it fits (for example in a greeting) — but "
            "don't overuse it. If it looks like a username/handle, a phone number, an "
            "email, or you are unsure it's a real name, do NOT use it at all."
        )
    if current_page:
        instruction += (
            "\n\n## Current page\n"
            f"The customer is currently viewing this page on the site: {current_page}\n"
            "Use this to understand context. If it is a product page (/products/{handle}), "
            "the handle tells you which product the customer means when they say "
            '"this jersey", "it", "this", etc. — call search_products with that team to get '
            "its details. Only use this when relevant — for general questions, ignore it."
        )
    return instruction


def _plausible_name(name: str | None) -> str:
    """Return the name only if it looks like a real personal name — not a phone
    number, email, or empty. Usernames/handles are left for the model to judge."""
    name = (name or "").strip()
    if not name or "@" in name:
        return ""
    compact = re.sub(r"[\s+\-()]", "", name)
    if compact.isdigit():  # phone number
        return ""
    return name


def gemini_generate(history, current_page: str | None = None, models=None,
                    customer_name: str | None = None, rep_direction: str | None = None,
                    as_rep: bool = False):
    """Try each model in order; fall to the next if one fails (503) or returns empty content."""
    system_instruction = build_system_instruction(current_page, customer_name, rep_direction, as_rep)
    active_tools = tools.REP_TOOLS if as_rep else TOOLS
    last_error = None
    for model_name in (models or MODELS):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=active_tools,
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


def _strip_reasoning(text: str) -> str:
    """Remove chain-of-thought a model may leak into its reply. Conservative:
    it only acts when the text actually carries a reasoning label
    (THOUGHT/THINK/REASONING), so an ordinary reply is never touched.
    - If a final-answer label (RESPONSE:/ANSWER:/REPLY:) follows the reasoning,
      keep only what comes after the last such label.
    - Else, if it opens with a reasoning block ending at a blank line, drop it."""
    if not text:
        return text
    t = text.strip()
    has_reasoning = re.search(r'(?i)\b(?:THOUGHT|THINK|REASONING)\b[ \t]*:?', t)
    if not has_reasoning:
        return t
    ans = list(re.finditer(r'(?i)\b(?:RESPONSE|ANSWER|FINAL ANSWER|REPLY)\b[ \t]*:[ \t]*', t))
    if ans:
        return t[ans[-1].end():].strip()
    m = re.match(r'(?is)^\s*(?:THOUGHT|THINK|REASONING)\b[ \t]*:?', t)
    if m:
        after = t[m.end():]
        blocks = re.split(r'\n[ \t]*\n', after, maxsplit=1)
        if len(blocks) == 2 and blocks[1].strip():
            return blocks[1].strip()
    return t


# Shown to the customer if the model's reply is a raw tool/JSON/HTML blob.
_RAW_FALLBACK = "מצטערים, נתקלנו בבעיה טכנית רגעית. אפשר לנסות שוב או לנסח מחדש את הפנייה?"


def _looks_like_raw_output(text: str) -> bool:
    """True if the reply is a raw tool/JSON/HTML blob that must never reach the
    customer — a leaked dict, error object, or HTML fragment (rather than a
    normal sentence). Kept tight so ordinary replies are never caught."""
    if not text:
        return False
    t = text.strip()
    if re.search(r'</?(?:body|html|head|div|span|table|td|tr|p|br|style)\b', t, re.IGNORECASE):
        return True
    if t.startswith("<") and ">" in t:
        return True
    if re.search(r'\{\s*["\'](?:error|success|order_id|found|message)["\']\s*:', t):
        return True
    if t.startswith(("{", "[")) and re.search(r'["\'](?:error|success|order_id|found)["\']', t):
        return True
    return False


def run_loop(session_id: str, history, current_page: str | None = None, models=None,
             rep_direction: str | None = None, as_rep: bool = False):
    """Run the tool-use loop over a prepared history. Returns (text, escalated)."""
    escalated = False
    # Give the model the customer's name (when we have a plausible one) so it can
    # address them by first name where it fits.
    sess = db.get_session(session_id) or {}
    customer_name = _plausible_name(sess.get("customer_name"))
    for _ in range(5):
        response = gemini_generate(history, current_page, models, customer_name, rep_direction, as_rep)
        content = response.candidates[0].content
        history.append(content)

        parts = content.parts or []
        tool_calls = [p for p in parts if p.function_call is not None]

        if not tool_calls:
            # Only the model's ANSWER goes to the customer — never its reasoning:
            # drop parts marked as "thought", then strip a labelled reasoning
            # block if the model wrote one as plain text.
            text = "".join(
                p.text for p in parts
                if p.text and not getattr(p, "thought", False)
            ).strip()
            text = _strip_reasoning(text)
            if _looks_like_raw_output(text):
                print(f"[raw output blocked] {text[:200]}")
                text = _RAW_FALLBACK
            return text, escalated

        tool_response_parts = []
        for part in tool_calls:
            fc   = part.function_call
            name = fc.name
            args = dict(fc.args)
            print(f"[Tool call] {name}({args})")
            try:
                result = dispatch_tool(name, args)
            except Exception as e:
                print(f"[Tool error] {name}: {e}")
                result = {"error": f"tool {name} failed: {e}"}

            # Side effects that depend on the session / result:
            if name == "escalate_to_human":
                escalated = True
                db.set_status(session_id, "escalated", args.get("reason", ""))
                notify_escalation(session_id, args.get("reason", ""), args.get("summary", ""))
            elif name == "collect_contact_email" and isinstance(result, dict) and result.get("saved"):
                db.set_contact_email(session_id, result.get("email", ""),
                                     result.get("name"), result.get("customer_id"))
            elif name == "get_order_by_email" and isinstance(result, dict):
                # An email used for an order lookup belongs to an existing customer —
                # link the contact automatically (no separate tool call needed).
                cust = result.get("customer") or {}
                em = cust.get("email") or args.get("email")
                if em:
                    db.set_contact_email(session_id, em, cust.get("name"), cust.get("id"))
            elif name == "get_order_by_number" and isinstance(result, dict):
                # An order found by number carries its customer — link them too, so
                # a visitor who only gave an order number still shows an email in
                # the panel. Guest checkouts without an email are left unlinked.
                cust = result.get("customer") or {}
                em = (cust.get("email") or "").strip()
                if em:
                    db.set_contact_email(session_id, em, cust.get("name"), cust.get("id"))

            tool_response_parts.append(types.Part(
                function_response=types.FunctionResponse(
                    name=name, response={"result": result}
                )
            ))
        history.append(types.Content(role="user", parts=tool_response_parts))

    return "מצטערים, נתקלנו בבעיה טכנית. נשמח לעזור לך בדרך אחרת.", escalated


# ─────────────────────────────────────────────
# Reply generation: newest message wins
# ─────────────────────────────────────────────
# Customers often send one thought across several quick messages. Each of those
# would otherwise spawn its own reply, racing each other. Every new customer
# message bumps the session's generation epoch; a reply whose epoch is stale when
# it finishes is discarded, so only the newest message answers — with every
# message already in its history.
# In-memory, like the webhook de-dup: fine for one worker at this volume.
_gen_lock = threading.Lock()
_gen_epoch: dict[str, int] = {}


def bump_generation(session_id: str) -> int:
    """A new customer message arrived — invalidate any reply being generated."""
    with _gen_lock:
        _gen_epoch[session_id] = _gen_epoch.get(session_id, 0) + 1
        return _gen_epoch[session_id]


def current_generation(session_id: str) -> int:
    with _gen_lock:
        return _gen_epoch.get(session_id, 0)


def run_bot_turn(session_id: str, epoch: int | None = None):
    """Generate a bot answer for the current conversation state. Runs in a
    background thread. Saves the assistant reply (so the widget can poll it) and,
    for IG/WhatsApp sessions, also delivers it out through Zernio.

    If a newer customer message arrives while the model is answering, this reply
    is discarded — the newer turn answers with the fuller conversation."""
    my_epoch = current_generation(session_id) if epoch is None else epoch
    try:
        history = build_history(session_id, skip_last_user=False)
        text, escalated = run_loop(session_id, history)

        # A newer message arrived while the model was answering — this reply is
        # stale; the newer turn will answer instead.
        if current_generation(session_id) != my_epoch:
            return
        # If a human took over again while we were generating, discard
        fresh = db.get_session(session_id) or {}
        if fresh.get("status") == "escalated" and not escalated:
            return
        if not text:
            return
        db.add_message(session_id, "assistant", text)
        # IG/WhatsApp: the customer isn't on the widget — send the reply out.
        channel = (fresh.get("channel") or "").lower()
        if channel in ("instagram", "whatsapp"):
            send_zernio_message(fresh.get("zernio_conversation_id") or "",
                                fresh.get("zernio_account_id") or "", text)
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


def _client_ip(request: Request) -> str:
    """Real visitor IP. On Render the app is behind a proxy, so prefer
    X-Forwarded-For (first hop) over the direct socket address."""
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else ""


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    # Silent IP block (widget only): blocked visitors get no reply, no model call.
    ip = _client_ip(request)
    if ip and db.is_ip_blocked(ip):
        return ChatResponse(reply="", session_id=req.session_id)

    db.ensure_session(req.session_id)
    if ip:
        db.set_last_ip(req.session_id, ip)

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

    # This message invalidates any reply still being generated for this session,
    # so if the customer sends several lines in a row only the last one answers.
    my_epoch = bump_generation(req.session_id)

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

    # The customer sent another message while the model was answering — this reply
    # is stale. Stay silent; the newer turn answers with the fuller conversation.
    if current_generation(req.session_id) != my_epoch:
        return ChatResponse(reply="", session_id=req.session_id)

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


def fetch_inbound_attachments(email_id: str) -> list:
    """Resend's webhook carries attachment metadata only. Fetch the list here;
    each item has filename, content_type, size and a signed download_url."""
    if not (RESEND_API_KEY and email_id):
        return []
    try:
        res = requests.get(
            f"https://api.resend.com/emails/receiving/{email_id}/attachments",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
            timeout=20,
        )
        if res.status_code == 200:
            return res.json().get("data") or []
        print(f"[inbound attachments] failed {res.status_code}: {res.text[:200]}")
    except Exception as e:
        print(f"[inbound attachments] error: {e}")
    return []


# Emails carry signature logos and tracking pixels as inline images. Skip tiny
# images so they don't flood the conversation, and cap how many we store.
MAX_EMAIL_ATTACHMENTS = 10
MIN_INLINE_IMAGE_BYTES = 3000


def store_email_attachments(session_id: str, email_id: str) -> None:
    """Download a received email's attachments, re-host them on Supabase, and add
    one message per file so they show in the panel. Runs in a background thread —
    Resend expects the webhook to return quickly."""
    try:
        atts = fetch_inbound_attachments(email_id)
        for att in atts[:MAX_EMAIL_ATTACHMENTS]:
            ctype = (att.get("content_type") or "").lower()
            size  = att.get("size") or 0
            name  = att.get("filename") or "file"
            url   = att.get("download_url")
            if not url:
                continue
            # Drop signature icons / tracking pixels.
            if ctype.startswith("image/") and size and size < MIN_INLINE_IMAGE_BYTES:
                continue
            try:
                r = requests.get(url, timeout=60)
                if r.status_code != 200:
                    print(f"[email attachment] download {r.status_code} for {name}")
                    continue
                hosted = db.upload_file(session_id, r.content,
                                        ctype or "application/octet-stream", name)
                if hosted:
                    db.add_message(session_id, "user", "", hosted)
            except Exception as e:
                print(f"[email attachment] {name}: {e}")
    except Exception as e:
        print(f"[store_email_attachments] {e}")


def parse_contact_form(body: str) -> dict | None:
    """Detect a Shopify contact-form notification and pull the real customer
    out of the body. Returns {name, email, content} or None if not a form.

    The Shopify form body looks like:
        ...
        Name:
        Somsak Rirerm
        אימייל:
        somsak500011@gmail.com
        תוכן:
        שלום, איפה החבילה שלי עכשיו?
    Labels may be Hebrew or English, value on the same or the next line.
    """
    text = body or ""
    # email — value right after an email label (allow value on the next line)
    em = re.search(
        r'(?:אימייל|מייל|דוא"?ל|דואל|e-?mail|email)\s*:\s*([^\s@]+@[^\s@]+\.[^\s@]+)',
        text, re.IGNORECASE,
    )
    # content — everything after a content label to the end
    cm = re.search(
        r'(?:^|\n)\s*(?:תוכן|הודעה|message|body|comment)\s*:\s*(.*)$',
        text, re.IGNORECASE | re.DOTALL,
    )
    if not (em and cm):
        return None  # not a recognizable contact-form email

    email   = em.group(1).strip().lower()
    content = cm.group(1).strip()
    if not content:
        return None

    nm = re.search(r'(?:^|\n)\s*(?:name|שם|שם מלא)\s*:\s*(.+)', text, re.IGNORECASE)
    name = nm.group(1).strip() if nm else ""
    name = re.sub(r"[\u200e\u200f\u202a-\u202e]", "", name).strip()
    if "@" in name:
        name = ""

    return {"name": name, "email": email, "content": content}


def _html_to_text(html: str) -> str:
    """Reduce an HTML email body to readable plain text. Drops <style>/<script>,
    converts <br> and block ends to newlines, strips all tags, unescapes entities,
    and collapses excess blank lines."""
    import html as _htmllib
    s = html or ""
    # remove style/script/head blocks entirely (where the huge markup lives)
    s = re.sub(r"(?is)<(style|script|head)[^>]*>.*?</\1>", " ", s)
    # line breaks for common block boundaries
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?i)</(p|div|tr|li|h[1-6]|table)>", "\n", s)
    # drop all remaining tags
    s = re.sub(r"(?s)<[^>]+>", "", s)
    s = _htmllib.unescape(s)
    # tidy whitespace
    s = re.sub(r"[ \t\u00a0]+", " ", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


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

    # Display name from the From header ("מאיר כהן <meir@x.com>" -> "מאיר כהן").
    from_name = from_raw.split("<")[0].strip().strip('"').strip() if "<" in from_raw else ""
    from_name = re.sub(r"[\u200e\u200f\u202a-\u202e]", "", from_name).strip()
    if "@" in from_name:  # display name is just the email -> treat as no name
        from_name = ""

    email_id = data.get("email_id") or data.get("id") or ""
    full     = fetch_inbound_email(email_id) if email_id else {}

    subject = (full.get("subject") or data.get("subject") or "").strip() or "(ללא נושא)"

    text_part = (full.get("text") or "").strip()
    html_part = (full.get("html") or "").strip()
    # Use the plain-text part if it's real text. Otherwise (no text part, or a
    # "text" part that's actually raw HTML) convert the HTML to readable text so
    # we never store a giant raw-markup blob.
    if text_part and "<html" not in text_part.lower() and "<body" not in text_part.lower():
        body = text_part
    else:
        body = _html_to_text(html_part or text_part)

    # If this is a Shopify contact-form submission, the real customer is INSIDE
    # the body (the From is shopify's mailer). Use the parsed customer instead,
    # so each customer becomes their own thread and replies go to them.
    form = parse_contact_form(body)
    shopify_cid = None
    if form:
        from_email = form["email"]
        from_name  = form["name"]
        subject    = "פנייה מטופס יצירת קשר"
        content    = form["content"]
        # Form submissions always link to a Shopify contact (find or create).
        try:
            contact = tools.collect_contact_email(from_email, from_name)
            if contact.get("saved"):
                shopify_cid = contact.get("customer_id")
        except Exception as e:
            print(f"[form contact] error: {e}")
    else:
        content = body

    stored = f"נושא: {subject}\n\n{content}"

    session_id = f"email-{from_email}"
    db.ensure_session(session_id)
    # Form submits are a website-origin channel ('form'); direct mail is 'email'.
    db.set_email_meta(session_id, from_email, subject, from_name,
                      channel=("form" if form else "email"),
                      shopify_customer_id=shopify_cid)
    db.add_message(session_id, "user", stored)
    db.set_status(session_id, "escalated", "פנייה במייל")

    # Attachments come from a separate Resend API — fetch them in the background
    # so the webhook returns fast. Each becomes its own message in the panel.
    if email_id and not form:
        threading.Thread(target=store_email_attachments,
                         args=(session_id, email_id), daemon=True).start()
    return {"ok": True}


@app.post("/zernio/inbound")
async def zernio_inbound(req: Request):
    """Inbound Instagram/WhatsApp messages from Zernio. Verifies the HMAC
    signature, de-dups by event id, and stores incoming DMs as escalated
    sessions keyed ig-{id} / wa-{id} — same shape as the email inbox."""
    raw = await req.body()
    if not _zernio_verify(raw, req.headers.get("X-Zernio-Signature", "")):
        raise HTTPException(status_code=401, detail="bad signature")

    try:
        payload = json.loads(raw)
    except Exception:
        return {"ok": True}

    if payload.get("event") not in ("message.received", "message.sent"):
        return {"ok": True}  # ignore delivery receipts, reactions, etc.

    event_id = payload.get("id") or req.headers.get("X-Zernio-Event-Id", "")
    if _zernio_already_seen(event_id):
        return {"ok": True}

    msg     = payload.get("message") or {}
    convo   = payload.get("conversation") or {}
    account = payload.get("account") or {}

    channel = ZERNIO_PLATFORMS.get((msg.get("platform") or "").lower())
    if not channel:
        return {"ok": True}

    # Outgoing = a message WE sent (panel reply, Zernio inbox, or synced phone).
    is_outgoing = (payload.get("event") == "message.sent"
                   or msg.get("direction") == "outgoing")

    text = (msg.get("text") or "").strip()

    # First image attachment (if any) — store its URL for display in the inbox.
    image_url = None
    for att in (msg.get("attachments") or []):
        atype = (att.get("type") or "").lower()
        if atype.startswith("image") or atype == "photo":
            image_url = att.get("url")
            break

    # participantId is always the CUSTOMER (the other party), regardless of
    # direction — so our messages land in the same session as theirs.
    participant_id   = convo.get("participantId") or ""
    participant_name = (convo.get("participantName")
                        or convo.get("participantUsername") or "")
    conversation_id  = msg.get("conversationId") or convo.get("id") or ""
    account_id       = account.get("id") or account.get("_id") or ""

    prefix = "ig" if channel == "instagram" else "wa"
    session_id = f"{prefix}-{participant_id or conversation_id}"
    if session_id in (f"{prefix}-", prefix):
        return {"ok": True}  # nothing to key on

    if is_outgoing:
        # A message we sent. If it echoes a reply we already stored ourselves
        # (panel reply or bot reply), skip it to avoid a duplicate. If it came
        # from elsewhere (Zernio inbox / phone), store it as a human reply.
        existed = db.get_session(session_id) is not None
        db.ensure_session(session_id)
        db.set_channel_meta(session_id, channel, conversation_id, account_id,
                            participant_name)
        if _recent_outgoing_echo(session_id, text):
            return {"ok": True}
        row = db.add_message(session_id, "human",
                             text or ("[נשלחה תמונה]" if image_url else ""), image_url)
        _store_inbound_image(row, session_id, image_url)
        # A conversation we started ourselves is human-handled from the start.
        if not existed:
            db.set_status(session_id, "escalated", "שיחה שנפתחה על ידך")
        return {"ok": True}

    # Incoming customer message — handled by the bot exactly like the widget:
    # the bot answers automatically, escalates when needed, and stays silent
    # while a human is handling the thread.
    db.ensure_session(session_id)

    # Read handling state BEFORE storing the new message (which bumps the timer).
    session_before = db.get_session(session_id) or {}
    was_escalated = session_before.get("status") == "escalated"

    # Auto-handback: escalated but quiet for over 1 hour -> back to the bot.
    auto_handback = False
    if was_escalated and _is_stale(session_before.get("updated_at"), 1):
        db.set_status(session_id, "bot")
        auto_handback = True

    db.set_channel_meta(session_id, channel, conversation_id, account_id,
                        participant_name)
    stored = text or ("[התקבלה תמונה]" if image_url else "")
    row = db.add_message(session_id, "user", stored, image_url)
    _store_inbound_image(row, session_id, image_url)

    # This message invalidates any reply still being generated for this session.
    epoch = bump_generation(session_id)

    # A human is handling it -> bot stays silent; the message still shows in the
    # panel (unread) so the rep sees it.
    if was_escalated and not auto_handback:
        return {"ok": True}

    # Otherwise let the bot answer in the background (webhooks must return fast).
    threading.Thread(target=run_bot_turn, args=(session_id, epoch), daemon=True).start()
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


MAX_REPLY_IMAGES = 5

# Channels that accept non-image files (pdf, doc...). Instagram's API has no
# document message type, and the widget only renders images.
FILE_CHANNELS = ("whatsapp", "email")


def _zernio_attachment_type(mime: str) -> str:
    """Map a mime type to Zernio's attachmentType."""
    m = (mime or "").lower()
    if m.startswith("image/"):
        return "image"
    if m.startswith("video/"):
        return "video"
    if m.startswith("audio/"):
        return "audio"
    return "document"


class ImageAttachment(BaseModel):
    base64: str
    mime: str = "image/jpeg"
    name: str = ""


class ReplyRequest(BaseModel):
    session_id: str
    content: str
    via: str = "widget"
    draft: str = ""
    images: list[ImageAttachment] = []


@app.get("/admin/api/sessions")
def admin_sessions(only_escalated: bool = False, days: int = 7,
                   date_from: str = "", date_to: str = "", search: str = "",
                   _: bool = Depends(check_admin)):
    return {"sessions": db.list_sessions(only_escalated, days,
                                         date_from or None, date_to or None,
                                         search or None)}


@app.get("/admin/api/messages")
def admin_messages(session_id: str, _: bool = Depends(check_admin)):
    msgs = db.get_messages(session_id)
    db.mark_read(session_id)
    return {"messages": msgs}


@app.post("/admin/api/reply")
def admin_reply(req: ReplyRequest, _: bool = Depends(check_admin)):
    imgs = req.images or []
    if len(imgs) > MAX_REPLY_IMAGES:
        return {"ok": False, "error": f"max {MAX_REPLY_IMAGES} files"}
    if not (req.content or imgs):
        return {"ok": False, "error": "empty message"}

    via  = (req.via or "widget").lower()
    sess = db.get_session(req.session_id) or {}
    channel = (sess.get("channel") or "").lower()
    email = (sess.get("customer_email") or "").strip()
    if not email and req.session_id.startswith("email-"):
        email = req.session_id[len("email-"):].strip()

    # Only WhatsApp and email can carry non-image files.
    if channel not in FILE_CHANNELS:
        for img in imgs:
            if not (img.mime or "").lower().startswith("image/"):
                return {"ok": False, "error": "בערוץ הזה אפשר לצרף תמונות בלבד"}

    # Upload every attachment so each channel can reference a public URL.
    uploads = []   # [(url, mime)]
    for img in imgs:
        try:
            data = base64.b64decode(img.base64)
            url = db.upload_file(req.session_id, data, img.mime or "image/jpeg", img.name)
        except Exception as e:
            print(f"[admin attachment] {e}")
            url = None
        if not url:
            return {"ok": False, "error": "file upload failed"}
        uploads.append((url, img.mime or "image/jpeg"))

    # The reply becomes one message per attachment, because WhatsApp/Instagram
    # send one attachment per message. WhatsApp supports a caption, so the text
    # rides with the first one. Instagram does NOT allow text + attachment in one
    # message (Meta drops the text), so there the text is sent on its own first.
    if uploads:
        if channel == "instagram" and req.content:
            parts = [(req.content, None, None)] + [("", u, m) for u, m in uploads]
        else:
            parts = ([(req.content, uploads[0][0], uploads[0][1])]
                     + [("", u, m) for u, m in uploads[1:]])
    else:
        parts = [(req.content, None, None)]

    for text, url, _mime in parts:
        db.add_message(req.session_id, "human", text, url)

    # Instagram / WhatsApp: deliver through Zernio — one message per part.
    if channel in ("instagram", "whatsapp"):
        conv = sess.get("zernio_conversation_id") or ""
        acct = sess.get("zernio_account_id") or ""
        ok = True
        for text, url, mime in parts:
            if not send_zernio_message(conv, acct, text, image_url=url, mime=mime):
                ok = False
        return {"ok": True, "sent": ok}

    # A pure email channel has no widget — always deliver by mail.
    force_mail = channel == "email"

    # "Reply by mail" sends the email AND keeps the widget copy above.
    # "Reply in widget" stores only (no mail), even if an email exists.
    if (via == "mail" or force_mail) and email:
        msgs = db.get_messages(req.session_id)
        last_user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
        quote = _clean_quote(last_user)
        # One email carrying all the images as attachments.
        ok = send_customer_email(
            email,
            sess.get("email_subject") or "פנייתך ל-Grinta",
            req.content,
            quote=quote,
            images=imgs,
        )
        return {"ok": True, "emailed": ok}

    # Widget: nothing to send — the visitor polls and picks up image_data.
    return {"ok": True, "emailed": False}


@app.post("/admin/api/generate")
def admin_generate(req: ReplyRequest, _: bool = Depends(check_admin)):
    # Build history from the conversation, then append an explicit instruction
    # so the model always has a concrete task — otherwise, if the conversation
    # already ends with a reply, the model returns an empty STOP completion.
    history = build_history(req.session_id, skip_last_user=False)
    if not history:
        return {"draft": ""}

    sess = db.get_session(req.session_id) or {}
    # Detect email sessions robustly: the channel flag OR the session-id prefix
    # (email sessions are always keyed "email-{address}"). This survives cases
    # where channel/customer_email weren't stored.
    is_email = sess.get("channel") == "email" or req.session_id.startswith("email-")

    if is_email:
        name  = (sess.get("customer_name") or "").strip()
        # fall back to the address embedded in the session id if not stored
        email = (sess.get("customer_email") or "").strip()
        if not email and req.session_id.startswith("email-"):
            email = req.session_id[len("email-"):].strip()
        name_line = f'- שם הפונה: {name}' if name else '- שם הפונה: לא ידוע'
        greet_hint = (f'פתח בפנייה אישית בשמו (למשל "היי {name}," או "שלום {name},").'
                      if name else 'פתח בברכה כללית (למשל "היי," או "שלום,"). '
                                   'לעולם אל תשתמש בכתובת מייל או במספר כשם של הלקוח.')
        instruction = (
            "זוהי פנייה במייל. כתוב טיוטת תשובת מייל מלאה ללקוח.\n\n"
            "פרטי הפונה:\n"
            f"{name_line}\n"
            f"- כתובת המייל של הפונה: {email}\n\n"
            "מבנה התשובה:\n"
            f"- {greet_hint}\n"
            "- אחר כך כתוב את גוף התשובה.\n"
            "- סיים בחתימה בשתי שורות: \"בברכה,\" ובשורה הבאה \"צוות גרינטה\".\n\n"
            "בדיקת הזמנות:\n"
            f"- כתובת המייל של הפונה ({email}) ידועה לך. השתמש בה לבדיקת הזמנות לפי מייל, "
            "וכן בכל פרט מזהה אחר שהלקוח מוסר (מספר הזמנה, שם, וכו') כדי לאתר את ההזמנה הרלוונטית.\n"
            "- אם נמצאו כמה הזמנות, התייחס להזמנה האחרונה לפי התאריך, אלא אם הלקוח מתייחס לאחת ספציפית.\n"
            "- אם הלקוח מסר מספר הזמנה ולא נמצאה הזמנה תואמת — אל תבקש שוב את אותו מספר; הסבר לו שלא מצאת "
            "הזמנה עם המספר הזה ובקש שיוודא את הפרטים.\n"
            "- אם אין בידך שום פרט מזהה ולא נמצאה הזמנה לפי המייל — בקש מהלקוח פרט מזהה (מספר הזמנה "
            "או הכתובת ששימשה בהזמנה).\n\n"
            "החזר רק את נוסח תשובת המייל, ללא הקדמות או הסברים."
        )
    else:
        instruction = (
            "השיחה שלמעלה היא בין הלקוח לבין הבוט. המקרה הופנה כעת אליך — נציג אנושי. "
            "על סמך השיחה, כתוב את תשובת הנציג האנושי ללקוח. "
            "החזר רק את נוסח התשובה ללקוח, ללא הקדמות או הסברים."
        )

    # The hint field (req.content) is the representative's DIRECTION. It goes into
    # the task turn at the END of contents (the last thing the model reads before
    # writing), so it steers the reply instead of competing with the conversation.
    #  - Reply box empty (no draft): direction steers a fresh draft.
    #  - Reply box has a draft: direction is an instruction to edit that draft.
    direction = (req.content or "").strip()
    existing_draft = (req.draft or "").strip()
    if existing_draft:
        instruction = (
            "להלן טיוטת התשובה הנוכחית שהוכנה ללקוח:\n\n"
            f"<טיוטה>\n{existing_draft}\n</טיוטה>\n\n"
            "ערוך את הטיוטה לפי ההנחיה הבאה — שנה רק את מה שההנחיה מבקשת, "
            "השאר את שאר הטקסט כפי שהוא, ושמור על שפת הטיוטה.\n\n"
            f"הנחיה: {direction}\n\n"
            "החזר רק את נוסח התשובה המעודכן ללקוח, ללא הקדמות, הסברים או תגיות."
        )
    elif direction:
        instruction += (
            f"\n\nהנחיית הנציג לתשובה: {direction}\n"
            "כתוב את תשובת הנציג האנושי בהתאם להנחיה הזו ולשיחה שלמעלה."
        )

    history.append(types.Content(role="user", parts=[types.Part(text=instruction)]))
    try:
        text, _ = run_loop(req.session_id, history,
                           models=GENERATE_MODELS if is_email else None,
                           as_rep=True)
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


class IpRequest(BaseModel):
    ip: str


@app.get("/admin/api/session_meta")
def admin_session_meta(session_id: str, _: bool = Depends(check_admin)):
    sess = db.get_session(session_id) or {}
    ip = (sess.get("last_ip") or "").strip()
    return {
        "channel": sess.get("channel") or "",
        "last_ip": ip,
        "ip_blocked": db.is_ip_blocked(ip) if ip else False,
    }


@app.post("/admin/api/block_ip")
def admin_block_ip(req: ReplyRequest, _: bool = Depends(check_admin)):
    sess = db.get_session(req.session_id) or {}
    ip = (sess.get("last_ip") or "").strip()
    if not ip:
        return {"ok": False, "error": "no ip on session"}
    ok = db.block_ip(ip, req.content or None)
    return {"ok": ok, "ip": ip, "error": None if ok else "db write failed"}


@app.post("/admin/api/unblock_ip")
def admin_unblock_ip(req: IpRequest, _: bool = Depends(check_admin)):
    db.unblock_ip(req.ip)
    return {"ok": True}


@app.post("/admin/api/mark_unread")
def admin_mark_unread(req: ReplyRequest, _: bool = Depends(check_admin)):
    db.mark_unread(req.session_id)
    return {"ok": True}


@app.post("/admin/api/clear_session")
def admin_clear_session(req: ReplyRequest, _: bool = Depends(check_admin)):
    ok = db.clear_session(req.session_id)
    return {"ok": ok}


@app.get("/admin/api/blocklist")
def admin_blocklist(_: bool = Depends(check_admin)):
    return {"ips": db.list_blocked_ips()}


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
  .rangefilter button { font-size:12px; padding:7px 0; }
  .rangefilter button.active { background:#fff7e0 !important; color:#7a5a12 !important; border-color:var(--gold) !important; font-weight:700; }
  .customRange { padding:10px; border-bottom:1px solid #eee; background:#fafafa; }
  .customRange .row { display:flex; gap:8px; align-items:center; margin-bottom:8px; }
  .customRange label { font-size:12px; color:#555; width:48px; }
  .customRange input { flex:1; padding:6px 8px; border:1px solid #d8d8d8; border-radius:8px; font-size:12px; font-family:inherit; }
  .customRange .apply { width:100%; padding:8px; border:none; background:var(--gold); color:#0a0a0a; border-radius:8px; font-weight:700; cursor:pointer; }
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
  .m { max-width:70%; min-width:0; padding:10px 14px; border-radius:14px; font-size:14px; line-height:1.5; white-space:pre-wrap; overflow-wrap:anywhere; word-break:break-word; }
  .m img { max-width:100%; height:auto; }
  .m.user { align-self:flex-start; background:#fff; border:1px solid #e2e2e2; }
  .m.assistant { align-self:flex-end; background:var(--black); color:#fff; }
  .m.human { align-self:flex-end; background:var(--gold); color:#0a0a0a; }
  .m .who { font-size:10px; opacity:.6; margin-bottom:3px; }
  .composer { display:flex; flex-wrap:wrap; align-items:flex-end; gap:8px; padding:12px; background:#fff; border-top:1px solid #e2e2e2; }
  .composer input, .composer textarea { flex:1; padding:11px 14px; border:1px solid #d8d8d8; border-radius:18px; font-family:inherit; font-size:14px; resize:vertical; line-height:1.5; }
  .composer button { padding:0 18px; height:44px; border:none; border-radius:22px; background:var(--gold); cursor:pointer; font-weight:700; }
  .composer .hb { background:#eee; }
  .empty { margin:auto; color:#999; }
</style>
</head>
<body>
<header>Grinta — <b>תיבת פניות</b></header>
<div class="wrap">
  <div class="list">
    <div style="padding:9px;border-bottom:1px solid #eee;">
      <div style="display:flex;align-items:center;gap:8px;border:1px solid #ddd;border-radius:8px;padding:6px 10px;background:#fafafa;">
        <span style="color:#999;font-size:14px;">🔍</span>
        <input id="searchBox" oninput="onSearchInput()" placeholder="חפש לפי מייל, שם או תוכן הודעה…" style="flex:1;border:none;background:transparent;outline:none;font-size:13px;font-family:inherit;">
        <span id="searchClear" onclick="clearSearch()" style="display:none;cursor:pointer;color:#999;font-size:14px;">✕</span>
      </div>
    </div>
    <div class="filter">
      <button id="f-esc" class="active" onclick="setFilter(true)">דורש טיפול</button>
      <button id="f-all" onclick="setFilter(false)">הכל</button>
      <button id="f-blocked" onclick="showBlocklist()" title="רשימת חסומים" style="flex:0 0 auto;padding:8px 11px;border:1px solid #e2b4b4;background:#fff5f5;color:#b00;border-radius:8px;cursor:pointer;">🚫</button>
    </div>
    <div class="filter rangefilter">
      <button id="r-7" class="active" onclick="setRange(7)">7 ימים</button>
      <button id="r-30" onclick="setRange(30)">30</button>
      <button id="r-90" onclick="setRange(90)">90</button>
      <button id="r-custom" onclick="setRange(null)">מותאם</button>
    </div>
    <div class="customRange" id="customRange" style="display:none;">
      <div class="row"><label>מתאריך</label><input type="date" id="cfrom"></div>
      <div class="row"><label>עד תאריך</label><input type="date" id="cto"></div>
      <button class="apply" onclick="applyCustom()">החל טווח</button>
    </div>
    <div id="sessions"></div>
  </div>
  <div class="conv">
    <div id="convhead" style="display:none;padding:6px 14px;background:#fff;border-bottom:1px solid #eee;justify-content:space-between;align-items:center;gap:10px;">
      <span id="ipbar" style="font-size:12px;color:#555;display:flex;align-items:center;gap:8px;"></span>
      <span style="display:flex;gap:8px;">
        <button onclick="markUnread()" style="font-size:12px;background:#fff;border:1px solid #ddd;border-radius:7px;padding:4px 10px;cursor:pointer;color:#555;white-space:nowrap;">✉️ סמן כלא נקרא</button>
        <button onclick="clearSession()" style="font-size:12px;background:#fff5f5;border:1px solid #e2b4b4;border-radius:7px;padding:4px 10px;cursor:pointer;color:#b00;white-space:nowrap;">🧹 נקה שיחה</button>
      </span>
    </div>
    <div id="pagebar" style="display:none;padding:8px 14px;background:#fff;border-bottom:1px solid #eee;font-size:13px;color:#555;"></div>
    <div class="msgs" id="msgs"><div class="empty">בחר שיחה מהרשימה</div></div>
    <div class="composer">
      <label id="mailToggleWrap" style="width:100%;display:none;justify-content:flex-end;align-items:center;gap:7px;font-size:12px;padding:2px 6px 4px;cursor:pointer;">
        <input type="checkbox" id="mailToggle" style="width:15px;height:15px;cursor:pointer;">
        <span id="mailToggleLabel"></span>
      </label>
      <div id="hintWrap" style="width:100%;">
        <div style="font-size:12px;color:#888;margin-bottom:4px;display:flex;align-items:center;gap:5px;"><span style="color:var(--gold);">✨</span> הנחיה לסוכן</div>
        <input id="hintBox" placeholder="למשל: תגיד לו שההזמנה תישלח מחר" style="width:100%;box-sizing:border-box;padding:9px 12px;border:1px solid #e0d3b0;background:#fffdf5;border-radius:12px;font-family:inherit;font-size:13px;outline:none;">
      </div>
      <div id="imgPreview" style="width:100%;display:none;padding:0 2px 6px;gap:6px;flex-wrap:wrap;"></div>
      <input id="imgFile" type="file" accept="image/*" multiple style="display:none" onchange="onAdminImagePicked(this)">
      <textarea id="reply" placeholder="כתוב תשובה ללקוח..." rows="2" oninput="autoGrow(this)" style="max-height:170px;overflow-y:auto;"></textarea>
        <button id="attachBtn" onclick="document.getElementById('imgFile').click()" title="צרף תמונה" style="background:#eee;padding:0 14px;display:flex;align-items:center;justify-content:center;"><svg viewBox="0 0 24 24" style="width:20px;height:20px;fill:#555;"><path d="M16.5 6v11.5a4 4 0 01-8 0V5a2.5 2.5 0 015 0v10.5a1 1 0 01-2 0V6H10v9.5a2.5 2.5 0 005 0V5a4 4 0 00-8 0v12.5a5.5 5.5 0 0011 0V6h-1.5z"/></svg></button>
        <button id="genBtn" onclick="generateDraft()" title="אפשר לכתוב הנחיה קצרה בתיבה לפני הלחיצה — הסוכן יכתוב את הטיוטה לפיה" style="background:#0a0a0a;color:var(--gold)">✨ צור טיוטה</button>
        <button id="sendBtn" onclick="sendReply()">שלח</button>
        <button class="hb" id="toggleBtn" style="display:none">✋ קח שליטה</button>
    </div>
  </div>
</div>
<script>
  let onlyEsc = true;
  let current = null;
  let sessionsData = [];
  let viewingBlocklist = false;
  let rangeDays = 7;       // 7 / 30 / 90, or null when a custom range is active
  let customFrom = '';
  let customTo = '';

  function setFilter(esc){
    onlyEsc = esc;
    document.getElementById('f-esc').classList.toggle('active', esc);
    document.getElementById('f-all').classList.toggle('active', !esc);
    loadSessions();
  }

  function setRange(d){
    rangeDays = d;
    ['7','30','90','custom'].forEach(k=>{
      document.getElementById('r-'+k).classList.toggle('active', (k==='custom' ? d===null : String(d)===k));
    });
    document.getElementById('customRange').style.display = (d===null ? 'block' : 'none');
    if(d !== null) loadSessions();   // presets apply immediately; custom waits for "החל טווח"
  }

  function applyCustom(){
    customFrom = document.getElementById('cfrom').value;
    customTo   = document.getElementById('cto').value;
    loadSessions();
  }

  let searchQuery = '';
  let searchTimer = null;

  function onSearchInput(){
    searchQuery = document.getElementById('searchBox').value;
    document.getElementById('searchClear').style.display = searchQuery ? 'inline' : 'none';
    clearTimeout(searchTimer);
    searchTimer = setTimeout(loadSessions, 300);
  }

  function clearSearch(){
    searchQuery = '';
    document.getElementById('searchBox').value = '';
    document.getElementById('searchClear').style.display = 'none';
    loadSessions();
  }

  function searchHighlight(text, q){
    var t = text || '';
    if(!q) return escapeHtml(t);
    var i = t.toLowerCase().indexOf(q.toLowerCase());
    if(i < 0) return escapeHtml(t);
    return escapeHtml(t.slice(0,i))
      + '<mark style="background:#fff3bf;color:#5c4400;padding:0 2px;border-radius:2px;">'
      + escapeHtml(t.slice(i, i+q.length)) + '</mark>'
      + escapeHtml(t.slice(i+q.length));
  }

  function searchSnippet(text, q){
    var t = text || '';
    if(t.indexOf('נושא:') === 0){
      var nl = t.indexOf(String.fromCharCode(10));
      if(nl >= 0) t = t.slice(nl+1).replace(/^[ \t]+/, '');
    }
    var i = t.toLowerCase().indexOf((q||'').toLowerCase());
    if(i < 0) return t.slice(0, 90);
    var start = Math.max(0, i - 30);
    return (start > 0 ? '…' : '') + t.slice(start, i + q.length + 50) + '…';
  }

  function sessionsUrl(){
    let u = '/admin/api/sessions?only_escalated=' + onlyEsc;
    if(rangeDays === null){
      if(customFrom) u += '&date_from=' + customFrom;
      if(customTo)   u += '&date_to=' + customTo;
    } else {
      u += '&days=' + rangeDays;
    }
    if(searchQuery && searchQuery.trim()){
      u += '&search=' + encodeURIComponent(searchQuery.trim());
    }
    return u;
  }

  function loadSessions(){
    fetch(sessionsUrl())
      .then(r=>r.json()).then(d=>{
        sessionsData = d.sessions || [];
        const box = document.getElementById('sessions');
        box.innerHTML = '';
        var sq = searchQuery && searchQuery.trim() ? searchQuery.trim() : '';
        if(sq){
          const cnt = document.createElement('div');
          cnt.style.cssText = 'padding:6px 12px;font-size:11px;color:#888;border-bottom:1px solid #f0f0f0;';
          cnt.textContent = sessionsData.length + ' שיחות נמצאו';
          box.appendChild(cnt);
        }
        sessionsData.forEach(s=>{
          const div = document.createElement('div');
          div.className = 'sess' + (s.session_id===current ? ' active':'');
          div.onclick = ()=>openConv(s.session_id);
        var chan = '';
        if (s.channel === 'email') {
          chan = '<span class="badge" style="background:#e7f6e7;color:#1a7a3a;margin-left:4px">📧 מייל</span>';
        } else if (s.channel === 'form') {
          chan = '<span class="badge" style="background:#eef4ff;color:#2456b8;margin-left:4px">📝 טופס</span>';
        } else if (s.channel === 'instagram') {
          chan = '<span class="badge" style="background:#fce4f3;color:#c026a3;margin-left:4px">📷 אינסטגרם</span>';
        } else if (s.channel === 'whatsapp') {
          chan = '<span class="badge" style="background:#e3f7e8;color:#1b8f4d;margin-left:4px">💬 וואטסאפ</span>';
        } else if (s.customer_email) {
          chan = '<span class="badge" style="background:#e7f6e7;color:#1a7a3a;margin-left:4px">💬+📧</span>';
        }
        var dot = s.unread ? '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#2563eb;margin-left:6px;vertical-align:middle;"></span>' : '';
        var pvStyle = s.unread ? 'font-weight:700;color:#111;' : '';
        var ident = sessionIdentifier(s);
        var previewHtml;
        if(sq && s.match_snippet){
          previewHtml = searchHighlight(searchSnippet(s.match_snippet, sq), sq);
        } else if(sq){
          var identPart = ident ? searchHighlight(ident, sq) + ' — ' : '';
          previewHtml = identPart + escapeHtml(s.last_message || '');
        } else {
          previewHtml = (ident ? ident + ' — ' : '') + (s.last_message || '');
        }
        div.innerHTML =
          '<div class="top"><span>'+dot+chan+'<span class="badge '+s.status+'">'+s.status+'</span></span>'+
          '<span style="font-size:11px;color:#aaa">'+dateTime(s.updated_at)+'</span></div>'+
          '<div class="preview" style="'+pvStyle+'">'+previewHtml+'</div>';
          box.appendChild(div);
        });
        updateToggle();
        updatePageBar();
        updateDelivery();
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
    viewingBlocklist = false;
    var comp = document.querySelector('.composer');
    if(comp) comp.style.display = '';
    document.getElementById('convhead').style.display = 'flex';
    if(current !== id){
      document.getElementById('reply').value = '';
      document.getElementById('hintBox').value = '';
      resetReply();
      clearAdminImage();
    }
    current = id;
    loadSessions();
    loadMsgs();
    updateToggle();
    updatePageBar();
    updateDelivery();
    loadSessionMeta(id);
    updateAttachAccept();
  }

  function loadSessionMeta(id){
    fetch('/admin/api/session_meta?session_id=' + encodeURIComponent(id))
      .then(r=>r.json()).then(renderIpBar).catch(()=>{});
  }

  function renderIpBar(meta){
    const bar = document.getElementById('ipbar');
    if(!bar) return;
    const ch = meta.channel || '';
    const isWidget = (ch === '' || ch === 'web' || ch === 'form');
    if(!isWidget || !meta.last_ip){ bar.innerHTML = ''; return; }
    if(meta.ip_blocked){
      bar.innerHTML = '<span style="color:#b00">🚫 IP חסום: ' + meta.last_ip + '</span>'
        + '<button data-ip="' + meta.last_ip + '" onclick="unblockIp(this.dataset.ip)" style="font-size:12px;background:#143a14;color:#9ad29a;border:1px solid #1d5a1d;border-radius:7px;padding:4px 10px;cursor:pointer">↩︎ בטל חסימה</button>';
    } else {
      bar.innerHTML = '<span>IP: ' + meta.last_ip + '</span>'
        + '<button onclick="blockIp()" style="font-size:12px;background:#3a1414;color:#ff8a8a;border:1px solid #5a1d1d;border-radius:7px;padding:4px 10px;cursor:pointer">🚫 חסום IP</button>';
    }
  }

  function blockIp(){
    if(!current) return;
    fetch('/admin/api/block_ip', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({session_id: current, content: ''})})
      .then(r=>r.json())
      .then(d=>{
        if(!d.ok){ alert('החסימה נכשלה' + (d.error ? ': ' + d.error : '')); }
        loadSessionMeta(current);
      })
      .catch(()=>alert('שגיאה בחסימה'));
  }

  function unblockIp(ip){
    fetch('/admin/api/unblock_ip', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ip: ip})})
      .then(()=>{ if(viewingBlocklist) showBlocklist(); else if(current) loadSessionMeta(current); });
  }

  function markUnread(){
    if(!current) return;
    var id = current;
    fetch('/admin/api/mark_unread', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({session_id: id, content: ''})})
      .then(()=>{
        current = null;
        document.getElementById('convhead').style.display='none';
        document.getElementById('pagebar').style.display='none';
        document.getElementById('msgs').innerHTML = '<div class="empty">בחר שיחה מהרשימה</div>';
        loadSessions();
      })
      .catch(()=>alert('שגיאה'));
  }

  function clearSession(){
    if(!current) return;
    if(!confirm('לנקות את תוכן השיחה? כל ההודעות והתמונות יימחקו — פרטי הפונה יישארו, והשיחה תישאר בפאנל.')) return;
    var id = current;
    fetch('/admin/api/clear_session', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({session_id: id, content: ''})})
      .then(r=>r.json())
      .then(d=>{
        if(!d.ok){ alert('הניקוי נכשל — בדוק את הלוגים'); return; }
        loadMsgs();      // conversation stays open, now empty
        loadSessions();  // row stays in the list; preview clears
      })
      .catch(()=>alert('שגיאה בניקוי'));
  }

  function showBlocklist(){
    viewingBlocklist = true;
    current = null;
    document.getElementById('convhead').style.display='none';
    document.getElementById('pagebar').style.display='none';
    var comp = document.querySelector('.composer');
    if(comp) comp.style.display='none';
    fetch('/admin/api/blocklist').then(r=>r.json()).then(d=>{
      const box = document.getElementById('msgs');
      const ips = d.ips || [];
      let html = '<div style="background:#fff5f5;color:#b00;padding:10px 14px;font-weight:700;border-radius:8px;margin-bottom:8px;">🚫 כתובות IP חסומות (' + ips.length + ')</div>';
      if(!ips.length){ html += '<div class="empty">אין כתובות חסומות</div>'; }
      ips.forEach(b=>{
        html += '<div style="display:flex;justify-content:space-between;align-items:center;padding:11px 12px;border-bottom:1px solid #f3f3f3;">'
          + '<div><div style="font-family:monospace;font-size:13px">' + b.ip + '</div>'
          + '<div style="font-size:11px;color:#999;margin-top:2px">נחסם ' + fmtDate(b.created_at) + (b.reason ? ' · ' + escapeHtml(b.reason) : '') + '</div></div>'
          + '<button data-ip="' + b.ip + '" onclick="unblockIp(this.dataset.ip)" style="font-size:12px;background:#143a14;color:#9ad29a;border:1px solid #1d5a1d;border-radius:7px;padding:5px 10px;cursor:pointer">בטל</button>'
          + '</div>';
      });
      box.innerHTML = html;
    });
  }

  function fmtDate(ts){
    if(!ts) return '';
    try { return new Date(ts).toLocaleDateString('he-IL',{day:'numeric',month:'numeric',timeZone:'Asia/Jerusalem'}); }
    catch(e){ return ''; }
  }

  // A stored attachment is an image (show it) or any other file (link to it).
  function attachmentHtml(url){
    var clean = (url || '').split('?')[0];
    var isImg = /[.](png|jpe?g|webp|gif|bmp|svg)$/i.test(clean);
    if(isImg){
      return '<br><img src="'+url+'" style="max-width:200px;border-radius:8px;margin-top:6px">';
    }
    var name = decodeURIComponent(clean.split('/').pop() || 'קובץ');
    name = name.replace(/^[0-9]{10,}-/, '');   // strip the upload timestamp prefix
    return '<br><a href="'+url+'" target="_blank" rel="noopener noreferrer" '
      + 'style="display:inline-flex;align-items:center;gap:6px;margin-top:6px;padding:7px 10px;'
      + 'background:#fafafa;border:1px solid #ddd;border-radius:8px;text-decoration:none;color:#0a58ca;">'
      + '📄 <span dir="ltr">' + escapeHtml(name) + '</span></a>';
  }

  function loadMsgs(){
    if(viewingBlocklist) return;
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
          if (m.image_data) html += attachmentHtml(m.image_data);
          if (m.created_at) html += '<div style="font-size:10px;opacity:.55;margin-top:4px;text-align:left">'+fmtTime(m.created_at)+'</div>';
          div.innerHTML = html;
          box.appendChild(div);
        });
        if (nearBottom) box.scrollTop = box.scrollHeight;
      });
  }

  function autoGrow(el){
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 170) + 'px';
  }

  function resetReply(){
    var el = document.getElementById('reply');
    if(el){ el.style.height = ''; }
  }

  var MAX_IMAGES = 5;
  var pendingImages = [];   // [{base64, mime, name, dataUrl}] attached to the next reply

  // Only WhatsApp and email conversations can carry non-image files.
  function currentChannel(){
    var s = sessionsData.find(x=>x.session_id===current);
    var ch = (s && s.channel) ? s.channel : '';
    if(!ch && current && current.indexOf('email-')===0) ch = 'email';
    return ch;
  }
  function filesAllowed(){
    var ch = currentChannel();
    return ch === 'whatsapp' || ch === 'email';
  }
  function updateAttachAccept(){
    var f = document.getElementById('imgFile');
    var b = document.getElementById('attachBtn');
    if(!f) return;
    if(filesAllowed()){
      f.accept = '';
      if(b) b.title = 'צרף תמונה או קובץ';
    } else {
      f.accept = 'image/*';
      if(b) b.title = 'צרף תמונה';
    }
  }

  function isImageMime(m){ return (m||'').indexOf('image/') === 0; }

  function renderImgPreview(){
    var box = document.getElementById('imgPreview');
    if(!pendingImages.length){ box.style.display='none'; box.innerHTML=''; return; }
    box.style.display = 'flex';
    box.innerHTML = '';
    pendingImages.forEach(function(im, i){
      var wrap = document.createElement('span');
      wrap.style.cssText = 'position:relative;display:inline-block;';
      var node;
      if(isImageMime(im.mime)){
        node = document.createElement('img');
        node.src = im.dataUrl;
        node.style.cssText = 'height:64px;width:64px;object-fit:cover;border-radius:8px;border:1px solid #ddd;display:block;';
      } else {
        node = document.createElement('div');
        node.style.cssText = 'height:64px;width:110px;border-radius:8px;border:1px solid #ddd;background:#fafafa;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;padding:0 6px;';
        node.innerHTML = '<div style="font-size:20px">📄</div>'
          + '<div style="font-size:10px;color:#666;max-width:98px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'
          + escapeHtml(im.name || 'קובץ') + '</div>';
      }
      var x = document.createElement('button');
      x.textContent = '×';
      x.title = 'הסר';
      x.style.cssText = 'position:absolute;top:-7px;left:-7px;width:20px;height:20px;padding:0;border:none;border-radius:50%;background:#0a0a0a;color:#fff;font-size:13px;line-height:1;cursor:pointer;';
      x.onclick = function(){ removeAdminImage(i); };
      wrap.appendChild(node);
      wrap.appendChild(x);
      box.appendChild(wrap);
    });
  }

  function onAdminImagePicked(inp){
    var files = Array.prototype.slice.call(inp.files || []);
    inp.value = '';
    if(!files.length) return;
    if(!filesAllowed()){
      var bad = files.filter(function(f){ return !isImageMime(f.type); });
      if(bad.length){ alert('בשיחה הזו אפשר לצרף תמונות בלבד.'); files = files.filter(function(f){ return isImageMime(f.type); }); }
      if(!files.length) return;
    }
    var room = MAX_IMAGES - pendingImages.length;
    if(room <= 0){ alert('אפשר לצרף עד ' + MAX_IMAGES + ' קבצים.'); return; }
    if(files.length > room){
      alert('אפשר לצרף עד ' + MAX_IMAGES + ' קבצים — נוספו ' + room + ' הראשונים.');
      files = files.slice(0, room);
    }
    files.forEach(function(f){
      if(f.size > 5*1024*1024){ alert('הקובץ "' + f.name + '" גדול מדי (מקסימום 5MB).'); return; }
      var r = new FileReader();
      r.onload = function(e){
        var dataUrl = e.target.result;
        pendingImages.push({ base64: dataUrl.split(',')[1], mime: f.type || 'application/octet-stream',
                             name: f.name || '', dataUrl: dataUrl });
        renderImgPreview();
      };
      r.readAsDataURL(f);
    });
  }

  function removeAdminImage(i){
    pendingImages.splice(i, 1);
    renderImgPreview();
  }

  function clearAdminImage(){
    pendingImages = [];
    document.getElementById('imgFile').value = '';
    renderImgPreview();
  }

  function sendReply(){
    const inp = document.getElementById('reply');
    const text = inp.value.trim();
    if(!text && !pendingImages.length) return;
    if(!current) return;
    const toggle = document.getElementById('mailToggle');
    const via = (toggle && !toggle.disabled && toggle.checked) ? 'mail' : 'widget';
    var body = {session_id: current, content: text, via: via};
    if(pendingImages.length){
      body.images = pendingImages.map(function(im){
        return {base64: im.base64, mime: im.mime, name: im.name};
      });
    }
    inp.value='';
    resetReply();
    clearAdminImage();
    fetch('/admin/api/reply', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(body)})
      .then(r=>r.json())
      .then(d=>{
        if(d.ok === false){ alert('השליחה נכשלה' + (d.error ? ': '+d.error : '')); }
        if(via==='mail' && d.emailed === false){ alert('נשמר, אך שליחת המייל ללקוח נכשלה — בדוק את הלוגים'); }
        if(d.sent === false){ alert('שליחת ההודעה ללקוח נכשלה — ייתכן שחלון 24 השעות נסגר. בדוק את הלוגים.'); }
        loadMsgs();
      })
      .catch(()=>{ alert('שגיאה בשליחה'); loadMsgs(); });
  }

  function currentEmail(){
    const s = sessionsData.find(x=>x.session_id===current);
    let e = (s && s.customer_email) ? s.customer_email : '';
    if(!e && current && current.indexOf('email-')===0){ e = current.substring(6); }
    return e;
  }

  function updateDelivery(){
    const wrap = document.getElementById('mailToggleWrap');
    const toggle = document.getElementById('mailToggle');
    const label = document.getElementById('mailToggleLabel');
    if(!wrap || !toggle) return;
    const s = sessionsData.find(x=>x.session_id===current);
    const channel = (s && s.channel) ? s.channel : '';
    // The mail toggle is only meaningful where a widget exists (website origin):
    // web chat + contact-form. Direct email / IG / WhatsApp deliver one way only.
    const isWidgetType = (channel === '' || channel === 'web' || channel === 'form');
    if(!current || !isWidgetType){ wrap.style.display='none'; return; }
    const e = currentEmail();
    wrap.style.display='flex';
    if(e){
      toggle.disabled=false;
      toggle.checked=true;
      wrap.style.opacity='1';
      label.textContent = "📧 שלח גם למייל: " + e;
    } else {
      toggle.checked=false;
      toggle.disabled=true;
      wrap.style.opacity='.5';
      label.textContent = "📧 שלח גם למייל (אין מייל שמור)";
    }
  }

    function generateDraft(){
      if(!current) return;
      var hint = document.getElementById('hintBox').value.trim();
      var draft = document.getElementById('reply').value.trim();
      // If there's already a draft, the hint is an edit instruction — require one.
      if(draft && !hint){ alert('כתוב בתיבת ההנחיה מה לשנות בטיוטה.'); return; }
      var btn = document.getElementById('genBtn');
      var old = btn.textContent;
      btn.textContent = '...חושב';
      btn.disabled = true;
      fetch('/admin/api/generate', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({session_id: current, content: hint, draft: draft})})
        .then(r=>r.json())
        .then(d=>{
          if(d.draft){ document.getElementById('reply').value = d.draft; autoGrow(document.getElementById('reply')); }
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

  // What to show before the message preview in the list, per channel:
  // WhatsApp -> phone number, Instagram -> the account name/handle, else email.
  function sessionIdentifier(s){
    if(s.channel === 'whatsapp'){
      var ph = (s.session_id || '').replace(/^wa-/, '');
      if(!ph) return s.customer_name || '';
      return ph.charAt(0) === '+' ? ph : '+' + ph;
    }
    if(s.channel === 'instagram'){
      return s.customer_name || (s.session_id || '').replace(/^ig-/, '');
    }
    return s.customer_email || '';
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
  function dateTime(ts){
    if(!ts) return '';
    var lbl = dateLabel(ts);
    if(lbl === 'היום' || lbl === 'אתמול') return lbl + ' ' + fmtTime(ts);
    try {
      var d = new Date(ts).toLocaleDateString('he-IL',{day:'2-digit',month:'2-digit',timeZone:'Asia/Jerusalem'});
      return d + ' ' + fmtTime(ts);
    } catch(e){ return fmtTime(ts); }
  }

  loadSessions();
  setInterval(loadSessions, 5000);
  setInterval(loadMsgs, 4000);
</script>
</body>
</html>
"""
