"""
Supabase data layer for Grinta CS Agent.
Uses the Supabase REST (PostgREST) API directly via requests —
no extra dependency, no version headaches.

Requires two env vars (set on Render):
    SUPABASE_URL   e.g. https://abcd1234.supabase.co
    SUPABASE_KEY   the service_role key (server-side only — keep secret)
"""
import os
import json
import time
import requests
from datetime import datetime, timezone, timedelta

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# Fallback to api.json for local development
if not SUPABASE_URL or not SUPABASE_KEY:
    try:
        with open("api.json", "r") as f:
            _cfg = json.load(f)
        SUPABASE_URL = SUPABASE_URL or _cfg.get("supabase_url", "").rstrip("/")
        SUPABASE_KEY = SUPABASE_KEY or _cfg.get("supabase_key", "")
    except Exception:
        pass

_REST = f"{SUPABASE_URL}/rest/v1"


def _headers(extra: dict = None) -> dict:
    h = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if extra:
        h.update(extra)
    return h


def is_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


# ─────────────────────────────────────────────
# Sessions
# ─────────────────────────────────────────────

def ensure_session(session_id: str) -> None:
    """Create the session row if it doesn't exist yet."""
    url = f"{_REST}/sessions"
    payload = {"session_id": session_id}
    # upsert: ignore if already present
    requests.post(
        url,
        headers=_headers({"Prefer": "resolution=ignore-duplicates,return=minimal"}),
        json=payload,
        timeout=20,
    )


def get_session(session_id: str) -> dict | None:
    url = f"{_REST}/sessions?session_id=eq.{session_id}&select=*"
    res = requests.get(url, headers=_headers(), timeout=20)
    rows = res.json() if res.status_code == 200 else []
    return rows[0] if rows else None


def set_status(session_id: str, status: str, reason: str = None) -> None:
    url = f"{_REST}/sessions?session_id=eq.{session_id}"
    payload = {"status": status, "updated_at": "now()"}
    if reason is not None:
        payload["escalation_reason"] = reason
    requests.patch(
        url,
        headers=_headers({"Prefer": "return=minimal"}),
        json=payload,
        timeout=20,
    )


def touch_session(session_id: str) -> None:
    url = f"{_REST}/sessions?session_id=eq.{session_id}"
    requests.patch(
        url,
        headers=_headers({"Prefer": "return=minimal"}),
        json={"updated_at": "now()"},
        timeout=20,
    )


def set_last_page(session_id: str, url_value: str) -> None:
    """Store the customer's most recent page URL on the session."""
    url = f"{_REST}/sessions?session_id=eq.{session_id}"
    requests.patch(
        url,
        headers=_headers({"Prefer": "return=minimal"}),
        json={"last_page": url_value},
        timeout=20,
    )


def list_sessions(only_escalated: bool = False, days: int = 7,
                  date_from: str = None, date_to: str = None) -> list:
    """List sessions for the admin inbox, filtered by last-activity date.

    - days: only sessions updated within the last N days (default 7).
    - date_from / date_to: an explicit custom range (YYYY-MM-DD). When given,
      these take precedence over `days`. date_to is inclusive of the whole day.
    """
    url = f"{_REST}/session_overview?order=updated_at.desc&limit=200"
    if only_escalated:
        url += "&status=eq.escalated"

    if date_from or date_to:
        # Custom range takes precedence over the day presets.
        if date_from:
            url += f"&updated_at=gte.{date_from}T00:00:00Z"
        if date_to:
            url += f"&updated_at=lte.{date_to}T23:59:59Z"
    elif days and days > 0:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        url += f"&updated_at=gte.{cutoff}"

    res = requests.get(url, headers=_headers(), timeout=20)
    return res.json() if res.status_code == 200 else []


# ─────────────────────────────────────────────
# Messages
# ─────────────────────────────────────────────

def add_message(session_id: str, role: str, content: str, image_data: str = None) -> dict | None:
    """role is one of: user, assistant, human. image_data is an optional image URL."""
    url = f"{_REST}/messages"
    payload = {"session_id": session_id, "role": role, "content": content}
    if image_data:
        payload["image_data"] = image_data
    res = requests.post(
        url,
        headers=_headers({"Prefer": "return=representation"}),
        json=payload,
        timeout=20,
    )
    touch_session(session_id)
    rows = res.json() if res.status_code in (200, 201) else []
    return rows[0] if rows else None


def upload_image(session_id: str, data_bytes: bytes, content_type: str) -> str | None:
    """Upload an image to Supabase Storage, return its public URL."""
    ext = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/gif": "gif"}.get(content_type, "img")
    fname = f"{session_id}/{int(time.time() * 1000)}.{ext}"
    url = f"{SUPABASE_URL}/storage/v1/object/chat-images/{fname}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": content_type,
    }
    res = requests.post(url, headers=headers, data=data_bytes, timeout=30)
    if res.status_code in (200, 201):
        return f"{SUPABASE_URL}/storage/v1/object/public/chat-images/{fname}"
    print(f"[storage] upload failed {res.status_code}: {res.text[:200]}")
    return None


def get_messages(session_id: str) -> list:
    url = f"{_REST}/messages?session_id=eq.{session_id}&order=created_at.asc&select=*"
    res = requests.get(url, headers=_headers(), timeout=20)
    return res.json() if res.status_code == 200 else []


def get_new_messages_after(session_id: str, after_id: int) -> list:
    """Return assistant + human messages newer than after_id (for widget polling)."""
    url = (
        f"{_REST}/messages?session_id=eq.{session_id}"
        f"&role=in.(human,assistant)&id=gt.{after_id}&order=id.asc&select=id,role,content,image_data,created_at"
    )
    res = requests.get(url, headers=_headers(), timeout=20)
    return res.json() if res.status_code == 200 else []

def set_email_meta(session_id: str, customer_email: str, subject: str, customer_name: str = None,
                   channel: str = "email", shopify_customer_id: str = None) -> None:
    url = f"{_REST}/sessions?session_id=eq.{session_id}"
    payload = {
        "channel": channel,
        "customer_email": customer_email,
        "email_subject": subject,
        "updated_at": "now()",
    }
    if customer_name:
        payload["customer_name"] = customer_name
    if shopify_customer_id:
        payload["shopify_customer_id"] = shopify_customer_id
    requests.patch(
        url,
        headers=_headers({"Prefer": "return=minimal"}),
        json=payload,
        timeout=20,
    )


def set_contact_email(session_id: str, customer_email: str, customer_name: str = None,
                      shopify_customer_id: str = None) -> None:
    """Save a contact email/name/Shopify-customer-id onto a session WITHOUT changing
    its channel. Used to capture a web visitor's email so the team can reply by mail."""
    if not customer_email:
        return
    url = f"{_REST}/sessions?session_id=eq.{session_id}"
    payload = {"customer_email": customer_email}
    if customer_name:
        payload["customer_name"] = customer_name
    if shopify_customer_id:
        payload["shopify_customer_id"] = shopify_customer_id
    requests.patch(
        url,
        headers=_headers({"Prefer": "return=minimal"}),
        json=payload,
        timeout=20,
    )
