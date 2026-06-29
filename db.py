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
from urllib.parse import quote

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
                  date_from: str = None, date_to: str = None,
                  search: str = None) -> list:
    """List sessions for the admin inbox, filtered by last-activity date.

    - days: only sessions updated within the last N days (default 7).
    - date_from / date_to: an explicit custom range (YYYY-MM-DD). When given,
      these take precedence over `days`. date_to is inclusive of the whole day.
    - search: keyword; keeps only sessions whose email/name matches OR that have
      a message containing the keyword. Composes with the date/status filters.
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
    rows = res.json() if res.status_code == 200 else []
    # Merge the unread flag from the sessions table (avoids recreating the view).
    try:
        ur = requests.get(f"{_REST}/sessions?unread=eq.true&select=session_id",
                          headers=_headers(), timeout=20)
        unread_ids = {r["session_id"] for r in ur.json()} if ur.status_code == 200 else set()
        for r in rows:
            r["unread"] = r.get("session_id") in unread_ids
    except Exception as e:
        print(f"[unread merge] {e}")

    # Keyword search: narrows the already-filtered rows by email/name/message text.
    if search and search.strip():
        kw = search.strip()
        kwl = kw.lower()
        snippet = {}
        try:
            mres = requests.get(
                f"{_REST}/messages?content=ilike.*{quote(kw)}*&select=session_id,content&limit=500",
                headers=_headers(), timeout=20,
            )
            if mres.status_code == 200:
                for m in mres.json():
                    sid = m.get("session_id")
                    if sid and sid not in snippet:
                        snippet[sid] = m.get("content", "")
        except Exception as e:
            print(f"[search messages] {e}")
        filtered = []
        for r in rows:
            sid = r.get("session_id")
            email = (r.get("customer_email") or "").lower()
            name = (r.get("customer_name") or "").lower()
            if sid in snippet or kwl in email or kwl in name:
                if sid in snippet:
                    r["match_snippet"] = snippet[sid]
                filtered.append(r)
        rows = filtered

    return rows


def mark_read(session_id: str) -> None:
    requests.patch(
        f"{_REST}/sessions?session_id=eq.{session_id}",
        headers=_headers({"Prefer": "return=minimal"}),
        json={"unread": False},
        timeout=20,
    )


def mark_unread(session_id: str) -> None:
    requests.patch(
        f"{_REST}/sessions?session_id=eq.{session_id}",
        headers=_headers({"Prefer": "return=minimal"}),
        json={"unread": True},
        timeout=20,
    )

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
    # Bump activity; a new CUSTOMER message also marks the thread unread.
    meta = {"updated_at": "now()"}
    if role == "user":
        meta["unread"] = True
    requests.patch(
        f"{_REST}/sessions?session_id=eq.{session_id}",
        headers=_headers({"Prefer": "return=minimal"}),
        json=meta,
        timeout=20,
    )
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


# ─────────────────────────────────────────────
# IP blocking (widget only)
# ─────────────────────────────────────────────

def set_last_ip(session_id: str, ip: str) -> None:
    """Store the visitor's most recent IP on the session (so it can be blocked)."""
    if not ip:
        return
    url = f"{_REST}/sessions?session_id=eq.{session_id}"
    requests.patch(
        url,
        headers=_headers({"Prefer": "return=minimal"}),
        json={"last_ip": ip},
        timeout=20,
    )


def is_ip_blocked(ip: str) -> bool:
    if not ip:
        return False
    url = f"{_REST}/blocklist_ip?ip=eq.{ip}&select=ip"
    res = requests.get(url, headers=_headers(), timeout=20)
    return bool(res.json()) if res.status_code == 200 else False


def block_ip(ip: str, reason: str = None) -> bool:
    if not ip:
        return False
    url = f"{_REST}/blocklist_ip"
    payload = {"ip": ip}
    if reason:
        payload["reason"] = reason
    res = requests.post(
        url,
        headers=_headers({"Prefer": "resolution=merge-duplicates,return=minimal"}),
        json=payload,
        timeout=20,
    )
    if res.status_code in (200, 201, 204):
        return True
    print(f"[block_ip] failed {res.status_code}: {res.text[:200]}")
    return False


def unblock_ip(ip: str) -> None:
    if not ip:
        return
    url = f"{_REST}/blocklist_ip?ip=eq.{ip}"
    requests.delete(url, headers=_headers({"Prefer": "return=minimal"}), timeout=20)


def list_blocked_ips() -> list:
    url = f"{_REST}/blocklist_ip?order=created_at.desc"
    res = requests.get(url, headers=_headers(), timeout=20)
    return res.json() if res.status_code == 200 else []
