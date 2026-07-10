import json
import time
import re
import requests
from datetime import datetime, timezone
from google.genai import types

# ─────────────────────────────────────────────
# Load credentials from env vars or api.json fallback
# ─────────────────────────────────────────────
import os

def _load_config():
    if os.environ.get("SHOPIFY_STORE"):
        return {
            "shopify_store":         os.environ["SHOPIFY_STORE"],
            "shopify_client_id":     os.environ["SHOPIFY_CLIENT_ID"],
            "shopify_client_secret": os.environ["SHOPIFY_CLIENT_SECRET"],
        }
    with open("api.json", "r") as f:
        return json.load(f)

_config       = _load_config()
SHOP          = _config["shopify_store"]
CLIENT_ID     = _config["shopify_client_id"]
CLIENT_SECRET = _config["shopify_client_secret"]
API_VERSION   = "2026-01"
BASE          = f"https://{SHOP}/admin/api/{API_VERSION}"


def _get_token() -> str:
    url = f"https://{SHOP}/admin/oauth/access_token"
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials",
    }
    res = requests.post(url, json=payload, timeout=30)
    if res.status_code == 200:
        return res.json().get("access_token")
    raise Exception(f"Token fetch failed ({res.status_code}): {res.text[:300]}")


def _headers() -> dict:
    return {
        "X-Shopify-Access-Token": _get_token(),
        "Content-Type": "application/json"
    }


# ─────────────────────────────────────────────
# Tool functions
# ─────────────────────────────────────────────

def _order_timing(created_at_iso: str) -> dict:
    """From a Shopify created_at ISO timestamp, compute how long ago the order was
    placed and whether it's still inside the 24-hour change/cancel window. Done
    server-side so the model never has to do date math (which it gets wrong).
    Returns {hours_since_order, within_24h}; values are None if unparseable."""
    out = {"hours_since_order": None, "within_24h": None}
    if not created_at_iso:
        return out
    try:
        ts = datetime.fromisoformat(str(created_at_iso).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
        out["hours_since_order"] = round(hours, 1)
        out["within_24h"] = hours < 24
    except Exception as e:
        print(f"[order timing] {e}")
    return out


def _tracking_info(order: dict):
    """Extract (tracking_number, tracking_url) from an order's first fulfillment.
    Shopify stores the real carrier tracking link in tracking_url / tracking_urls;
    we must return it so the agent uses the real link instead of inventing one."""
    fs = order.get("fulfillments") or []
    if not fs:
        return None, None
    f = fs[0]
    number = f.get("tracking_number")
    url = f.get("tracking_url")
    if not url:
        urls = f.get("tracking_urls") or []
        url = urls[0] if urls else None
    return number, url


def get_order_by_email(email: str) -> dict:
    url = f"{BASE}/orders.json?email={email}&status=any"
    res = requests.get(url, headers=_headers())
    orders = res.json().get("orders", [])
    if not orders:
        return {"found": False, "message": "No orders found for this email."}
    cust = orders[0].get("customer") or {}
    customer = {}
    if cust:
        full = " ".join(x for x in [cust.get("first_name"), cust.get("last_name")] if x).strip()
        customer = {
            "id": str(cust["id"]) if cust.get("id") else None,
            "email": cust.get("email") or email,
            "name": full or None,
        }
    result = []
    for o in orders:
        tracking, tracking_url = _tracking_info(o)
        timing = _order_timing(o.get("created_at"))
        result.append({
            "order_number": o["order_number"],
            "fulfillment_status": o.get("fulfillment_status") or "unfulfilled",
            "financial_status": o.get("financial_status"),
            "created_at": o["created_at"][:10],
            "hours_since_order": timing["hours_since_order"],
            "within_24h": timing["within_24h"],
            "tracking_number": tracking,
            "tracking_url": tracking_url,
            "items": [i["title"] for i in o["line_items"]]
        })
    return {"found": True, "orders": result, "customer": customer}


def get_order_by_number(order_number: str) -> dict:
    clean = order_number.replace("#", "").strip()
    url = f"{BASE}/orders.json?name=%23{clean}&status=any"
    res = requests.get(url, headers=_headers())
    orders = res.json().get("orders", [])
    if not orders:
        return {"found": False, "message": f"Order #{clean} not found."}
    o = orders[0]
    tracking, tracking_url = _tracking_info(o)
    timing = _order_timing(o.get("created_at"))
    cust = o.get("customer") or {}
    customer = {}
    if cust:
        full = " ".join(x for x in [cust.get("first_name"), cust.get("last_name")] if x).strip()
        customer = {
            "id": str(cust["id"]) if cust.get("id") else None,
            "email": cust.get("email") or None,
            "name": full or None,
        }
    return {
        "found": True,
        "order_id": str(o["id"]),
        "order_number": o["order_number"],
        "fulfillment_status": o.get("fulfillment_status") or "unfulfilled",
        "financial_status": o.get("financial_status"),
        "created_at": o["created_at"][:10],
        "hours_since_order": timing["hours_since_order"],
        "within_24h": timing["within_24h"],
        "tracking_number": tracking,
        "tracking_url": tracking_url,
        "customer": customer,
        "items": [i["title"] for i in o["line_items"]]
    }


# ─────────────────────────────────────────────
# Product catalog (injected into the prompt, no tool)
# ─────────────────────────────────────────────

_catalog_cache = {"text": None, "ts": 0, "products": None, "pts": 0}


def _tags_list(p) -> list:
    """Shopify REST returns tags as a comma-separated string; normalize to a list."""
    t = p.get("tags", "")
    if isinstance(t, list):
        return [x.strip() for x in t]
    return [x.strip() for x in t.split(",") if x.strip()]


def _describe_product(title: str, tags: list) -> str:
    """Compute sizes + add-on options from the title and tags (Easify variants are not visible)."""
    tagset = set(tags)
    t = title

    is_tracksuit = "אימונית" in t
    is_jacket    = ("ג'קט" in t) or ("ג׳קט" in t) or ("מעיל" in t)
    is_kids      = "ילדים" in t
    is_kids_suit = ("חליפת ילדים" in t) or ("חליפה" in t and is_kids)
    is_pants     = (("מכנס" in t) or ("מכנסיים" in t)) and not is_tracksuit and not is_jacket and ("חולצה" not in t and "חולצת" not in t)
    is_shirt     = ("חולצה" in t) or ("חולצת" in t)
    is_long      = "ארוכה" in t or "ארוכות" in t
    is_women     = "נשים" in t

    # --- sizes ---
    if is_tracksuit:
        sizes = "S עד 2XL"
        if "טווח מידות ילדים ומבוגרים" in tagset:
            sizes = "S עד 2XL וגם 16-28 (ילדים)"
    elif is_jacket:
        sizes = "16-28 (ילדים)" if is_kids else "S עד 2XL"
    elif is_kids_suit:
        sizes = "16-28"
    elif is_pants:
        sizes = "S עד 2XL"
    elif is_shirt:
        if is_long or is_women:
            sizes = "S עד 2XL"
        else:  # men's shirt — may be extended
            sizes = "S עד 4XL" if "טווח מידות מורחב" in tagset else "S עד 2XL"
    else:
        sizes = "S עד 2XL"

    # --- add-on options ---
    opts = []
    if is_shirt and not is_kids_suit:
        if "אופציות מכנס וגרביים קיימות" in tagset:
            opts.append("ניתן להוסיף מכנס וגרביים")
        elif "אופציית מכנסיים קיימת" in tagset:
            opts.append("ניתן להוסיף מכנס")
    elif is_kids_suit or is_pants:
        if "אופציית גרביים קיימת" in tagset:
            opts.append("ניתן להוסיף גרביים")
    elif is_jacket and not is_kids:
        if "אופציית מכנסיים קיימת" in tagset:
            opts.append("ניתן להוסיף מכנס")

    desc = f"מידות: {sizes}"
    if opts:
        desc += " — " + ", ".join(opts)
    return desc


def _fetch_all_products() -> list:
    """Fetch all active products with pagination (Shopify caps at 250 per page)."""
    products = []
    since_id = 0
    for _ in range(20):  # up to 5000 products
        url = (f"{BASE}/products.json?limit=250&status=active"
               f"&since_id={since_id}&fields=id,title,tags,handle")
        res = requests.get(url, headers=_headers(), timeout=30)
        batch = res.json().get("products", [])
        if not batch:
            break
        products.extend(batch)
        if len(batch) < 250:
            break
        since_id = batch[-1]["id"]
    return products


def get_catalog_text(max_age: int = 600) -> str:
    """Return a cached Hebrew catalog string (title + sizes + options per product)."""
    now = time.time()
    if _catalog_cache["text"] and (now - _catalog_cache["ts"] < max_age):
        return _catalog_cache["text"]
    try:
        products = _fetch_all_products()
        lines = []
        for p in products:
            title = (p.get("title") or "").strip()
            if not title:
                continue
            handle = (p.get("handle") or "").strip()
            link = f"https://grinta.co.il/products/{handle}" if handle else ""
            line = f"- {title} | {_describe_product(title, _tags_list(p))}"
            if link:
                line += f" | קישור: {link}"
            lines.append(line)
        text = "\n".join(lines)
        _catalog_cache["text"] = text
        _catalog_cache["ts"] = now
        return text
    except Exception as e:
        print(f"[catalog] error: {e}")
        return _catalog_cache["text"] or ""


# ─────────────────────────────────────────────
# Team index + product search (replaces dumping the whole catalog)
# ─────────────────────────────────────────────

def _cached_products(max_age: int = 600) -> list:
    """Active products, cached — shared by the team index and product search."""
    now = time.time()
    if _catalog_cache["products"] and (now - _catalog_cache["pts"] < max_age):
        return _catalog_cache["products"]
    try:
        products = _fetch_all_products()
        _catalog_cache["products"] = products
        _catalog_cache["pts"] = now
        return products
    except Exception as e:
        print(f"[products] error: {e}")
        return _catalog_cache["products"] or []


# Every team we stock has a plain men's home shirt: "חולצת {קבוצה} בית".
# These three prefixes are other shirt types and must NOT be read as team names.
_TEAM_EXCLUDE = ("חולצת נשים", "חולצה ארוכה", "חולצת אימון")
_TEAM_RE = re.compile(r"^חולצת\s+(.+?)\s+בית\b")


def get_team_index() -> set:
    """The set of team names exactly as they appear in the catalog, derived from
    the men's home-shirt title pattern 'חולצת {קבוצה} בית'. Women's / long-sleeve /
    training shirts are excluded so their words never leak in as a team name."""
    teams = set()
    for p in _cached_products():
        title = (p.get("title") or "").strip()
        if not title or title.startswith(_TEAM_EXCLUDE):
            continue
        m = _TEAM_RE.match(title)
        if m:
            teams.add(m.group(1).strip())
    return teams


def get_team_index_text() -> str:
    """The team list as one comma-separated line, for the system instruction."""
    teams = sorted(get_team_index())
    return ", ".join(teams)


def _product_line(p: dict) -> str:
    """One catalog line for a product: title | sizes + options | link.
    Same format the full catalog used, so nothing downstream changes."""
    title = (p.get("title") or "").strip()
    handle = (p.get("handle") or "").strip()
    line = f"- {title} | {_describe_product(title, _tags_list(p))}"
    if handle:
        line += f" | קישור: https://grinta.co.il/products/{handle}"
    return line


_SEARCH_RULES = (
    "כל מוצר שמופיע ברשימה הזו קיים ובמלאי, וכל המידות בטווח המידות שלו זמינות. "
    "לעולם אל תאמר שמידה מסוימת אזלה מהמלאי. "
    "כשמוסרים ללקוח קישור למוצר — השתמש בקישור המדויק שמופיע כאן, לעולם אל תמציא קישור. "
    "אל תמסור ללקוח את טווח המידות או את אופציות התוספת (מכנס/גרביים) אלא אם הוא שאל עליהן במפורש — "
    "ברשימת מוצרים מסור רק את שם המוצר והקישור. "
    "מסור רק את המוצרים שהלקוח שאל עליהם — לא את כל הרשימה. אם יש עוד סוגי מוצרים לקבוצה, "
    "הצע בקצרה להראות אותם."
)


def search_products(team: str) -> dict:
    """Return every product whose title contains the given team name.
    The agent gets the team names verbatim from the team index, so a plain
    substring match is enough — no normalization needed."""
    team = (team or "").strip()
    if not team:
        return {"found": False, "message": "No team provided."}
    needle = team.casefold()
    lines = [
        _product_line(p) for p in _cached_products()
        if needle in (p.get("title") or "").casefold()
    ]
    if not lines:
        return {
            "found": False,
            "message": ("לא נמצאו מוצרים עבור השם הזה. אל תאמר ללקוח שאיננו מוכרים את הקבוצה — "
                        "נסה לחפש שוב עם מילת הליבה של שם הקבוצה, ואם עדיין אין תוצאות, "
                        "הצע ללקוח לבדוק עבורו (בקש מועדון, עונה וסוג ערכה)."),
        }
    return {"found": True, "rules": _SEARCH_RULES, "count": len(lines),
            "products": "\n".join(lines)}


def _now_israel_stamp() -> str:
    """Timestamp for order notes, in Israel local time."""
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Jerusalem"))
    except Exception:
        now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M")


def add_order_note(order_id: str, note: str) -> dict:
    order_id = str(order_id or "").strip()
    note = (note or "").strip()
    if not note:
        return {"success": False, "message": "Empty note."}
    # Guard against hallucinated ids: order_id must be numeric AND must belong to
    # a real order (verify with a lookup before writing anything to Shopify).
    if not order_id.isdigit():
        return {"success": False,
                "message": "Invalid order id. Do not invent an order id — first find the order with get_order_by_number, then use the order_id it returns."}
    check = requests.get(f"{BASE}/orders/{order_id}.json", headers=_headers())
    if check.status_code != 200:
        return {"success": False,
                "message": "No order exists with that id. Do not guess an order id — look the order up first."}

    # Shopify keeps ONE note field per order, and a PUT replaces it. Read the
    # existing note (null when there is none) and append below it, so earlier
    # notes are never destroyed. Each entry is stamped with the time it was added.
    try:
        existing = (check.json().get("order", {}).get("note") or "").strip()
    except Exception:
        existing = ""
    entry = f"[{_now_israel_stamp()}] {note}"
    combined = f"{existing}\n{entry}" if existing else entry

    url = f"{BASE}/orders/{order_id}.json"
    payload = {"order": {"id": order_id, "note": combined}}
    res = requests.put(url, headers=_headers(), json=payload)
    if res.status_code == 200:
        return {"success": True, "message": "Note added to order successfully."}
    return {"success": False, "message": "Could not add the note right now."}


def escalate_to_human(reason: str, summary: str) -> dict:
    print("\n" + "="*50)
    print("[ESCALATION REQUIRED]")
    print(f"Reason:  {reason}")
    print(f"Summary: {summary}")
    print("="*50 + "\n")
    # TODO: replace with WhatsApp/email notification
    return {
        "escalated": True,
        "message": "הפנייה שלך הועברה לצוות Grinta. נחזור אליך בהקדם האפשרי 🙏"
    }


def find_customer_by_email(email: str) -> dict | None:
    """Return the Shopify customer matching this email, or None."""
    email = (email or "").strip().lower()
    if not email:
        return None
    url = f"{BASE}/customers/search.json?query=email:{email}"
    res = requests.get(url, headers=_headers())
    if res.status_code != 200:
        print(f"[customer search] failed {res.status_code}: {res.text[:200]}")
        return None
    custs = res.json().get("customers", [])
    return custs[0] if custs else None


def create_customer(email: str, name: str = None) -> dict | None:
    """Create a new Shopify customer with email + name. Returns the customer or None."""
    first, last = "", ""
    if name:
        parts = name.strip().split()
        first = parts[0]
        last = " ".join(parts[1:]) if len(parts) > 1 else ""
    payload = {"customer": {"email": email, "first_name": first, "last_name": last}}
    res = requests.post(f"{BASE}/customers.json", headers=_headers(), json=payload)
    if res.status_code in (200, 201):
        return res.json().get("customer")
    print(f"[create_customer] failed {res.status_code}: {res.text[:200]}")
    return None


def collect_contact_email(email: str, name: str = None) -> dict:
    """Link the email to a Shopify customer (contact): find the existing customer,
    or create a new one (which needs the customer's name). The session save happens
    in app.run_loop using the returned customer_id/email/name."""
    email = (email or "").strip().lower()
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return {"saved": False, "message": "כתובת מייל לא תקינה."}

    existing = find_customer_by_email(email)
    if existing:
        full = " ".join(x for x in [existing.get("first_name"), existing.get("last_name")] if x).strip()
        return {
            "saved": True, "existing": True,
            "customer_id": str(existing.get("id")) if existing.get("id") else None,
            "email": existing.get("email") or email,
            "name": full or name,
            "message": "איש הקשר נמצא ונשמר — נוכל לחזור ללקוח במייל.",
        }

    # No existing customer — we must create one, which requires a name.
    if not name or not name.strip():
        return {
            "saved": False, "need_name": True,
            "message": "כדי לשמור איש קשר חדש צריך גם את השם המלא של הלקוח. בקש את שמו ואז קרא לכלי שוב עם השם.",
        }

    created = create_customer(email, name)
    if created:
        return {
            "saved": True, "existing": False,
            "customer_id": str(created.get("id")) if created.get("id") else None,
            "email": email, "name": name.strip(),
            "message": "איש קשר חדש נוצר ונשמר — נוכל לחזור ללקוח במייל.",
        }
    return {"saved": False, "message": "לא הצלחתי לשמור את איש הקשר כרגע."}


# ─────────────────────────────────────────────
# Tool schemas (google-genai format)
# ─────────────────────────────────────────────

TOOLS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="search_products",
            description="Get all Grinta products for a football team (club or national team) — shirts, kids kits, pants, jackets, tracksuits — with their sizes, options and product links. Call this for ANY question about products, teams, jerseys, availability, or product links. Pass the team name exactly as it appears in the team list given in your instructions (translate the customer's nickname yourself, e.g. 'בארסה' -> 'ברצלונה').",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "team": types.Schema(type=types.Type.STRING, description="The team name exactly as it appears in the team list, e.g. 'ריאל מדריד'"),
                },
                required=["team"]
            )
        ),
        types.FunctionDeclaration(
            name="get_order_by_email",
            description="Search for customer orders using their email address. Use when a customer asks about order status, shipping, tracking, or delivery and provides their email.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "email": types.Schema(type=types.Type.STRING, description="The customer's email address"),
                },
                required=["email"]
            )
        ),
        types.FunctionDeclaration(
            name="get_order_by_number",
            description="Find a specific order by order number. Use when a customer provides an order number (with or without #).",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "order_number": types.Schema(type=types.Type.STRING, description="The order number e.g. '1042' or '#1042'"),
                },
                required=["order_number"]
            )
        ),
        types.FunctionDeclaration(
            name="add_order_note",
            description="Add a note to an existing order. Use when a customer requests cancellation or has a special instruction. You must already know the order_id from a previous tool call.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "order_id": types.Schema(type=types.Type.STRING, description="The Shopify internal order ID (not the order number)"),
                    "note": types.Schema(type=types.Type.STRING, description="The note to add to the order"),
                },
                required=["order_id", "note"]
            )
        ),
        types.FunctionDeclaration(
            name="escalate_to_human",
            description="Forward the conversation to the Grinta team. Use when the customer is angry, requests a refund, asks for a human, or after 2 failed resolution attempts.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "reason": types.Schema(type=types.Type.STRING, description="Short reason why escalation is needed"),
                    "summary": types.Schema(type=types.Type.STRING, description="Brief summary of the conversation and what the customer needs"),
                },
                required=["reason", "summary"]
            )
        ),
        types.FunctionDeclaration(
            name="collect_contact_email",
            description="Save the customer's own email address so the Grinta team can follow up by email later. Call this whenever the customer provides their email — including when they give it to check an order, or when you ask for it before escalating. Never call it with an email that is not the customer's own (e.g. an address quoted from somewhere else).",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "email": types.Schema(type=types.Type.STRING, description="The customer's own email address"),
                    "name": types.Schema(type=types.Type.STRING, description="The customer's name, if known (optional)"),
                },
                required=["email"]
            )
        ),
    ])
]


# Tools available when a human representative is drafting a reply from the panel.
# escalate_to_human and collect_contact_email are excluded — the representative
# IS the team, so there is nobody to escalate to and no contact to collect.
_REP_EXCLUDED_TOOLS = {"escalate_to_human", "collect_contact_email"}

REP_TOOLS = [
    types.Tool(function_declarations=[
        fd for fd in TOOLS[0].function_declarations
        if fd.name not in _REP_EXCLUDED_TOOLS
    ])
]


# ─────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────

def dispatch_tool(name: str, args: dict) -> dict:
    if name == "search_products":
        return search_products(**args)
    elif name == "get_order_by_email":
        return get_order_by_email(**args)
    elif name == "get_order_by_number":
        return get_order_by_number(**args)
    elif name == "add_order_note":
        return add_order_note(**args)
    elif name == "escalate_to_human":
        return escalate_to_human(**args)
    elif name == "collect_contact_email":
        return collect_contact_email(**args)
    else:
        return {"error": f"Unknown tool: {name}"}
