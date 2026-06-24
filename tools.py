import json
import time
import requests
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

def get_order_by_email(email: str) -> dict:
    url = f"{BASE}/orders.json?email={email}&status=any"
    res = requests.get(url, headers=_headers())
    orders = res.json().get("orders", [])
    if not orders:
        return {"found": False, "message": "No orders found for this email."}
    result = []
    for o in orders:
        tracking = None
        if o.get("fulfillments"):
            tracking = o["fulfillments"][0].get("tracking_number")
        result.append({
            "order_number": o["order_number"],
            "fulfillment_status": o.get("fulfillment_status") or "unfulfilled",
            "financial_status": o.get("financial_status"),
            "created_at": o["created_at"][:10],
            "tracking_number": tracking,
            "items": [i["title"] for i in o["line_items"]]
        })
    return {"found": True, "orders": result}


def get_order_by_number(order_number: str) -> dict:
    clean = order_number.replace("#", "").strip()
    url = f"{BASE}/orders.json?name=%23{clean}&status=any"
    res = requests.get(url, headers=_headers())
    orders = res.json().get("orders", [])
    if not orders:
        return {"found": False, "message": f"Order #{clean} not found."}
    o = orders[0]
    tracking = None
    if o.get("fulfillments"):
        tracking = o["fulfillments"][0].get("tracking_number")
    return {
        "found": True,
        "order_id": str(o["id"]),
        "order_number": o["order_number"],
        "fulfillment_status": o.get("fulfillment_status") or "unfulfilled",
        "financial_status": o.get("financial_status"),
        "created_at": o["created_at"][:10],
        "tracking_number": tracking,
        "items": [i["title"] for i in o["line_items"]]
    }


# ─────────────────────────────────────────────
# Product catalog (injected into the prompt, no tool)
# ─────────────────────────────────────────────

_catalog_cache = {"text": None, "ts": 0}


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
               f"&since_id={since_id}&fields=id,title,tags")
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
            lines.append(f"- {title} | {_describe_product(title, _tags_list(p))}")
        text = "\n".join(lines)
        _catalog_cache["text"] = text
        _catalog_cache["ts"] = now
        return text
    except Exception as e:
        print(f"[catalog] error: {e}")
        return _catalog_cache["text"] or ""


def add_order_note(order_id: str, note: str) -> dict:
    url = f"{BASE}/orders/{order_id}.json"
    payload = {"order": {"id": order_id, "note": note}}
    res = requests.put(url, headers=_headers(), json=payload)
    if res.status_code == 200:
        return {"success": True, "message": "Note added to order successfully."}
    return {"success": False, "message": f"Failed to add note. Status: {res.status_code}"}


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


# ─────────────────────────────────────────────
# Tool schemas (google-genai format)
# ─────────────────────────────────────────────

TOOLS = [
    types.Tool(function_declarations=[
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
    ])
]


# ─────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────

def dispatch_tool(name: str, args: dict) -> dict:
    if name == "get_order_by_email":
        return get_order_by_email(**args)
    elif name == "get_order_by_number":
        return get_order_by_number(**args)
    elif name == "add_order_note":
        return add_order_note(**args)
    elif name == "escalate_to_human":
        return escalate_to_human(**args)
    else:
        return {"error": f"Unknown tool: {name}"}