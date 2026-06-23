import json
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


def get_product(query: str) -> dict:
    url = f"{BASE}/products.json?title={query}&limit=5"
    res = requests.get(url, headers=_headers())
    products = res.json().get("products", [])
    if not products:
        return {"found": False, "message": f"No products found for '{query}'."}
    result = []
    for p in products:
        variants = []
        for v in p["variants"]:
            variants.append({
                "size": v.get("option1", "N/A"),
                "available": (v.get("inventory_quantity") or 0) > 0,
                "price": v["price"]
            })
        result.append({"title": p["title"], "variants": variants})
    return {"found": True, "products": result}


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
            name="get_product",
            description="Search for a product by name or team. Use when a customer asks about availability, sizes, or price of a jersey.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING, description="Product name or team name e.g. 'Barcelona' or 'Real Madrid'"),
                },
                required=["query"]
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
    elif name == "get_product":
        return get_product(**args)
    elif name == "add_order_note":
        return add_order_note(**args)
    elif name == "escalate_to_human":
        return escalate_to_human(**args)
    else:
        return {"error": f"Unknown tool: {name}"}