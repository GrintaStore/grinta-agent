import json
from pathlib import Path

from google import genai
from google.genai import types
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tools import TOOLS, dispatch_tool

# ─────────────────────────────────────────────
# Load config
# ─────────────────────────────────────────────
import os

def _get_gemini_key():
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    with open("api.json", "r") as f:
        return json.load(f)["gemini_api_key"]

client = genai.Client(api_key=_get_gemini_key())
MODEL  = "gemini-2.5-flash-lite"

# ─────────────────────────────────────────────
# Load knowledge base
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
- Respond in the same language the customer uses (Hebrew or English)
- Be concise — no unnecessary filler text
- Never make up information you don't have

## Your knowledge
{KNOWLEDGE}

## Tool usage rules
- If a customer asks about their order, ALWAYS call get_order_by_email or get_order_by_number before responding
- Never invent order information — only report what the tools return
- If a customer provides an order number, use get_order_by_number
- If a customer provides an email, use get_order_by_email
- If neither is provided, ask the customer for their email or order number first
- If a customer asks about a specific jersey or team availability, call get_product
- If a customer requests cancellation, first check the order status, then add a note with add_order_note saying "CANCELLATION REQUESTED BY CUSTOMER"
- If the order is already shipped, explain it cannot be cancelled and offer the return process instead

## Escalation rules
Call escalate_to_human when:
- The customer is angry or uses aggressive language
- The customer explicitly asks to speak to a human
- The customer requests a refund
- You fail to resolve the issue after 2 attempts
- The request is too complex or unusual

## Image handling
- If a customer sends an image of a jersey, assess if there is visible damage or defect
- If there is a clear defect, apologize and escalate to human immediately

## Important
- Never discuss competitors
- Never promise things not in the policies
- Never share other customers' information
- If unsure, say so and offer to escalate
"""

# ─────────────────────────────────────────────
# Session storage (in-memory)
# ─────────────────────────────────────────────
sessions: dict[str, list] = {}

# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(title="Grinta CS Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str = ""
    image_base64: str | None = None   # base64-encoded image data (no data: prefix)
    image_mime: str | None = None     # e.g. "image/jpeg" or "image/png"


class ChatResponse(BaseModel):
    reply: str
    session_id: str


@app.get("/")
def root():
    return {"status": "Grinta CS Agent is running"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = sessions.get(req.session_id, [])

    # Build the user message parts (text + optional image)
    user_parts = []
    if req.image_base64:
        import base64
        try:
            img_bytes = base64.b64decode(req.image_base64)
            user_parts.append(
                types.Part.from_bytes(
                    data=img_bytes,
                    mime_type=req.image_mime or "image/jpeg",
                )
            )
        except Exception as e:
            print(f"[Image decode error] {e}")
    # Always include some text so the model has an instruction
    user_parts.append(types.Part(text=req.message or "הלקוח שלח תמונה. בדוק אותה ועזור בהתאם."))

    history.append(types.Content(role="user", parts=user_parts))

    # Gemini tool-use loop
    for _ in range(5):
        response = client.models.generate_content(
            model=MODEL,
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=TOOLS,
            )
        )

        candidate = response.candidates[0]
        content   = candidate.content

        # Add model turn to history
        history.append(content)

        # Check for tool calls
        tool_calls = [p for p in content.parts if p.function_call is not None]

        if not tool_calls:
            # No tool calls — return text response
            text = "".join(
                p.text for p in content.parts if p.text
            ).strip()
            sessions[req.session_id] = history
            return ChatResponse(reply=text, session_id=req.session_id)

        # Execute tools and collect results
        tool_response_parts = []
        for part in tool_calls:
            fc   = part.function_call
            name = fc.name
            args = dict(fc.args)
            print(f"[Tool call] {name}({args})")
            result = dispatch_tool(name, args)
            print(f"[Tool result] {result}")
            tool_response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=name,
                        response={"result": result}
                    )
                )
            )

        # Add tool results as user turn
        history.append(types.Content(role="user", parts=tool_response_parts))

    # Fallback
    sessions[req.session_id] = history
    return ChatResponse(
        reply="מצטערים, נתקלנו בבעיה טכנית. נשמח לעזור לך בדרך אחרת.",
        session_id=req.session_id
    )


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"cleared": True}