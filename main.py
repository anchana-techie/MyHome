"""
main.py
-------
The backend "brain" behind your portfolio chat widget - now running on
Gemini for both generation and embeddings (one API key for everything).

Two ideas combined here:

1. RAG (Retrieval-Augmented Generation)
   The `search_portfolio` function embeds the user's question, searches the
   FAISS index built by ingest.py, and returns the most relevant chunks of
   your real content. This is what stops the AI from inventing facts about
   your work.

2. Agentic AI
   Instead of always running search -> answer, Gemini is given a small set
   of TOOLS (search_portfolio, get_contact_info) via function calling and
   decides on its own, per question, which tool(s) to call - the way a
   dispatcher decides which stored procedure to run based on input. We loop
   until Gemini stops requesting function calls and gives a final answer.

Run locally:
    uvicorn main:app --reload

Deploy: see README.md for Render deployment steps.
"""

import json
import os

import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv(override=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INDEX_PATH = os.path.join(DATA_DIR, "portfolio.index")
META_PATH = os.path.join(DATA_DIR, "portfolio_meta.json")

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"  # fast + cheap, has a free tier; upgrade to
                                  # gemini-3 models later if you want
FALLBACK_MESSAGE = "Time for bed. See you tomorrow."


def get_gemini_api_keys() -> list[str]:
    load_dotenv(override=True)
    keys = []
    for env_name in ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"):
        value = os.environ.get(env_name, "").strip()
        if value:
            keys.append(value)
    return keys


def call_with_key_rotation(operation, *args, **kwargs):
    api_keys = get_gemini_api_keys()
    if not api_keys:
        raise RuntimeError("No Gemini API keys configured. Add GEMINI_API_KEY, GEMINI_API_KEY_2, and GEMINI_API_KEY_3 to your .env file.")

    last_error = None
    for index, key in enumerate(api_keys, start=1):
        try:
            client = genai.Client(api_key=key)
            return operation(client, *args, **kwargs)
        except Exception as exc:
            last_error = exc
            message = str(exc)
            print(f"Gemini request failed with key {index}/{len(api_keys)}: {message}")
            if "429" in message or "RESOURCE_EXHAUSTED" in message or "quota" in message.lower():
                continue
            break
    raise RuntimeError(f"All Gemini API keys failed. Last error: {last_error}") from last_error

def load_portfolio_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return faiss.read_index(INDEX_PATH), meta

    try:
        from ingest import main as build_index

        build_index()
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return faiss.read_index(INDEX_PATH), meta
    except Exception as exc:
        print(f"Warning: could not build portfolio index: {exc}")
        return None, []


# ---- Load index once at startup (not per-request) ----
faiss_index, portfolio_meta = load_portfolio_data()

app = FastAPI(title="Anchana Portfolio AI")

# ---- CORS: allow your GitHub Pages site to call this API ----
ALLOWED_ORIGINS = [
    "https://anchana-techie.github.io",
    "http://localhost:5500",  # for local testing with e.g. VS Code Live Server
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "null",  # browsers send this literal Origin header for file:// pages (double-clicked HTML)
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# RAG retrieval function
# ---------------------------------------------------------------------------
def search_portfolio(query: str, top_k: int = 3) -> str:
    if not portfolio_meta:
        return "Portfolio content is not available yet."

    if faiss_index is None:
        query_lower = query.lower()
        hits = []
        for chunk in portfolio_meta:
            haystack = f"{chunk['title']} {chunk['text']}".lower()
            if query_lower in haystack:
                hits.append(f"[{chunk['title']}] {chunk['text']}")
        return "\n\n".join(hits[:top_k]) if hits else "No relevant information found."

    try:
        result = call_with_key_rotation(
            lambda client: client.models.embed_content(
                model=EMBED_MODEL,
                contents=query,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )
        )
    except Exception as exc:
        print(f"Embedding failed: {exc}")
        return FALLBACK_MESSAGE

    vec = np.array([result.embeddings[0].values], dtype="float32")
    faiss.normalize_L2(vec)

    scores, idxs = faiss_index.search(vec, top_k)
    hits = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        chunk = portfolio_meta[idx]
        hits.append(f"[{chunk['title']}] {chunk['text']}")

    return "\n\n".join(hits) if hits else "No relevant information found."


def get_contact_info() -> str:
    return "Email: anchana.professional@gmail.com | LinkedIn: linkedin.com/in/anchana-prabakaran-231331233"


# ---------------------------------------------------------------------------
# Agent tool definitions - Gemini decides when to call these
# ---------------------------------------------------------------------------
search_portfolio_decl = types.FunctionDeclaration(
    name="search_portfolio",
    description=(
        "Search Anchana's portfolio content (experience, projects, skills, "
        "certifications, education) for information relevant to the user's "
        "question. Use this for any question about her background or work."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for"}
        },
        "required": ["query"],
    },
)

get_contact_info_decl = types.FunctionDeclaration(
    name="get_contact_info",
    description="Return Anchana's email and LinkedIn profile link.",
    parameters={"type": "object", "properties": {}},
)

TOOLS = [types.Tool(function_declarations=[search_portfolio_decl, get_contact_info_decl])]

SYSTEM_PROMPT = (
    "You are the AI assistant embedded in Anchana Prabakaran's portfolio "
    "website. You answer visitor questions about her experience, skills, "
    "projects, certifications, Beyond Work, CSR - and only that. Always use the "
    "search_portfolio tool before answering questions about her background; "
    "never invent details that aren't returned by the tool. If a question is "
    "unrelated to Anchana's professional profile, politely redirect the "
    "visitor back to portfolio-related topics. Keep answers concise and "
    "friendly, 2-4 sentences unless more detail is clearly requested."
)


def run_tool(name: str, tool_input: dict) -> str:
    if name == "search_portfolio":
        return search_portfolio(tool_input["query"])
    if name == "get_contact_info":
        return get_contact_info()
    return "Unknown tool"


def to_gemini_contents(messages: list) -> list:
    """Convert our simple {role, content} history into Gemini's Content format."""
    contents = []
    for m in messages:
        role = "model" if m["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    return contents


# ---------------------------------------------------------------------------
# Agent loop: call Gemini, run any requested function calls, repeat until done
# ---------------------------------------------------------------------------
def run_agent(messages: list) -> str:
    contents = to_gemini_contents(messages)

    for _ in range(5):  # safety cap on tool-use round trips
        try:
            response = call_with_key_rotation(
                lambda client: client.models.generate_content(
                    model=CHAT_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        tools=TOOLS,
                        max_output_tokens=600,
                    ),
                )
            )
        except Exception as exc:
            print(f"Chat generation failed: {exc}")
            return FALLBACK_MESSAGE

        candidate = response.candidates[0]
        function_calls = [
            part.function_call for part in candidate.content.parts if part.function_call
        ]

        if not function_calls:
            # Final answer - collect text parts
            return "".join(
                part.text for part in candidate.content.parts if part.text
            )

        # Gemini wants to call one or more functions
        contents.append(candidate.content)
        function_response_parts = []
        for fc in function_calls:
            result = run_tool(fc.name, dict(fc.args))
            function_response_parts.append(
                types.Part.from_function_response(
                    name=fc.name, response={"result": result}
                )
            )
        contents.append(types.Content(role="user", parts=function_response_parts))

    return FALLBACK_MESSAGE


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # [{"role": "user"/"assistant", "content": "..."}]


@app.post("/chat")
def chat(req: ChatRequest):
    messages = req.history + [{"role": "user", "content": req.message}]
    reply = run_agent(messages)
    return {"reply": reply}


@app.get("/")
def health():
    return {"status": "ok"}