"""
ingest.py
---------
This is the "build the index" step of RAG.

What it does, in data-pipeline terms:
1. Reads your portfolio content (data/portfolio_content.json) - think of this
   as your source table.
2. Converts each chunk of text into a vector (a list of numbers that captures
   its meaning) using Gemini's embedding model - think of this as a
   transformation step, like a DAX measure but for meaning instead of numbers.
3. Stores those vectors in a FAISS index (a small, fast local "table" built
   for similarity search) plus a metadata file mapping each vector back to
   its original text.

Run this ONCE locally whenever your portfolio content changes:
    python ingest.py

It produces two files in data/:
    - portfolio.index     (the FAISS vector index)
    - portfolio_meta.json (the original text, aligned to each vector)
"""

import json
import os

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)

ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data")
CONTENT_PATH = os.path.join(ROOT_DIR, "portfolio_content.json")
if not os.path.exists(CONTENT_PATH):
    CONTENT_PATH = os.path.join(DATA_DIR, "portfolio_content.json")
INDEX_PATH = os.path.join(DATA_DIR, "portfolio.index")
META_PATH = os.path.join(DATA_DIR, "portfolio_meta.json")

EMBED_MODEL = "gemini-embedding-001"  # stable, text-only, has a free tier


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
        raise RuntimeError("No Gemini API keys set. Add GEMINI_API_KEY, GEMINI_API_KEY_2, and GEMINI_API_KEY_3 to your .env file.")

    last_error = None
    for index, key in enumerate(api_keys, start=1):
        try:
            client = genai.Client(api_key=key)
            return operation(client, *args, **kwargs)
        except Exception as exc:
            last_error = exc
            message = str(exc)
            print(f"Embedding failed with key {index}/{len(api_keys)}: {message}")
            if "429" in message or "RESOURCE_EXHAUSTED" in message or "quota" in message.lower():
                continue
            break
    raise RuntimeError(f"All Gemini API keys failed. Last error: {last_error}") from last_error


def main():
    if not get_gemini_api_keys():
        raise SystemExit(
            "No Gemini API keys set. Add GEMINI_API_KEY, GEMINI_API_KEY_2, and GEMINI_API_KEY_3 to your .env file."
        )

    os.makedirs(DATA_DIR, exist_ok=True)

    with open(CONTENT_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [f"{c['title']}. {c['text']}" for c in chunks]

    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL}...")
    vectors = []
    for text in texts:
        # Gemini's embed_content takes one input per call in the stable API
        result = call_with_key_rotation(
            lambda client: client.models.embed_content(
                model=EMBED_MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
        )
        vectors.append(result.embeddings[0].values)

    vectors = np.array(vectors, dtype="float32")

    # Normalize vectors so we can use cosine similarity via inner product
    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # IP = inner product = cosine sim (post-normalize)
    index.add(vectors)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Done. Wrote {INDEX_PATH} and {META_PATH}")


if __name__ == "__main__":
    main()