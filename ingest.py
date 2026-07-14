"""
ingest.py
---------
This is the "build the index" step of RAG.

What it does, in data-pipeline terms:
1. Reads your portfolio content (portfolio_content.json) - think of this
   as your source table.
2. Converts each chunk of text into a vector (a list of numbers that captures
   its meaning) using a local, free embedding model (fastembed / BAAI's
   bge-small-en-v1.5, running via ONNX Runtime on CPU) - think of this as a
   transformation step, like a DAX measure but for meaning instead of numbers.
   This runs entirely on your machine (or on Render) - no API key, no cost,
   no rate limit. Groq itself does not offer an embeddings endpoint, so this
   step no longer talks to Groq at all; only chat/generation (in main.py)
   uses Groq.
3. Stores those vectors in a FAISS index (a small, fast local "table" built
   for similarity search) plus a metadata file mapping each vector back to
   its original text.

Run this ONCE locally whenever your portfolio content changes:
    python ingest.py

The first run downloads the small embedding model (~130 MB) from
Hugging Face and caches it locally; subsequent runs are instant.

It produces two files in data/:
    - portfolio.index     (the FAISS vector index)
    - portfolio_meta.json (the original text, aligned to each vector)

NOTE: Because we switched from Gemini's embedding model to bge-small-en-v1.5,
the vectors are a different size/shape than before. You must re-run this
script once after migrating so data/portfolio.index is rebuilt - old index
files created by the Gemini version will NOT work with the new main.py.
"""

import json
import os

import faiss
import numpy as np
from fastembed import TextEmbedding

ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data")
CONTENT_PATH = os.path.join(ROOT_DIR, "portfolio_content.json")
if not os.path.exists(CONTENT_PATH):
    CONTENT_PATH = os.path.join(DATA_DIR, "portfolio_content.json")
INDEX_PATH = os.path.join(DATA_DIR, "portfolio.index")
META_PATH = os.path.join(DATA_DIR, "portfolio_meta.json")

# Free, local, no API key needed. 384-dim, small (~130MB), good quality.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

_embedder = None


def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBED_MODEL)
    return _embedder


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(CONTENT_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [f"{c['title']}. {c['text']}" for c in chunks]

    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL} (local, free)...")
    embedder = get_embedder()
    # bge models are trained with a "passage:" style prefix for documents.
    prefixed = [f"passage: {t}" for t in texts]
    vectors = np.array(list(embedder.embed(prefixed)), dtype="float32")

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
