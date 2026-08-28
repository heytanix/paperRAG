#!/usr/bin/env python3
"""Answer questions over the paper_agents Chroma collection using Gemini."""

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

import chromadb
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

OUT_DIR = Path(__file__).parent
CHROMA_DIR = OUT_DIR / "chroma_db"
COLLECTION_NAME = "paper_agents"
MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.6-flash"

load_dotenv(OUT_DIR / ".env")

_embedding_model = None
_chroma_collection = None
_gemini_client = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(MODEL_NAME)
    return _embedding_model


def get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _chroma_collection = client.get_collection(name=COLLECTION_NAME)
    return _chroma_collection


def get_gemini_api_key():
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return api_key
    try:
        import streamlit as st

        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        api_key = get_gemini_api_key()
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set (expected in .env, environment, or st.secrets)"
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def build_context_block(chunks_with_meta):
    lines = []
    for i, (text, meta) in enumerate(chunks_with_meta, start=1):
        lines.append(
            f"[Chunk {i} | Paper: {meta['paper_title']} | Page: {meta['page']} | arXiv: {meta['arxiv_id']}]\n"
            f"{text}"
        )
    return "\n\n".join(lines)


def build_prompt(question, context_block):
    return f"""You are a research assistant answering questions using only the provided excerpts from academic papers.

Context excerpts:
{context_block}

Question: {question}

Instructions:
- Answer using ONLY the information in the context excerpts above. Do not use outside knowledge.
- For each part of your answer, explicitly cite which paper(s) it came from (use the paper title, and page number if relevant).
- If the context does not contain enough information to answer the question, say so clearly instead of guessing.
"""


def answer_question(question: str, top_k: int = 5):
    embedding_model = get_embedding_model()
    collection = get_collection()

    query_embedding = embedding_model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    chunks_with_meta = list(zip(documents, metadatas))
    context_block = build_context_block(chunks_with_meta)
    prompt = build_prompt(question, context_block)

    client = get_gemini_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    answer_text = response.text

    sources = [
        {
            "paper_title": meta["paper_title"],
            "page": meta["page"],
            "arxiv_id": meta["arxiv_id"],
        }
        for meta in metadatas
    ]

    return answer_text, sources


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 query_engine.py \"<question>\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    answer, sources = answer_question(question)

    print("=== Answer ===")
    print(answer)
    print("\n=== Sources ===")
    for i, src in enumerate(sources, start=1):
        print(f"{i}. {src['paper_title']} (page {src['page']}, arXiv:{src['arxiv_id']})")
