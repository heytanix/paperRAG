#!/usr/bin/env python3
"""Embed chunks.jsonl with sentence-transformers and store in a persistent Chroma collection."""

import json
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

OUT_DIR = Path(__file__).parent
CHUNKS_PATH = OUT_DIR / "chunks.jsonl"
CHROMA_DIR = OUT_DIR / "chroma_db"
COLLECTION_NAME = "paper_agents"
MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZE = 100


def load_chunks():
    chunks = []
    with CHUNKS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main():
    start_time = time.time()

    print(f"Loading chunks from {CHUNKS_PATH}...")
    chunks = load_chunks()
    total_chunks = len(chunks)
    print(f"Loaded {total_chunks} chunks.")

    print(f"Loading embedding model {MODEL_NAME!r}...")
    model = SentenceTransformer(MODEL_NAME)
    embedding_dim = model.get_embedding_dimension()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    indexed = 0
    for batch_start in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]

        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [
            {
                "paper_title": c["paper_title"],
                "arxiv_id": c["arxiv_id"],
                "filename": c.get("filename", ""),
                "page": c["page"],
                "chunk_id": c["chunk_id"],
            }
            for c in batch
        ]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        indexed += len(batch)
        if indexed % 100 == 0 or indexed == total_chunks:
            print(f"Embedded {indexed}/{total_chunks} chunks...")

    elapsed = time.time() - start_time

    print("\n--- Summary ---")
    print(f"Total chunks indexed: {indexed}")
    print(f"Embedding dimension:  {embedding_dim}")
    print(f"Time taken:           {elapsed:.1f}s")
    print(f"Chroma collection:    {COLLECTION_NAME!r} persisted at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
