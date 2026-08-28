#!/usr/bin/env python3
"""Extract text from papers/*.pdf and split into overlapping chunks -> chunks.jsonl."""

import json
import re
from pathlib import Path

import pdfplumber

OUT_DIR = Path(__file__).parent
PAPERS_DIR = OUT_DIR / "papers"
METADATA_PATH = OUT_DIR / "papers_metadata.json"
CHUNKS_PATH = OUT_DIR / "chunks.jsonl"

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

# No tokenizer dependency: approximate a "token" as a whitespace-separated word.
TOKEN_RE = re.compile(r"\S+")


def load_metadata_by_filename():
    entries = json.loads(METADATA_PATH.read_text())
    return {e["filename"]: e for e in entries}


def extract_pages(pdf_path: Path):
    """Return list of (page_number, text) for pages with extractable text."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append((i, text))
    return pages


def tokenize_with_offsets(text: str):
    """Return list of (word, start_char, end_char) for the text."""
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def chunk_paper(pages, chunk_size, overlap):
    """
    Flatten all pages into one token stream (tracking each token's source page),
    then slide a window of chunk_size tokens with the given overlap.
    Returns list of (chunk_text, page_number) where page_number is the page
    of the chunk's first token (approximate).
    """
    all_tokens = []  # (word, page_number)
    for page_num, text in pages:
        for word, _, _ in tokenize_with_offsets(text):
            all_tokens.append((word, page_num))

    if not all_tokens:
        return []

    chunks = []
    step = chunk_size - overlap
    i = 0
    n = len(all_tokens)
    while i < n:
        window = all_tokens[i : i + chunk_size]
        if not window:
            break
        chunk_text = " ".join(w for w, _ in window)
        page_num = window[0][1]
        chunks.append((chunk_text, page_num))
        if i + chunk_size >= n:
            break
        i += step
    return chunks


def main():
    metadata_by_filename = load_metadata_by_filename()
    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

    total_chunks = 0
    processed_papers = 0
    failed_papers = []
    chunk_id_counter = 0

    with CHUNKS_PATH.open("w") as out_f:
        for pdf_path in pdf_files:
            meta = metadata_by_filename.get(pdf_path.name)
            if meta is None:
                title = pdf_path.stem
                arxiv_id = "unknown"
            else:
                title = meta["title"]
                arxiv_id = meta["arxiv_id"]

            try:
                pages = extract_pages(pdf_path)
            except Exception as e:
                failed_papers.append((pdf_path.name, f"extraction error: {e}"))
                continue

            paper_chunks = chunk_paper(pages, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)

            if not paper_chunks:
                failed_papers.append((pdf_path.name, "no extractable text (likely scanned/image-based)"))
                continue

            for chunk_text, page_num in paper_chunks:
                record = {
                    "text": chunk_text,
                    "paper_title": title,
                    "arxiv_id": arxiv_id,
                    "filename": pdf_path.name,
                    "page": page_num,
                    "chunk_id": f"{arxiv_id}_{chunk_id_counter}",
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunk_id_counter += 1
                total_chunks += 1

            processed_papers += 1
            print(f"[{processed_papers + len(failed_papers)}/{len(pdf_files)}] "
                  f"{title[:60]!r}: {len(paper_chunks)} chunks")

    print("\n--- Summary ---")
    print(f"Total papers found:      {len(pdf_files)}")
    print(f"Papers processed OK:     {processed_papers}")
    print(f"Total chunks created:    {total_chunks}")
    if processed_papers:
        print(f"Average chunks/paper:    {total_chunks / processed_papers:.1f}")
    print(f"Papers failed:           {len(failed_papers)}")
    if failed_papers:
        print("\nFailed papers:")
        for name, reason in failed_papers:
            print(f"  - {name}: {reason}")
    print(f"\nSaved chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
