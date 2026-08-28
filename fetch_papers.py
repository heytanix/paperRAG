#!/usr/bin/env python3
"""Fetch LLM agent / tool-use papers from arXiv: PDFs + metadata JSON."""

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
REQUEST_DELAY = 3.5  # seconds between arXiv API/PDF requests (rate-limit friendly)

SEARCH_QUERIES = [
    'abs:"LLM agent" OR abs:"LLM agents"',
    'abs:"tool use" AND abs:"language model"',
    'abs:"function calling" AND abs:"language model"',
    'abs:"ReAct" AND abs:"language model"',
    'abs:"autonomous agent" AND abs:"language model"',
    'abs:"tool-augmented" AND abs:"language model"',
    'abs:"agentic" AND abs:"large language model"',
]

# Require the title or abstract to actually mention agent/tool-use concepts,
# since arXiv's relevance ranking still lets unrelated papers slip through.
RELEVANCE_KEYWORDS = [
    "agent",
    "tool use",
    "tool-use",
    "tool calling",
    "function calling",
    "react",
    "toolformer",
    "tool-augmented",
    "agentic",
]

TARGET_MIN = 30
TARGET_MAX = 50
RESULTS_PER_QUERY = 20

OUT_DIR = Path(__file__).parent
PAPERS_DIR = OUT_DIR / "papers"
METADATA_PATH = OUT_DIR / "papers_metadata.json"


def sanitize_filename(title: str, max_len: int = 120) -> str:
    name = re.sub(r"\s+", " ", title).strip()
    name = re.sub(r'[\\/:*?"<>|]', "", name)
    name = name.replace(" ", "_")
    return name[:max_len]


def fetch_query(query: str, max_results: int):
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "paperRAG-fetch-script/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    root = ET.fromstring(data)
    entries = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        arxiv_id_full = entry.find(f"{ATOM_NS}id").text.strip()
        arxiv_id = arxiv_id_full.rsplit("/abs/", 1)[-1]
        title = entry.find(f"{ATOM_NS}title").text.strip()
        abstract = entry.find(f"{ATOM_NS}summary").text.strip()
        authors = [
            a.find(f"{ATOM_NS}name").text.strip()
            for a in entry.findall(f"{ATOM_NS}author")
        ]
        pdf_url = None
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href")
                break
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        entries.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "pdf_url": pdf_url,
            }
        )
    return entries


def download_pdf(pdf_url: str, dest: Path):
    req = urllib.request.Request(pdf_url, headers={"User-Agent": "paperRAG-fetch-script/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())


def main():
    PAPERS_DIR.mkdir(exist_ok=True)

    seen_ids = set()
    collected = []

    for query in SEARCH_QUERIES:
        if len(collected) >= TARGET_MAX:
            break
        print(f"Searching arXiv for: {query!r}")
        try:
            entries = fetch_query(query, RESULTS_PER_QUERY)
        except Exception as e:
            print(f"  query failed: {e}")
            continue
        time.sleep(REQUEST_DELAY)

        for entry in entries:
            base_id = entry["arxiv_id"].split("v")[0]
            if base_id in seen_ids:
                continue
            haystack = (entry["title"] + " " + entry["abstract"]).lower()
            if not any(kw in haystack for kw in RELEVANCE_KEYWORDS):
                continue
            seen_ids.add(base_id)
            collected.append(entry)
            if len(collected) >= TARGET_MAX:
                break

    print(f"Collected {len(collected)} unique candidate papers. Downloading PDFs...")

    metadata = []
    used_filenames = set()

    for i, entry in enumerate(collected, 1):
        filename = sanitize_filename(entry["title"]) + ".pdf"
        stem, ext = filename[:-4], ".pdf"
        n = 1
        while filename in used_filenames:
            n += 1
            filename = f"{stem}_{n}{ext}"
        used_filenames.add(filename)
        dest_path = PAPERS_DIR / filename

        print(f"[{i}/{len(collected)}] {entry['title'][:70]!r} -> {filename}")
        try:
            download_pdf(entry["pdf_url"], dest_path)
        except Exception as e:
            print(f"  download failed: {e}")
            used_filenames.discard(filename)
            continue

        metadata.append(
            {
                "title": entry["title"],
                "authors": entry["authors"],
                "arxiv_id": entry["arxiv_id"],
                "abstract": entry["abstract"],
                "filename": filename,
            }
        )

        time.sleep(REQUEST_DELAY)

        if len(metadata) >= TARGET_MAX:
            break

    if len(metadata) < TARGET_MIN:
        print(
            f"Warning: only downloaded {len(metadata)} papers, "
            f"target minimum was {TARGET_MIN}."
        )

    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved metadata for {len(metadata)} papers to {METADATA_PATH}")


if __name__ == "__main__":
    main()
