#!/usr/bin/env python3
"""Streamlit UI for paper-rag: ask questions over the indexed arXiv papers."""

import streamlit as st

from query_engine import MODEL_NAME, answer_question

TOTAL_PAPERS = 50
TOTAL_CHUNKS = 718

st.set_page_config(page_title="Paper RAG", page_icon="📄")

st.title("Paper RAG")
st.write(
    "Ask questions about LLM agents and tool use, answered using only the "
    "indexed arXiv papers, with citations back to source pages."
)

with st.sidebar:
    st.header("Corpus stats")
    st.metric("Total papers", TOTAL_PAPERS)
    st.metric("Total chunks", TOTAL_CHUNKS)
    st.write(f"**Embedding model:** {MODEL_NAME}")

question = st.text_input("Your question", placeholder="e.g. How do LLM agents decide when to call a tool?")
ask_clicked = st.button("Ask")

if ask_clicked and question.strip():
    with st.spinner("Searching papers and generating answer..."):
        answer, sources = answer_question(question)

    st.markdown(answer)

    with st.expander("Sources"):
        for i, src in enumerate(sources, start=1):
            arxiv_id = src["arxiv_id"]
            abs_url = f"https://arxiv.org/abs/{arxiv_id}"
            st.markdown(
                f"{i}. **{src['paper_title']}** — page {src['page']} "
                f"([arXiv:{arxiv_id}]({abs_url}))"
            )
elif ask_clicked:
    st.warning("Please enter a question.")
