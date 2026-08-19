"""
Clinical Guideline Text Retrieval & Grounded Generation Engine (CAD / CVD)
Streamlit app — combines the Day 2 retrieval pipeline (extraction, chunking,
FAISS/BM25/Hybrid/Reranked search) with the Day 3 grounded generation layer
(strict system prompt, citations, refusal, confidence).

Run locally with:  streamlit run app.py
"""

import re
import numpy as np
import pandas as pd
import pdfplumber
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from google import genai

st.set_page_config(page_title="Clinical Guideline RAG", layout="wide")

LLM_MODEL = "gemini-3.6-flash"


# ============================================================
# Cached model loaders (loaded once per session, not per rerun)
# ============================================================
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading reranker model...")
def load_rerank_model():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ============================================================
# Extraction & cleaning (with garbled-text fallback fix)
# ============================================================
def _garbled_ratio(text: str) -> float:
    """Fraction of tokens that are lone alphabetic characters -- high values
    mean pdfplumber split words into individual letters (common on dense
    reference sections / small-font columns)."""
    words = text.split()
    if not words:
        return 1.0
    single_char = sum(1 for w in words if len(w) == 1 and w.isalpha())
    return single_char / len(words)


def extract_pages(file) -> list[dict]:
    """Tries a few pdfplumber extraction configs per page and keeps the
    least-garbled result. `file` is a Streamlit UploadedFile (file-like)."""
    extraction_configs = [
        {"layout": True, "x_tolerance": 2, "y_tolerance": 3},
        {"layout": True, "x_tolerance": 3, "y_tolerance": 3},
        {"layout": False, "x_tolerance": 2, "y_tolerance": 3},
    ]
    pages = []
    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages):
            candidates = []
            for kwargs in extraction_configs:
                try:
                    t = page.extract_text(**kwargs) or ""
                except Exception:
                    t = ""
                candidates.append(t)
            best_text = min(candidates, key=_garbled_ratio)
            pages.append({"page": i + 1, "text": best_text})
    return pages


def fix_concatenated_words(text: str) -> str:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
    return text


def clean_text(text: str) -> str:
    text = re.sub(r"\(cid:\d+\)", "", text)
    text = re.sub(r"[\u200b\uf0b7\ue000-\uf8ff]", "", text)
    text = re.sub(r"\uf02d", "-", text)
    text = re.sub(r"\uf02c", ",", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # strip non-ASCII font corruption
    text = re.sub(r"([a-zA-Z])- ([a-zA-Z])", r"\1\2", text)  # reconnect hyphenated breaks
    text = fix_concatenated_words(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


GARBLED_PAGE_THRESHOLD = 0.30  # drop pages still >30% single-letter tokens after cleaning


def process_pdf(file) -> list[dict]:
    raw_pages = extract_pages(file)
    cleaned_pages = []
    for p in raw_pages:
        cleaned = clean_text(p["text"])
        if not cleaned:
            continue
        if _garbled_ratio(cleaned) > GARBLED_PAGE_THRESHOLD:
            continue
        cleaned_pages.append({"page": p["page"], "text": cleaned})
    return cleaned_pages


# ============================================================
# Section-aware adaptive chunking
# ============================================================
SECTION_HEADERS = [
    "abstract", "introduction", "importance", "objective", "evidence review",
    "findings", "conclusions and recommendation", "rationale", "clinical considerations",
    "methods", "materials and methods", "results", "discussion", "recommendation",
    "recommendations", "summary", "background", "assessment", "screening",
    "treatment", "harms", "benefits", "references", "limitations",
]

RECOMMENDATION_SECTIONS = {
    "recommendation", "recommendations", "conclusions and recommendation",
    "summary", "clinical considerations",
}


def is_table_heavy(text: str, digit_threshold: float = 0.08) -> bool:
    if not text:
        return False
    digit_count = sum(c.isdigit() for c in text)
    return (digit_count / len(text)) > digit_threshold


def chunk_document_by_section(
    pages: list[dict],
    default_chunk_size: int = 300,
    default_overlap: int = 30,
    rec_chunk_size: int = 450,
    rec_overlap_pct: float = 0.125,
    table_chunk_size: int = 800,
    table_overlap_pct: float = 0.125,
) -> list[dict]:
    full_text = ""
    char_to_page = []
    for page in pages:
        full_text += page["text"] + "\n"
        char_to_page.extend([page["page"]] * (len(page["text"]) + 1))

    pattern = r"(?im)(?:^|\n|\.\s+)\b(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")(?=\b|:)"
    matches = list(re.finditer(pattern, full_text))

    breakpoints = [(0, "preamble")]
    for m in matches:
        breakpoints.append((m.start(), m.group(1).lower()))

    raw_sections = []
    for i, (start, sec_name) in enumerate(breakpoints):
        end = breakpoints[i + 1][0] if i + 1 < len(breakpoints) else len(full_text)
        sec_text = full_text[start:end].strip()
        if sec_text:
            page_num = char_to_page[min(start, len(char_to_page) - 1)]
            raw_sections.append({"section": sec_name, "page": page_num, "text": sec_text})

    merged_sections = []
    for sec in raw_sections:
        if merged_sections and merged_sections[-1]["section"] == sec["section"]:
            merged_sections[-1]["text"] += " " + sec["text"]
        else:
            merged_sections.append(dict(sec))

    chunks = []
    for sec in merged_sections:
        if sec["section"] == "references":
            continue
        if is_table_heavy(sec["text"]):
            chunk_size, overlap, exp_tag = table_chunk_size, int(table_chunk_size * table_overlap_pct), "table_heavy_800w"
        elif sec["section"] in RECOMMENDATION_SECTIONS:
            chunk_size, overlap, exp_tag = rec_chunk_size, int(rec_chunk_size * rec_overlap_pct), "recommendation_450w"
        else:
            chunk_size, overlap, exp_tag = default_chunk_size, default_overlap, "default_300w"

        words = sec["text"].split()
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            if chunk_text.strip():
                chunks.append({
                    "section": sec["section"], "page": sec["page"], "text": chunk_text,
                    "chunk_size_used": chunk_size, "overlap_used": overlap, "experiment": exp_tag,
                })
            start += max(1, chunk_size - overlap)

    return chunks


# ============================================================
# Build the chunk DataFrame + FAISS + BM25 index
# ============================================================
def build_index(all_pages: dict, embed_model) -> tuple[pd.DataFrame, faiss.Index, BM25Okapi]:
    records = []
    for doc_name, pages in all_pages.items():
        chunks = chunk_document_by_section(pages)
        for i, chunk in enumerate(chunks):
            records.append({
                "chunk_id": f"{doc_name[:6]}_{i}",
                "document": doc_name,
                "section": chunk["section"],
                "page": chunk["page"],
                "chunk_index": i,
                "experiment": chunk["experiment"],
                "text": chunk["text"],
            })

    df = pd.DataFrame(records)
    texts = df["text"].tolist()

    embeddings = embed_model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    tokenized_corpus = [doc.lower().split() for doc in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    return df, index, bm25


# ============================================================
# Search functions (df / index / bm25 / models passed explicitly --
# Streamlit reruns the script, so nothing should rely on module-level state)
# ============================================================
def semantic_search(query, df, index, embed_model, top_k=5):
    """Score range ~[-1, 1] (cosine similarity)."""
    query_vec = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vec, top_k)
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        row = df.iloc[idx]
        results.append({
            "rank": rank + 1, "score": float(score), "chunk_id": row["chunk_id"],
            "document": row["document"], "section": row["section"], "page": row["page"], "text": row["text"],
        })
    return results


def keyword_search(query, df, bm25, top_k=5):
    """Score range: unbounded raw BM25 score."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_indices):
        row = df.iloc[idx]
        results.append({
            "rank": rank + 1, "score": float(scores[idx]), "chunk_id": row["chunk_id"],
            "document": row["document"], "section": row["section"], "page": row["page"], "text": row["text"],
        })
    return results


def hybrid_search(query, df, index, bm25, embed_model, top_k=5, semantic_weight=0.6, keyword_weight=0.4):
    """Score range: [0, 1] (normalized fusion)."""
    query_vec = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    sem_scores, sem_indices = index.search(query_vec, len(df))

    semantic_array = np.zeros(len(df))
    for score, idx in zip(sem_scores[0], sem_indices[0]):
        semantic_array[idx] = score

    keyword_array = np.array(bm25.get_scores(query.lower().split()))

    def min_max_norm(arr):
        denom = arr.max() - arr.min()
        return np.zeros_like(arr) if denom == 0 else (arr - arr.min()) / denom

    combined = (semantic_weight * min_max_norm(semantic_array)) + (keyword_weight * min_max_norm(keyword_array))
    top_indices = np.argsort(combined)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices):
        row = df.iloc[idx]
        results.append({
            "rank": rank + 1, "score": float(combined[idx]), "chunk_id": row["chunk_id"],
            "document": row["document"], "section": row["section"], "page": row["page"], "text": row["text"],
        })
    return results


def reranked_search(query, df, index, embed_model, rerank_model, top_k=5, initial_k=15):
    """Score range: unbounded cross-encoder logit, typically roughly [-10, 10]."""
    candidates = semantic_search(query, df, index, embed_model, top_k=max(initial_k, top_k))
    pairs = [[query, c["text"]] for c in candidates]
    rerank_scores = rerank_model.predict(pairs)
    for c, score in zip(candidates, rerank_scores):
        c["score"] = float(score)
    reranked = sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]
    for rank, r in enumerate(reranked):
        r["rank"] = rank + 1
    return reranked


SEARCH_METHODS = {
    "Reranked (Cross-Encoder)": "reranked",
    "Hybrid (Semantic + BM25)": "hybrid",
    "Semantic (FAISS Cosine)": "semantic",
    "Keyword (BM25 Okapi)": "keyword",
}


def run_search(method_key, query, df, index, bm25, embed_model, rerank_model, top_k=5):
    if method_key == "reranked":
        return reranked_search(query, df, index, embed_model, rerank_model, top_k=top_k)
    if method_key == "hybrid":
        return hybrid_search(query, df, index, bm25, embed_model, top_k=top_k)
    if method_key == "semantic":
        return semantic_search(query, df, index, embed_model, top_k=top_k)
    return keyword_search(query, df, bm25, top_k=top_k)


# ============================================================
# Grounded generation layer (Day 3)
# ============================================================
SYSTEM_PROMPT = """You are an evidence-grounded clinical decision support assistant.

RULES (do not break these):
1. Use ONLY the retrieved guideline context provided below. Never use outside
   medical knowledge, training data, or general clinical intuition.
2. If the retrieved context does not clearly support an answer, you MUST say
   the evidence is insufficient. Do not guess or fill gaps.
3. Never provide patient-specific diagnosis, dosage, or treatment decisions.
   You support clinicians; you do not replace clinical judgment.
4. Every recommendation or claim must be traceable to a specific retrieved
   chunk (document, section, page).
5. Do not invent thresholds, statistics, or citations that are not present
   in the provided context.

You must respond using exactly this structure:

Recommendation: <short, direct answer based only on retrieved chunks, or
  "Insufficient evidence to answer this question." if context doesn't support it>

Supporting Evidence:
- <bullet point mapped to a specific retrieved chunk, short excerpt allowed>

Citations:
- <Document name>, Section: <section>, Page: <page> (chunk_id: <id>)

Confidence: <High | Medium | Low | Insufficient Evidence>
Disclaimer: This is guideline-based information support, not a substitute
  for clinical judgment. Verify against the original source before acting.
"""

REFUSAL_SCORE_THRESHOLD = -7.0  # calibrated to this corpus's actual cross-encoder score range
MIN_CHUNKS_REQUIRED = 1


def should_refuse(retrieved_chunks: list[dict]) -> tuple[bool, str]:
    if len(retrieved_chunks) < MIN_CHUNKS_REQUIRED:
        return True, "No relevant chunks were retrieved."
    top_score = retrieved_chunks[0]["score"]
    if top_score < REFUSAL_SCORE_THRESHOLD:
        return True, f"Retrieval confidence too low (top score={top_score:.3f})."
    return False, ""


def format_context_for_prompt(retrieved_chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(retrieved_chunks, start=1):
        blocks.append(
            f"[Chunk {i}] Document: {c['document']} | Section: {c['section']} | "
            f"Page: {c['page']} | chunk_id: {c['chunk_id']} | score: {c['score']:.3f}\n"
            f"Text: {c['text'][:800]}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context = format_context_for_prompt(retrieved_chunks)
    return f"""Question: {question}

Retrieved guideline context:
{context}

Answer strictly using the structure defined in the system prompt. Only cite
chunks listed above, using their exact document name, section, page, and
chunk_id."""


def estimate_confidence(retrieved_chunks: list[dict]) -> str:
    if not retrieved_chunks:
        return "Insufficient Evidence"
    top_score = retrieved_chunks[0]["score"]
    if top_score >= -2.0:
        return "High"
    elif top_score >= -5.0:
        return "Medium"
    elif top_score >= REFUSAL_SCORE_THRESHOLD:
        return "Low"
    else:
        return "Insufficient Evidence"


def call_llm(client, system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> str:
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=user_prompt,
        config={
            "system_instruction": system_prompt,
            "max_output_tokens": max_tokens,
        },
    )
    return response.text


def grounded_answer(question, df, index, bm25, embed_model, rerank_model, client, top_k=5) -> dict:
    retrieved = reranked_search(question, df, index, embed_model, rerank_model, top_k=top_k)

    refuse, reason = should_refuse(retrieved)
    if refuse:
        return {
            "question": question, "refused": True, "reason": reason,
            "answer": ("The retrieved guidelines do not provide sufficient evidence "
                       "to answer this question reliably. Please consult the relevant "
                       "clinical guideline or a qualified clinician."),
            "confidence": "Insufficient Evidence", "retrieved_chunks": retrieved,
        }

    user_prompt = build_user_prompt(question, retrieved)
    llm_output = call_llm(client, SYSTEM_PROMPT, user_prompt)

    return {
        "question": question, "refused": False, "reason": None,
        "answer": llm_output, "confidence": estimate_confidence(retrieved),
        "retrieved_chunks": retrieved,
    }


# ============================================================
# Streamlit UI
# ============================================================
def main():
    st.title("🫀 Clinical Guideline RAG — Grounded Q&A")
    st.caption("Upload CAD/CVD guideline PDFs, then ask questions with evidence-grounded, cited answers.")

    if "processed" not in st.session_state:
        st.session_state["processed"] = False

    with st.sidebar:
        st.header("1. Gemini API key")
        # On Streamlit Community Cloud, set GEMINI_API_KEY in the app's Secrets
        # (Settings -> Secrets) so it's never exposed in the deployed UI or repo.
        secret_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
        if secret_key:
            api_key = secret_key
            st.success("API key loaded from app secrets.")
        else:
            api_key = st.text_input(
                "AIza...", type="password",
                help="aistudio.google.com/apikey -> Create API key"
            )

        st.header("2. Upload guideline PDFs")
        uploaded_files = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)

        process_btn = st.button("Process documents", type="primary", disabled=not uploaded_files)

        if process_btn:
            embed_model = load_embed_model()
            all_pages = {}
            progress = st.progress(0.0, text="Extracting & cleaning text...")
            for i, f in enumerate(uploaded_files):
                all_pages[f.name] = process_pdf(f)
                progress.progress((i + 1) / len(uploaded_files), text=f"Processed {f.name}")

            with st.spinner("Chunking, embedding, indexing..."):
                df, index, bm25 = build_index(all_pages, embed_model)

            st.session_state["df"] = df
            st.session_state["index"] = index
            st.session_state["bm25"] = bm25
            st.session_state["processed"] = True
            st.success(f"Indexed {len(df)} chunks from {len(uploaded_files)} document(s).")

        if st.session_state["processed"]:
            st.divider()
            st.metric("Chunks indexed", len(st.session_state["df"]))
            st.metric("Documents", st.session_state["df"]["document"].nunique())

    if not st.session_state["processed"]:
        st.info("Upload PDFs and click **Process documents** in the sidebar to get started.")
        return

    df = st.session_state["df"]
    index = st.session_state["index"]
    bm25 = st.session_state["bm25"]
    embed_model = load_embed_model()

    tab_grounded, tab_raw = st.tabs(["Grounded Answer (LLM + citations)", "Raw retrieval (debug)"])

    # ---------------- Grounded answer tab ----------------
    with tab_grounded:
        question = st.text_input("Ask a clinical guideline question", key="grounded_q")
        top_k = st.slider("Chunks to retrieve", 3, 10, 5, key="grounded_k")
        ask_btn = st.button("Ask", type="primary", key="grounded_ask")

        if ask_btn and question:
            if not api_key:
                st.error("Enter your Anthropic API key in the sidebar first.")
            else:
                rerank_model = load_rerank_model()
                client = genai.Client(api_key=api_key)
                with st.spinner("Retrieving evidence and generating grounded answer..."):
                    result = grounded_answer(question, df, index, bm25, embed_model, rerank_model, client, top_k=top_k)

                if result["refused"]:
                    st.warning(f"**Refused to answer.** {result['reason']}")
                st.markdown(result["answer"])
                st.caption(f"Confidence signal: **{result['confidence']}**")

                with st.expander("View retrieved evidence (verify citations against this)"):
                    for c in result["retrieved_chunks"]:
                        st.markdown(f"**{c['document']}** | {c['section']} | p.{c['page']} | `{c['chunk_id']}` | score={c['score']:.3f}")
                        st.text(c["text"][:400] + "...")
                        st.divider()

    # ---------------- Raw retrieval tab ----------------
    with tab_raw:
        raw_question = st.text_input("Query", key="raw_q")
        method_label = st.selectbox("Search method", list(SEARCH_METHODS.keys()), key="raw_method")
        raw_top_k = st.slider("Top K", 3, 10, 5, key="raw_k")
        search_btn = st.button("Search", key="raw_search")

        if search_btn and raw_question:
            rerank_model = load_rerank_model() if SEARCH_METHODS[method_label] == "reranked" else None
            results = run_search(
                SEARCH_METHODS[method_label], raw_question, df, index, bm25,
                embed_model, rerank_model, top_k=raw_top_k,
            )
            for r in results:
                st.markdown(f"**#{r['rank']}** {r['document']} | {r['section']} | p.{r['page']} | score={r['score']:.4f}")
                st.text(r["text"][:400] + "...")
                st.divider()


if __name__ == "__main__":
    main()
