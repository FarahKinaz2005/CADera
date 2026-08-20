"""
Clinical Guideline Text Retrieval & Grounded Generation Engine (CAD / CVD)

Streamlit app:
- PDF extraction and cleaning
- Section-aware chunking
- FAISS semantic retrieval
- BM25 keyword retrieval
- Cross-encoder reranking
- Grounded Gemini generation
- Evidence sufficiency check
- Refusal for unsupported / out-of-scope questions
- Confidence estimation based on retrieval quality

Run:
    streamlit run app.py

PDFs:
    Put all guideline PDFs inside ./data
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pdfplumber
import faiss
import streamlit as st

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from google import genai


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Clinical Guideline RAG",
    layout="wide"
)

LLM_MODEL = "gemini-3.5-flash-lite"


# ============================================================
# MODEL LOADERS
# ============================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading reranker model...")
def load_rerank_model():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ============================================================
# PDF EXTRACTION
# ============================================================

def _garbled_ratio(text: str) -> float:
    """
    Detect pages where PDF extraction produced many
    single-letter tokens.
    """
    words = text.split()

    if not words:
        return 1.0

    single_char = sum(
        1 for w in words
        if len(w) == 1 and w.isalpha()
    )

    return single_char / len(words)


def extract_pages(file_path) -> list[dict]:
    """
    Try multiple pdfplumber extraction configurations
    and keep the least-garbled result for each page.
    """

    extraction_configs = [
        {
            "layout": True,
            "x_tolerance": 2,
            "y_tolerance": 3
        },
        {
            "layout": True,
            "x_tolerance": 3,
            "y_tolerance": 3
        },
        {
            "layout": False,
            "x_tolerance": 2,
            "y_tolerance": 3
        }
    ]

    pages = []

    with pdfplumber.open(file_path) as pdf:

        for i, page in enumerate(pdf.pages):

            candidates = []

            for kwargs in extraction_configs:

                try:
                    text = page.extract_text(**kwargs) or ""
                except Exception:
                    text = ""

                candidates.append(text)

            best_text = min(
                candidates,
                key=_garbled_ratio
            )

            pages.append(
                {
                    "page": i + 1,
                    "text": best_text
                }
            )

    return pages


def fix_concatenated_words(text: str) -> str:

    text = re.sub(
        r"([a-z])([A-Z])",
        r"\1 \2",
        text
    )

    text = re.sub(
        r"([A-Za-z])(\d)",
        r"\1 \2",
        text
    )

    text = re.sub(
        r"(\d)([A-Za-z])",
        r"\1 \2",
        text
    )

    return text


def clean_text(text: str) -> str:

    text = re.sub(
        r"\(cid:\d+\)",
        "",
        text
    )

    text = re.sub(
        r"[\u200b\uf0b7\ue000-\uf8ff]",
        "",
        text
    )

    text = re.sub(
        r"\uf02d",
        "-",
        text
    )

    text = re.sub(
        r"\uf02c",
        ",",
        text
    )

    # Remove non-ASCII font corruption
    text = re.sub(
        r"[^\x00-\x7F]+",
        " ",
        text
    )

    # Reconnect hyphenated line breaks
    text = re.sub(
        r"([a-zA-Z])- ([a-zA-Z])",
        r"\1\2",
        text
    )

    text = fix_concatenated_words(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


GARBLED_PAGE_THRESHOLD = 0.30


def process_pdf(file_path) -> list[dict]:

    raw_pages = extract_pages(file_path)

    cleaned_pages = []

    for page in raw_pages:

        cleaned = clean_text(page["text"])

        if not cleaned:
            continue

        if _garbled_ratio(cleaned) > GARBLED_PAGE_THRESHOLD:
            continue

        cleaned_pages.append(
            {
                "page": page["page"],
                "text": cleaned
            }
        )

    return cleaned_pages


# ============================================================
# SECTION-AWARE CHUNKING
# ============================================================

SECTION_HEADERS = [
    "abstract",
    "introduction",
    "importance",
    "objective",
    "evidence review",
    "findings",
    "conclusions and recommendation",
    "rationale",
    "clinical considerations",
    "methods",
    "materials and methods",
    "results",
    "discussion",
    "recommendation",
    "recommendations",
    "summary",
    "background",
    "assessment",
    "screening",
    "treatment",
    "harms",
    "benefits",
    "references",
    "limitations",
]


RECOMMENDATION_SECTIONS = {
    "recommendation",
    "recommendations",
    "conclusions and recommendation",
    "summary",
    "clinical considerations",
}


def is_table_heavy(
    text: str,
    digit_threshold: float = 0.08
) -> bool:

    if not text:
        return False

    digit_count = sum(
        c.isdigit()
        for c in text
    )

    return (
        digit_count / len(text)
    ) > digit_threshold


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

        char_to_page.extend(
            [page["page"]]
            * (len(page["text"]) + 1)
        )

    pattern = (
        r"(?im)(?:^|\n|\.\s+)\b("
        + "|".join(
            re.escape(h)
            for h in SECTION_HEADERS
        )
        + r")(?=\b|:)"
    )

    matches = list(
        re.finditer(
            pattern,
            full_text
        )
    )

    breakpoints = [
        (0, "preamble")
    ]

    for match in matches:

        breakpoints.append(
            (
                match.start(),
                match.group(1).lower()
            )
        )

    raw_sections = []

    for i, (start, section_name) in enumerate(
        breakpoints
    ):

        end = (
            breakpoints[i + 1][0]
            if i + 1 < len(breakpoints)
            else len(full_text)
        )

        section_text = (
            full_text[start:end]
            .strip()
        )

        if section_text:

            page_num = char_to_page[
                min(
                    start,
                    len(char_to_page) - 1
                )
            ]

            raw_sections.append(
                {
                    "section": section_name,
                    "page": page_num,
                    "text": section_text
                }
            )

    # Merge repeated sections
    merged_sections = []

    for section in raw_sections:

        if (
            merged_sections
            and merged_sections[-1]["section"]
            == section["section"]
        ):

            merged_sections[-1]["text"] += (
                " " + section["text"]
            )

        else:

            merged_sections.append(
                dict(section)
            )

    chunks = []

    for section in merged_sections:

        # References are not useful for answering questions
        if section["section"] == "references":
            continue

        if is_table_heavy(section["text"]):

            chunk_size = table_chunk_size
            overlap = int(
                table_chunk_size
                * table_overlap_pct
            )

            experiment = "table_heavy_800w"

        elif section["section"] in RECOMMENDATION_SECTIONS:

            chunk_size = rec_chunk_size
            overlap = int(
                rec_chunk_size
                * rec_overlap_pct
            )

            experiment = "recommendation_450w"

        else:

            chunk_size = default_chunk_size
            overlap = default_overlap
            experiment = "default_300w"

        words = section["text"].split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk_text = " ".join(
                words[start:end]
            )

            if chunk_text.strip():

                chunks.append(
                    {
                        "section": section["section"],
                        "page": section["page"],
                        "text": chunk_text,
                        "chunk_size_used": chunk_size,
                        "overlap_used": overlap,
                        "experiment": experiment,
                    }
                )

            start += max(
                1,
                chunk_size - overlap
            )

    return chunks


# ============================================================
# BUILD INDEX
# ============================================================

def build_index(
    all_pages: dict,
    embed_model
):

    records = []

    for doc_name, pages in all_pages.items():

        chunks = chunk_document_by_section(
            pages
        )

        for i, chunk in enumerate(chunks):

            records.append(
                {
                    "chunk_id": f"{doc_name[:6]}_{i}",
                    "document": doc_name,
                    "section": chunk["section"],
                    "page": chunk["page"],
                    "chunk_index": i,
                    "experiment": chunk["experiment"],
                    "text": chunk["text"],
                }
            )

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(
            "No usable text chunks were extracted from the PDFs."
        )

    texts = df["text"].tolist()

    embeddings = embed_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    tokenized_corpus = [
        text.lower().split()
        for text in texts
    ]

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    return df, index, bm25


# ============================================================
# HYBRID RETRIEVAL + RERANKING
# ============================================================

def semantic_search(
    query,
    df,
    index,
    embed_model,
    top_k=20
):
    """Retrieve candidates using FAISS cosine similarity."""

    if len(df) == 0:
        return []

    top_k = min(top_k, len(df))

    query_vec = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_vec, top_k)

    results = []

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0:
            continue

        row = df.iloc[idx]
        results.append({
            "semantic_rank": rank + 1,
            "semantic_score": float(score),
            "chunk_id": row["chunk_id"],
            "document": row["document"],
            "section": row["section"],
            "page": row["page"],
            "text": row["text"],
        })

    return results


def bm25_search(query, df, bm25, top_k=20):
    """Retrieve candidates using BM25 keyword matching."""

    if len(df) == 0:
        return []

    tokens = query.lower().split()
    scores = np.asarray(bm25.get_scores(tokens), dtype="float32")
    top_k = min(top_k, len(scores))

    indices = np.argsort(scores)[::-1][:top_k]
    results = []

    for rank, idx in enumerate(indices):
        row = df.iloc[int(idx)]
        results.append({
            "bm25_rank": rank + 1,
            "bm25_score": float(scores[idx]),
            "chunk_id": row["chunk_id"],
            "document": row["document"],
            "section": row["section"],
            "page": row["page"],
            "text": row["text"],
        })

    return results


def hybrid_search(
    query,
    df,
    index,
    bm25,
    embed_model,
    initial_k=30
):
    """Combine semantic and keyword retrieval before reranking."""

    semantic = semantic_search(
        query, df, index, embed_model, top_k=initial_k
    )
    lexical = bm25_search(
        query, df, bm25, top_k=initial_k
    )

    merged = {}

    for item in semantic:
        merged[item["chunk_id"]] = dict(item)
        merged[item["chunk_id"]]["semantic_rank"] = item["semantic_rank"]

    for item in lexical:
        cid = item["chunk_id"]
        if cid not in merged:
            merged[cid] = dict(item)
        else:
            merged[cid]["bm25_rank"] = item["bm25_rank"]
            merged[cid]["bm25_score"] = item["bm25_score"]

    # Reciprocal Rank Fusion is used only to order the candidate pool.
    # The cross-encoder remains the final relevance ranker.
    for item in merged.values():
        semantic_rank = item.get("semantic_rank")
        bm25_rank = item.get("bm25_rank")
        rrf = 0.0

        if semantic_rank is not None:
            rrf += 1.0 / (60.0 + semantic_rank)
        if bm25_rank is not None:
            rrf += 1.0 / (60.0 + bm25_rank)

        item["hybrid_rrf"] = rrf
        item.setdefault("semantic_score", 0.0)
        item.setdefault("bm25_score", 0.0)

    return sorted(
        merged.values(),
        key=lambda x: x["hybrid_rrf"],
        reverse=True
    )[:initial_k]


def reranked_search(
    query,
    df,
    index,
    bm25,
    embed_model,
    rerank_model,
    top_k=5,
    initial_k=30
):
    """Hybrid FAISS + BM25 retrieval followed by CrossEncoder reranking."""

    candidates = hybrid_search(
        query,
        df,
        index,
        bm25,
        embed_model,
        initial_k=max(initial_k, top_k)
    )

    if not candidates:
        return []

    pairs = [[query, c["text"]] for c in candidates]
    rerank_scores = rerank_model.predict(pairs)

    for candidate, score in zip(candidates, rerank_scores):
        candidate["rerank_score"] = float(score)

    candidates.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    top_score = candidates[0]["rerank_score"]
    second_score = (
        candidates[1]["rerank_score"]
        if len(candidates) > 1
        else top_score
    )
    margin = max(0.0, top_score - second_score)

    # IMPORTANT: CrossEncoder scores are logits, NOT probabilities.
    # Never display the raw score as confidence.
    rerank_quality = 1.0 / (
        1.0 + np.exp(-np.clip((top_score + 4.0) / 1.5, -20, 20))
    )

    top_semantic = candidates[0].get("semantic_score", 0.0)
    semantic_quality = np.clip(
        (top_semantic - 0.20) / 0.45,
        0.0,
        1.0
    )

    score_range = max(
        c["rerank_score"] for c in candidates
    ) - min(
        c["rerank_score"] for c in candidates
    )
    margin_quality = np.clip(
        margin / score_range if score_range > 1e-9 else 0.0,
        0.0,
        1.0
    )

    # Hybrid agreement: reward evidence retrieved by BOTH FAISS and BM25.
    hybrid_agreement = 1.0 if (
        candidates[0].get("semantic_rank") is not None
        and candidates[0].get("bm25_rank") is not None
    ) else 0.0

    retrieval_quality = (
        0.45 * rerank_quality
        + 0.25 * semantic_quality
        + 0.15 * margin_quality
        + 0.15 * hybrid_agreement
    )

    for candidate in candidates:
        candidate["confidence_score"] = 0.0

    candidates[0]["confidence_score"] = float(
        np.clip(retrieval_quality, 0.0, 1.0)
    )

    reranked = candidates[:top_k]

    for rank, result in enumerate(reranked, start=1):
        result["rank"] = rank

    return reranked

# ============================================================
# CONFIDENCE / REFUSAL
# ============================================================

# Confidence is a bounded evidence-quality signal, not the raw
# CrossEncoder score. The verifier is the final answerability gate.
REFUSAL_THRESHOLD = 0.30
LOW_THRESHOLD = 0.45
MEDIUM_THRESHOLD = 0.65
HIGH_THRESHOLD = 0.80


def estimate_confidence(retrieved_chunks: list[dict]) -> str:
    if not retrieved_chunks:
        return "Insufficient Evidence"

    quality = float(
        np.clip(
            retrieved_chunks[0].get("confidence_score", 0.0),
            0.0,
            1.0
        )
    )

    if quality >= HIGH_THRESHOLD:
        return "High"
    if quality >= MEDIUM_THRESHOLD:
        return "Medium"
    if quality >= LOW_THRESHOLD:
        return "Low"
    return "Insufficient Evidence"


def should_refuse(retrieved_chunks: list[dict]):
    """Only reject an empty/very weak retrieval pool.

    Exact evidence sufficiency is checked by verify_answerability(),
    so retrieval quality alone cannot incorrectly reject a valid answer.
    """
    if not retrieved_chunks:
        return True, "No relevant evidence was retrieved."

    quality = float(
        np.clip(
            retrieved_chunks[0].get("confidence_score", 0.0),
            0.0,
            1.0
        )
    )

    if quality < REFUSAL_THRESHOLD:
        return True, (
            "The retrieved guidelines do not contain sufficiently "
            "relevant evidence for this question."
        )

    return False, ""

# ============================================================
# GROUNDED GENERATION PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an evidence-grounded clinical guideline assistant.

Your ONLY source of information is the retrieved guideline context
provided in the user message.

STRICT RULES:

1. NEVER use outside medical knowledge.
2. NEVER use your training knowledge to fill missing information.
3. NEVER guess.
4. Every factual claim must be directly supported by the retrieved context.
5. If the retrieved context only mentions a concept but does not actually
   answer the question, the evidence is insufficient.
6. Do not treat a keyword mention as evidence that answers the question.
7. If the question asks for information that is not supported by the
   retrieved context, refuse to answer.
8. Do not provide patient-specific diagnosis, dosage, or treatment decisions.
9. Do not invent numbers, thresholds, recommendations, citations, or facts.
10. Do not cite a chunk unless that chunk actually supports the claim.

IMPORTANT:

A chunk that merely contains related words is NOT sufficient evidence.

For example, if the question asks:
"What are the traditional cardiovascular risk factors?"

and the retrieved text only says:
"traditional and non-traditional cardiovascular risk factors were assessed"

that does NOT answer the question.

In that case, the correct result is:
INSUFFICIENT.

Likewise, if the user asks something completely outside the clinical
guidelines, such as "What is food?", the correct result is:
INSUFFICIENT.

You must return ONLY one of these two formats.

FORMAT A — if the retrieved evidence clearly answers the question:

Answer:
<direct answer>

Supporting Evidence:
- <specific evidence from the retrieved context>

Citations:
- <document>, Section: <section>, Page: <page> (chunk_id: <id>)
- Do not include retrieval scores or any other ranking metadata in citations.

Confidence:
<High | Medium | Low>

FORMAT B — if the retrieved evidence does NOT clearly answer the question:

INSUFFICIENT

Do NOT write a long explanation.
Do NOT provide citations.
Do NOT provide confidence.
"""


# ============================================================
# LLM ANSWER
# ============================================================

def call_llm(client, system_prompt, user_prompt):
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=user_prompt,
            config={"system_instruction": system_prompt}
        )
        # Verify that the response contains text before reading it.
        if response.text:
            return response.text
        else:
            return "Insufficient evidence in the provided documents."
    except Exception:
        return "Insufficient evidence in the provided documents."


# ============================================================
# ANSWERABILITY VERIFICATION
# ============================================================

def build_verification_prompt(
    question,
    retrieved_chunks
):

    context_blocks = []

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        context_blocks.append(
            f"""
[Chunk {i}]
Document: {chunk['document']}
Section: {chunk['section']}
Page: {chunk['page']}
Chunk ID: {chunk['chunk_id']}

Text:
{chunk['text'][:1200]}
"""
        )

    context = "\n".join(
        context_blocks
    )

    return f"""
Question:
{question}

Retrieved guideline evidence:
{context}

Determine whether the retrieved evidence CLEARLY answers the question.

Important:
- Related terminology is NOT enough.
- A chunk merely mentioning the topic is NOT enough.
- The evidence must actually contain the information needed to answer
  the question.
- Do not use outside knowledge.
- If the answer is not explicitly or clearly supported, return INSUFFICIENT.

Return ONLY:

ANSWERABLE

or

INSUFFICIENT
"""


def verify_answerability(
    client,
    question,
    retrieved_chunks
):

    prompt = build_verification_prompt(
        question,
        retrieved_chunks
    )

    result = call_llm(
        client,
        """
You are an evidence sufficiency classifier.

Use ONLY the supplied retrieved guideline chunks.

Return ANSWERABLE only when the chunks clearly contain enough information
to answer the user's exact question.

A related topic, keyword match, or mention is NOT enough.

Otherwise return INSUFFICIENT.

Return exactly one word:
ANSWERABLE
or
INSUFFICIENT
""",
        prompt,
        max_tokens=20
    )

    result = result.upper().strip()

    return result == "ANSWERABLE"


# ============================================================
# MAIN GROUNDED ANSWER PIPELINE
# ============================================================

def grounded_answer(
    question,
    df,
    index,
    bm25,
    embed_model,
    rerank_model,
    client,
    top_k=5
):

    # --------------------------------------------------------
    # Step 1: Retrieve
    # --------------------------------------------------------

    retrieved = reranked_search(
        question,
        df,
        index,
        bm25,
        embed_model,
        rerank_model,
        top_k=top_k
    )

    # --------------------------------------------------------
    # Step 3: Evidence sufficiency check
    #
    # This is the important fix for:
    #
    # "traditional cardiovascular risk factors"
    #
    # where retrieval found a related chunk but it didn't
    # actually contain the answer.
    # --------------------------------------------------------

    answerable = verify_answerability(
        client,
        question,
        retrieved
    )

    if not answerable:

        return {
            "question": question,
            "refused": True,
            "reason": (
                "The retrieved guideline evidence does not "
                "clearly answer this question."
            ),
            "answer": "Refused to answer.",
            "confidence": None,
            "retrieved_chunks": retrieved,
        }

    # --------------------------------------------------------
    # Step 4: Estimate confidence AFTER evidence verification
    # --------------------------------------------------------
    refuse, reason = should_refuse(retrieved)

    if refuse:
        return {
            "question": question,
            "refused": True,
            "reason": reason,
            "answer": "Refused to answer.",
            "confidence": None,
            "retrieved_chunks": retrieved,
        }

    # --------------------------------------------------------
    # Step 5: Estimate confidence
    # --------------------------------------------------------

    confidence = estimate_confidence(
        retrieved
    )

    # --------------------------------------------------------
    # Step 5: Build grounded answer prompt
    # --------------------------------------------------------

    context_blocks = []

    for i, chunk in enumerate(
        retrieved,
        start=1
    ):

        context_blocks.append(
            f"""
[Chunk {i}]
Document: {chunk['document']}
Section: {chunk['section']}
Page: {chunk['page']}
Chunk ID: {chunk['chunk_id']}

Text:
{chunk['text'][:1000]}
"""
        )

    context = "\n".join(
        context_blocks
    )

    user_prompt = f"""
Question:
{question}

Retrieved guideline context:
{context}

The retrieval system has already determined that the evidence is sufficient
to answer the question.

Use ONLY the retrieved context.

The calculated confidence is:

{confidence}

Use exactly this confidence value.

Do not invent any information.
Do not use outside knowledge.
"""

    # --------------------------------------------------------
    # Step 6: Generate
    # --------------------------------------------------------

    llm_output = call_llm(
        client,
        SYSTEM_PROMPT,
        user_prompt
    )

    # --------------------------------------------------------
    # Step 7: Safety consistency check
    #
    # If the model somehow says insufficient, do NOT show
    # High/Medium/Low beside it.
    # --------------------------------------------------------

    if (
        "insufficient" in llm_output.lower()
        or llm_output.strip().upper() == "INSUFFICIENT"
    ):

        return {
            "question": question,
            "refused": True,
            "reason": (
                "The generated answer could not be grounded "
                "sufficiently in the retrieved evidence."
            ),
            "answer": "Refused to answer.",
            "confidence": None,
            "retrieved_chunks": retrieved,
        }

    return {
        "question": question,
        "refused": False,
        "reason": None,
        "answer": llm_output,
        "confidence": confidence,
        "retrieved_chunks": retrieved,
    }


# ============================================================
# LOAD DOCUMENTS
# ============================================================

PDF_DIR = Path(__file__).parent / "data"


@st.cache_resource(
    show_spinner="Loading clinical guideline documents..."
)
def load_documents_from_repo():

    pdf_paths = sorted(
        PDF_DIR.glob("*.pdf")
    )

    if not pdf_paths:

        raise FileNotFoundError(
            f"No PDF files found in '{PDF_DIR}'. "
            "Add your guideline PDFs to the data folder."
        )

    embed_model = load_embed_model()

    all_pages = {}

    for pdf_path in pdf_paths:

        all_pages[pdf_path.name] = process_pdf(
            pdf_path
        )

    df, index, bm25 = build_index(
        all_pages,
        embed_model
    )

    return (
        df,
        index,
        bm25,
        [p.name for p in pdf_paths]
    )


# ============================================================
# STREAMLIT UI
# ============================================================

def main():

    st.title(
        "🫀 Clinical Guideline RAG — Grounded Q&A"
    )

    st.caption(
        "Ask clinical guideline questions and receive "
        "evidence-grounded answers with citations."
    )

    # --------------------------------------------------------
    # API KEY
    # --------------------------------------------------------

    secret_key = (
        st.secrets.get(
            "GEMINI_API_KEY",
            ""
        )
        if hasattr(st, "secrets")
        else ""
    )

    if not secret_key:

        st.error(
            "Gemini API key is not configured in Streamlit Secrets."
        )

        return

    # --------------------------------------------------------
    # LOAD DATABASE
    # --------------------------------------------------------

    try:

        df, index, bm25, document_names = (
            load_documents_from_repo()
        )

    except FileNotFoundError as e:

        st.error(str(e))

        return

    except Exception as e:

        st.error(
            f"Failed to load the knowledge base: {e}"
        )

        return

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    col1.metric(
        "Chunks indexed",
        len(df)
    )

    col2.metric(
        "Documents",
        len(document_names)
    )

    with st.expander(
        "Loaded documents"
    ):

        for name in document_names:
            st.text(name)

    st.divider()

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    embed_model = load_embed_model()

    # --------------------------------------------------------
    # QUESTION
    # --------------------------------------------------------

    question = st.text_input(
        "Ask a clinical guideline question",
        key="grounded_q"
    )

    top_k = st.slider(
        "Chunks to retrieve",
        3,
        10,
        5,
        key="grounded_k"
    )

    ask_btn = st.button(
        "Ask",
        type="primary",
        key="grounded_ask"
    )

    # --------------------------------------------------------
    # ASK
    # --------------------------------------------------------

    if ask_btn and question.strip():

        rerank_model = load_rerank_model()

        client = genai.Client(
            api_key=secret_key
        )

        with st.spinner(
            "Retrieving evidence and generating grounded answer..."
        ):

            result = grounded_answer(
                question,
                df,
                index,
                bm25,
                embed_model,
                rerank_model,
                client,
                top_k=top_k
            )

        # ----------------------------------------------------
        # REFUSED
        # ----------------------------------------------------

        if result["refused"]:

            st.markdown(
                "### Refused to answer."
            )

        # ----------------------------------------------------
        # VALID ANSWER
        # ----------------------------------------------------

        else:

            st.markdown(
                result["answer"]
            )

            st.caption(
                f"Confidence signal: "
                f"**{result['confidence']}**"
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
