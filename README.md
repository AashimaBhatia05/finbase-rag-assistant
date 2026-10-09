# FinBase Support Assistant (RAG)

A customer support assistant for the fictional fintech company FinBase. It answers questions about fixed deposits, payments and UPI, credit cards, savings accounts, personal loans and KYC using **only** the six FinBase policy documents. Every answer shows its sources, and the assistant says so when the answer is not in the documents.

**Live app:** https://finbase-rag-assistant.streamlit.app/

## Features

- Retrieval-Augmented Generation: question → retrieval → context → LLM → grounded answer
- Source citations (document, section, FAQ id) on every answer
- Refuses to answer when the knowledge base has no answer
- Follow-up questions (the question is rewritten into a standalone query using chat history)
- Match-strength score per answer and a toggle to inspect retrieved passages
- Three retrieval modes (vector, BM25, hybrid) and a reproducible evaluation script

## Architecture

```
PDFs → extraction (PyMuPDF, tables kept intact) → cleaning (₹ fix, de-duplication)
     → section / sub-clause chunking (120 chunks with metadata)
     → embeddings (BGE-small, 384-d) → ChromaDB (cosine)
                                            │
User question → [rewrite if follow-up] → retriever (top 5) → relevance gate (0.55)
                                            │
                      grounded prompt + numbered passages → Gemini → JSON {found, answer, sources}
                                            │
                           answer + source citations + match strength (Streamlit UI)
```

## Tech stack and why

| Part | Choice | Why |
|---|---|---|
| Language / UI | Python, Streamlit | Fast to build, chat components built in, free hosting |
| PDF extraction | PyMuPDF | Fast, and its table detection keeps rows together |
| Embeddings | BAAI/bge-small-en-v1.5 (384-d) | Small, strong for English retrieval, runs on CPU. The app runs it through `fastembed` (ONNX) so it fits the free hosting memory limit (no PyTorch) |
| Vector DB | ChromaDB (cosine, in-memory) | Simple, no server. The index has only 120 chunks, so it is rebuilt at startup in seconds |
| Keyword search | rank-bm25 | Used for the hybrid mode and the retrieval comparison |
| LLM | Google Gemini Flash-Lite, temperature 0, with fallback models | Free tier, fast (about 3 s per answer), reliable JSON output |
| Hosting | Streamlit Community Cloud | Free, deploys from GitHub |

## Data and preprocessing

The 6 PDFs (about 25 pages each) are mostly repeated filler:

- Sections 4–20 of every document are one template repeated with only the numbers changed.
- The FAQ has 100 entries per document but only 10 unique questions.

What the pipeline does:

1. Extracts text page by page and renders tables as rows (`cell | cell | cell`).
2. Fixes extraction artefacts (the ₹ symbol came out as the letter "I"), strips markdown noise and the table of contents.
3. Removes a repeated section only if its amounts, percentages and durations are identical to an earlier one, so no real information is lost.
4. Collapses the duplicate FAQ entries into one chunk each and keeps the other ids as metadata.

## Chunking

Chunks follow the document structure instead of a fixed size:

- One chunk per section, split further by sub-clause (for example Section 1.2) so every citation points to a real clause.
- Tables are never split.
- Each FAQ entry is its own chunk.
- Every chunk starts with a header (document name, code, section) so the embedding knows where the text comes from.

Result: 120 chunks (60 section chunks, 60 FAQ chunks), 330–1,300 characters each, average about 630.

## Retrieval

- Default: vector search (cosine) over all chunks, top 5 passed to the LLM.
- Also implemented: BM25, and hybrid (reciprocal rank fusion of vector and BM25).
- Relevance gate: if the best similarity is below 0.55 the question is treated as unrelated and no LLM call is made.
- Set `RETRIEVAL_MODE=hybrid` or `bm25` to switch modes.

## Prompting and hallucination mitigation

- The LLM may use only the numbered passages and must return JSON: `found`, `answer`, `sources`.
- `found=false` produces a fixed "not found in the knowledge base" message.
- Rules cover: exact quoting of amounts, stating each value when sections disagree, preferring numbered sections over FAQ summaries, never mixing facts from different documents, and never showing passage numbers to the customer.
- Citations come from the passages the model reports using, validated against the retrieved set.
- Temperature 0.

## Evaluation

65 questions in `data/processed/eval_set.json`:

- 30 written by hand (documents 1–3, including 6 multi-source questions)
- 30 auto-generated from the FAQs of documents 4–6 (LLM paraphrase, expected source and key amounts taken from the FAQ)
- 5 questions that are not in the knowledge base

**Retrieval** (correct source in the top k, 60 answerable questions):

| Mode | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| **Vector (default)** | **0.917** | **1.000** | **1.000** | **0.953** |
| Hybrid | 0.883 | 0.950 | 0.983 | 0.919 |
| BM25 | 0.783 | 0.883 | 0.883 | 0.828 |

**End to end** (vector mode):

| Metric | Result |
|---|---|
| Answered (not wrongly refused) | 100% |
| Fact correctness (key amounts present) | 98% |
| Citation accuracy (a correct source cited) | 100% |
| Groundedness (LLM judge) | 98% |
| Correct refusals on not-in-KB questions | 100% (5 of 5) |
| Latency, mean / max | 3.0 s / 22 s |

Reproduce with `python evaluate.py` (add `--retrieval-only` to skip LLM calls).

**Honest notes**

- The test set is small and was built from the same documents, so treat the numbers as indicative.
- Three auto-generated test items had wrong expected facts and were corrected by hand. The retrieval miss on "How long do I have to pay back the loan?" under hybrid mode is left in the results.
- One question still fails (the Luxe annual-fee waiver): the handbook words the waiver period differently in Section 1.2 and in the FAQ.

## Key trade-offs

- Structure-aware chunks instead of fixed-size windows: better citations, needs per-document parsing.
- Vector over hybrid as default: it scored higher here, so the extra BM25 complexity did not pay off on this data.
- `fastembed` (ONNX) instead of PyTorch in the deployed app: slightly different runtime, much smaller memory footprint.
- The similarity gate only blocks clearly unrelated questions. Questions about the right topic whose answer is missing (for example reward-point redemption) look relevant to vector search, so the prompt has to refuse them.

## Known limitations

- Answers are only as good as the documents. The documents contain a few internal conflicts, which the assistant reports instead of resolving.
- Free LLM tiers change often. Fallback models are configured with `LLM_MODELS`.
- No user accounts or real customer data. It is a knowledge-base assistant, not a banking system.

## Project structure

```
app.py                      Streamlit chat UI
rag_core.py                 RagEngine: embeddings, vector store, retrieval, prompt, LLM calls
evaluate.py                 Retrieval and end-to-end evaluation
rag_pipeline.ipynb          Development notebook: extraction, cleaning, chunking, experiments
data/processed/chunks.json  The 120 chunks (generated by the notebook)
data/processed/eval_set.json  Evaluation questions
requirements.txt
```

## Setup

```bash
git clone https://github.com/AashimaBhatia05/finbase-rag-assistant.git
cd finbase-rag-assistant
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key
```

A free key is available at https://aistudio.google.com/apikey. Then run:

```bash
streamlit run app.py
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Google AI Studio key. On Streamlit Cloud, add it under Settings → Secrets as `GEMINI_API_KEY = "..."` |
| `LLM_MODELS` | No | Comma-separated Gemini models, tried in order. Default: `gemini-flash-lite-latest,gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-flash-latest` |
| `RETRIEVAL_MODE` | No | `vector` (default), `hybrid` or `bm25` |

## Rebuilding the chunks from the PDFs

The source PDFs are not in this repository. To rebuild `data/processed/chunks.json`, put the six PDFs in `data-raw/` and run Steps 1–3 of `rag_pipeline.ipynb` (extraction, cleaning, chunking).

## Python API

```python
from rag_core import RagEngine

engine = RagEngine()
res = engine.answer("What is the foreign currency markup on the Luxe card?")
res["answer"]    # grounded answer text
res["found"]     # False if the knowledge base has no answer
res["sources"]   # list of chunks used (each has "citation" and "text")
res["top_sim"]   # best similarity score of the retrieved passages
```

Pass `history=[{"q": ..., "a": ...}]` to handle follow-up questions.

## Deployment

The app is deployed on Streamlit Community Cloud from this repository (`app.py`), with `GEMINI_API_KEY` stored in the app secrets.

## What I would improve with more time

- A cross-encoder reranker on the top 20 candidates
- A larger, fully hand-written evaluation set with human-rated answers
- Routing questions to the right document (metadata filters) before retrieval
- Streaming answers, caching of repeated questions, unit tests and Docker
- An explicit precedence rule when two sections of a document disagree
