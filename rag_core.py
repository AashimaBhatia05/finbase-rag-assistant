"""FinBase RAG core: hybrid retrieval over the FinBase documents + grounded answer generation."""
import json
import logging
import os
import re
import time
from pathlib import Path

import chromadb
import numpy as np
from fastembed import TextEmbedding
from google import genai
from google.genai import types
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).parent
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("finbase_rag")

CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
LLM_MODELS = [m.strip() for m in os.getenv(
    "LLM_MODELS",
    "gemini-flash-lite-latest,gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-flash-latest",
).split(",")]
MIN_SIM = 0.55   # below this the question is treated as unrelated to FinBase
TOP_K = 5
DEFAULT_MODE = os.getenv("RETRIEVAL_MODE", "vector")   # chosen from the evaluation: vector > hybrid > bm25

def _find_chunks():
    """Locate chunks.json even if the folder layout differs (paths are case-sensitive on Linux)."""
    if CHUNKS_PATH.exists():
        return CHUNKS_PATH
    found = sorted(BASE_DIR.rglob("chunks.json"))
    if found:
        return found[0]
    top = sorted(x.name for x in BASE_DIR.iterdir())
    raise FileNotFoundError(f"chunks.json not found under {BASE_DIR}. Top-level files: {top}. "
                            "Push data/processed/chunks.json to the repository.")


NOT_FOUND_MSG = ("I couldn't find this in the FinBase knowledge base. "
                 "Please contact FinBase customer support for help with this.")

PROMPT_TEMPLATE = """You are FinBase's customer support assistant. Answer ONLY from the numbered context passages below.

Rules:
1. Use only facts that appear in the context. Never use outside knowledge and never guess.
2. If the context does not contain the answer, set "found" to false.
3. If the context says FinBase does not offer something, say it is not offered and cite it (found = true).
4. If passages give different values for the same thing (for example for different transaction types), state each value and the section it comes from.
5. Quote amounts, percentages and timelines exactly as written, keeping the ₹ symbol.
6. Keep the answer short and in plain language (2-4 sentences). Give no personal financial advice.
7. In "sources", list the numbers of the passages you actually used.
8. Never write passage numbers such as [1] or "source 3" inside the answer text. When you say where something is stated, name the document and section from the passage label, for example "Payments SOP, Section 1".
9. Numbered sections are the main policy text and FAQ passages are summaries of them. If an FAQ passage differs from a numbered section, follow the numbered section and briefly note the difference.
10. Passages can come from different FinBase documents (fixed deposits, payments, credit cards, savings, loans, KYC). Never merge facts from different documents into one statement. If the question could apply to more than one document, answer for each one and name the document.

Return ONLY valid JSON, with no other text:
{{"found": true or false, "answer": "your answer", "sources": [1, 2]}}

Context:
{context}

Question: {question}
"""


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


def _parse_json(text):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class RagEngine:
    def __init__(self, api_key=None):
        self.chunks = json.loads(_find_chunks().read_text(encoding="utf-8"))
        self.by_id = {c["chunk_id"]: c for c in self.chunks}

        self.embedder = TextEmbedding(model_name=EMBED_MODEL)
        vectors = self._embed([c["text"] for c in self.chunks])

        chroma = chromadb.EphemeralClient()
        try:
            chroma.delete_collection("finbase")
        except Exception:
            pass
        self.collection = chroma.create_collection("finbase", metadata={"hnsw:space": "cosine"})
        self.collection.add(
            ids=[c["chunk_id"] for c in self.chunks],
            embeddings=vectors.tolist(),
            documents=[c["text"] for c in self.chunks],
            metadatas=[{
                "doc_stem": c["doc_stem"], "doc_code": c["doc_code"], "type": c["type"],
                "section_id": str(c.get("section_id") or ""), "sub_id": c.get("sub_id") or "",
                "page": int(c["page"]), "citation": c["citation"],
            } for c in self.chunks],
        )
        self.bm25 = BM25Okapi([_tokenize(c["text"]) for c in self.chunks])

        key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=key.strip().strip('"').strip("'")) if key else None
        log.info("Engine ready: %d chunks", len(self.chunks))

    # ---------- retrieval ----------
    def _embed(self, texts):
        vecs = np.array(list(self.embedder.embed(texts)), dtype="float32")
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def vector_search(self, query):
        q = self._embed([QUERY_PREFIX + query])
        res = self.collection.query(query_embeddings=q.tolist(), n_results=len(self.chunks))
        return [(cid, 1.0 - d) for cid, d in zip(res["ids"][0], res["distances"][0])]

    def bm25_search(self, query, k=20):
        scores = self.bm25.get_scores(_tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i]["chunk_id"], float(scores[i])) for i in top]

    def retrieve(self, query, k=TOP_K, mode="hybrid"):
        """mode: 'vector', 'bm25' or 'hybrid' (reciprocal rank fusion of both)."""
        vec = self.vector_search(query)
        sims = dict(vec)
        if mode == "vector":
            ranked = vec[:k]
        elif mode == "bm25":
            ranked = self.bm25_search(query, k)
        else:
            fused = {}
            for ranking in (vec[:20], self.bm25_search(query, 20)):
                for rank, (cid, _) in enumerate(ranking):
                    fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank + 1)
            ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        return [{"chunk": self.by_id[cid], "score": float(s), "vec_sim": float(sims[cid])} for cid, s in ranked]

    # ---------- generation ----------
    def _call_llm(self, prompt):
        if self.client is None:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        last_err = None
        for model_name in LLM_MODELS:
            for attempt in range(2):
                try:
                    r = self.client.models.generate_content(
                        model=model_name, contents=prompt,
                        config=types.GenerateContentConfig(temperature=0.0),
                    )
                    return (r.text or ""), model_name
                except Exception as e:
                    last_err = e
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"All LLM models failed: {str(last_err)[:200]}")

    def rewrite_query(self, question, history):
        """Turn a follow-up like 'and for Metal?' into a standalone question."""
        if not history:
            return question
        convo = "\n".join(f"User: {h['q']}\nAssistant: {h['a'][:300]}" for h in history[-3:])
        prompt = (
            "Rewrite the user's latest question as one standalone question that makes sense "
            "without the conversation. Keep all product names, amounts and terms. "
            "If it is already standalone, return it unchanged. Output only the question.\n\n"
            f"Conversation:\n{convo}\n\nLatest question: {question}\n\nStandalone question:"
        )
        text, _ = self._call_llm(prompt)
        return text.strip().strip('"') or question

    def answer(self, question, history=None, k=TOP_K, mode=DEFAULT_MODE):
        t0 = time.time()
        standalone = self.rewrite_query(question, history or [])
        hits = self.retrieve(standalone, k=k, mode=mode)
        top_sim = max(h["vec_sim"] for h in hits)

        result = {"question": question, "standalone": standalone, "top_sim": top_sim,
                  "retrieved": hits, "model": None, "found": False, "answer": NOT_FOUND_MSG,
                  "sources": [], "reason": "", "seconds": 0.0}

        if top_sim < MIN_SIM:
            result["reason"] = "low_similarity"
        else:
            blocks = [f"[{i}] ({h['chunk']['citation']})\n{h['chunk']['text']}" for i, h in enumerate(hits, 1)]
            prompt = PROMPT_TEMPLATE.format(context="\n\n".join(blocks), question=standalone)
            data, model_name = None, None
            for _ in range(2):
                text, model_name = self._call_llm(prompt)
                data = _parse_json(text)
                if data and "answer" in data:
                    break
            result["model"] = model_name

            if not data or "answer" not in data:
                result["reason"] = "bad_llm_output"
            elif not data.get("found", False):
                result["reason"] = "llm_not_found"
            else:
                idx = [i for i in data.get("sources", []) if isinstance(i, int) and 1 <= i <= len(hits)]
                idx = list(dict.fromkeys(idx))
                if not idx:
                    idx, result["reason"] = [1], "source_fallback"
                result.update({"found": True, "answer": str(data["answer"]).strip(),
                               "sources": [hits[i - 1]["chunk"] for i in idx]})

        result["seconds"] = time.time() - t0
        log.info("q=%r found=%s sim=%.2f reason=%s %.1fs", question, result["found"],
                 top_sim, result["reason"] or "ok", result["seconds"])
        return result