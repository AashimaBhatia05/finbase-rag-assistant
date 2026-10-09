import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag_core import RagEngine  # noqa: E402

st.set_page_config(page_title="FinBase Support Assistant", page_icon="💬", layout="centered")

EXAMPLES = [
    "What is the interest rate for senior citizens on a 1-year FD?",
    "My UPI payment failed but money was deducted. What now?",
    "What is the annual fee for the Luxe credit card?",
    "What is the foreclosure charge on a personal loan?",
    "Which documents are accepted for KYC?",
    "Can I buy Bitcoin on FinBase?",
]


@st.cache_resource(show_spinner="Loading the FinBase knowledge base...")
def load_engine():
    key = None
    try:
        key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        key = os.getenv("GEMINI_API_KEY")
    return RagEngine(api_key=key)


def strength(sim):
    return "High" if sim >= 0.75 else "Medium" if sim >= 0.65 else "Low"


engine = load_engine()
if engine.client is None:
    st.error("GEMINI_API_KEY is not configured. Add it to .env (local) or the app secrets (deployed).")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Try asking")
    for q in EXAMPLES:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q
    st.divider()
    show_context = st.checkbox("Show retrieved passages", value=False)
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("Answers come only from the FinBase policy documents: fixed deposits, payments, "
               "credit cards, savings, personal loans and KYC.")


def render(msg):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] != "assistant":
            return
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})", expanded=True):
                for s in msg["sources"]:
                    st.markdown(f"**{s['citation']}**")
                    st.caption(s["snippet"])
        if msg.get("meta"):
            st.caption(msg["meta"])
        if show_context and msg.get("retrieved"):
            with st.expander("Retrieved passages"):
                for r in msg["retrieved"]:
                    st.markdown(f"`{r['sim']:.2f}`  {r['citation']}")


st.title("FinBase Support Assistant")
st.caption("Ask a question about FinBase deposits, payments, cards, savings, loans or KYC.")

for m in st.session_state.messages:
    render(m)

prompt = st.chat_input("Type your question")
if not prompt and "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    msgs = st.session_state.messages
    history = [{"q": m["question"], "a": m["content"]} for m in msgs if m["role"] == "assistant"][-3:]
    user_msg = {"role": "user", "content": prompt}
    msgs.append(user_msg)
    render(user_msg)

    try:
        with st.spinner("Searching the FinBase documents..."):
            res = engine.answer(prompt, history)
    except Exception:
        st.error("Sorry, the answer service is busy right now. Please try again in a moment.")
        msgs.pop()
        st.stop()

    if res["found"]:
        meta = f"Match strength: {strength(res['top_sim'])} ({res['top_sim']:.2f}) · {res['seconds']:.1f}s"
    else:
        meta = "Not found in the FinBase knowledge base"

    assistant_msg = {
        "role": "assistant",
        "question": prompt,
        "content": res["answer"],
        "meta": meta,
        "sources": [{"citation": c["citation"], "snippet": c["text"].split("\n", 1)[-1][:400]}
                    for c in res["sources"]],
        "retrieved": [{"citation": h["chunk"]["citation"], "sim": h["vec_sim"]} for h in res["retrieved"]],
    }
    msgs.append(assistant_msg)
    render(assistant_msg)