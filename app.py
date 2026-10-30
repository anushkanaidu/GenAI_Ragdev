"""Buddy: HR onboarding Q&A over 10 policy documents (Streamlit UI)."""
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# Streamlit Cloud: read the key from Secrets if it is not in the environment.
try:
    if "GROQ_API_KEY" in st.secrets and not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

from rag import build_vectorstore, pure_llm_answer, rag_answer  # noqa: E402

st.set_page_config(page_title="Buddy", page_icon="🤝")
st.title("🤝 Buddy")
st.caption("HR onboarding assistant · answers from 10 company policy documents (RAG)")


@st.cache_resource(show_spinner="Loading HR documents...")
def get_store():
    return build_vectorstore()


store, chunks = get_store()

if not os.getenv("GROQ_API_KEY"):
    st.error("Add GROQ_API_KEY to a .env file (local) or to Streamlit Secrets.")
    st.stop()

question = st.text_input("Ask Buddy", placeholder="How many PTO days do I get?")
compare = st.checkbox("Also show the answer without retrieval (pure LLM)")

if question:
    with st.spinner("Searching the HR documents..."):
        answer, sources, results = rag_answer(store, question)
    st.subheader("Buddy (RAG)")
    st.write(answer)
    st.caption("Sources: " + ", ".join(sources))

    with st.expander("Retrieved chunks"):
        for doc, score in results:
            st.markdown(f"**{doc.metadata['source']}** · similarity {score:.2f}")
            st.text(doc.page_content)

    if compare:
        with st.spinner("Asking the LLM without documents..."):
            st.subheader("Pure LLM (no retrieval)")
            st.write(pure_llm_answer(question))
