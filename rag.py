"""Buddy RAG pipeline: load HR documents, chunk, embed, store in Chroma, retrieve, generate.

Used by both app.py (the chat UI) and evaluate.py (the evaluation).
"""
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path(__file__).parent / "documents"
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Chunking strategy: each document is split on markdown sections first,
# then into ~400-character chunks with 50 characters of overlap. Every chunk
# keeps its source file name so retrieval can be scored per document.
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K = 3


def load_documents(docs_dir=DOCS_DIR):
    docs = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        docs.append(Document(page_content=path.read_text(), metadata={"source": path.name}))
    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(docs)


def build_vectorstore(docs_dir=DOCS_DIR):
    chunks = chunk_documents(load_documents(docs_dir))
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    # In-memory Chroma collection, rebuilt at startup (10 small documents).
    store = Chroma.from_documents(
        chunks, embeddings, collection_name="buddy_hr",
        collection_metadata={"hnsw:space": "cosine"},  # cosine similarity
    )
    return store, chunks


def retrieve(store, question, k=TOP_K):
    """Return the top-k chunks as (Document, similarity score) pairs."""
    return store.similarity_search_with_relevance_scores(question, k=k)


RAG_PROMPT = """You are Buddy, a friendly HR and onboarding assistant.
Answer the employee's question using ONLY the HR context below.
If the context does not contain the answer, say you don't know and suggest contacting HR.
Keep the answer short (2-4 sentences).

HR context:
{context}

Question: {question}
Answer:"""

PURE_LLM_PROMPT = """You are Buddy, a friendly HR and onboarding assistant at a company.
Answer the employee's question. Keep the answer short (2-4 sentences).

Question: {question}
Answer:"""


def _groq_client():
    from groq import Groq

    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to a .env file or your environment.")
    return Groq(api_key=key)


def _generate(prompt, client=None):
    client = client or _groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def rag_answer(store, question, client=None, k=TOP_K):
    results = retrieve(store, question, k=k)
    context = "\n\n".join(f"[{doc.metadata['source']}]\n{doc.page_content}" for doc, _ in results)
    answer = _generate(RAG_PROMPT.format(context=context, question=question), client)
    sources = sorted({doc.metadata["source"] for doc, _ in results})
    return answer, sources, results


def pure_llm_answer(question, client=None):
    """Same model, no retrieval: the baseline for the comparison."""
    return _generate(PURE_LLM_PROMPT.format(question=question), client)
