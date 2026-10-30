# Buddy RAG — HR Onboarding Q&A (Assignment 2)

Buddy is a retrieval-augmented generation (RAG) assistant that answers new-employee questions from a company's HR policy documents. It grew out of my original [Buddy](https://github.com/anushkanaidu/buddy) prototype, which kept the whole handbook in one short text. This version splits the handbook into 10 documents so that retrieval actually has to choose, and adds an evaluation.

## Pipeline

| Step | Implementation |
|---|---|
| Documents | 10 HR policy documents in `documents/` (PTO, background verification, tools, IT support, training, 1-on-1s, onboarding stages, remote work, expenses, benefits) |
| Chunking | LangChain `RecursiveCharacterTextSplitter`: split on section headings first, then ~400-character chunks with 50-character overlap → 29 chunks. Each chunk keeps its source file name. |
| Embeddings | Hugging Face `all-MiniLM-L6-v2` |
| Vector store | Chroma (cosine similarity), top-k = 3 |
| Generation | Groq `openai/gpt-oss-20b`, temperature 0 (Groq retired the Llama 3.1 model used by the original Buddy), prompt restricted to the retrieved context with an "I don't know" instruction |
| Baseline | Same model and settings, no retrieval (pure LLM) |

## Files

```
app.py               Streamlit chat UI (shows sources, retrieved chunks, optional pure-LLM answer)
rag.py               RAG pipeline (load → chunk → embed → Chroma → retrieve → generate)
evaluate.py          Retrieval metrics + RAG vs. pure LLM comparison
documents/           10 HR policy documents
tests/questions.csv  10 test questions with labelled relevant documents and expected facts
results/             Evaluation output (created by evaluate.py)
```

## Test questions

10 questions of different types: simple facts, specific details, lists with deadlines, procedures, multi-document questions, a reasoning question (date calculation), and one out-of-scope question (parental leave, not covered by any document) to test whether the system admits it doesn't know.

## Evaluation

- **Retrieval quality:** precision@k, recall@k, hit rate and MRR for k = 1, 3, 5, scored against the labelled relevant documents.
- **RAG vs. pure LLM:** each question answered with and without retrieval; answers checked for the expected facts (or an honest "I don't know" for the out-of-scope question). The keyword check is a first pass; answers were also read manually.

## Run it

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then paste your Groq API key into .env

python evaluate.py            # writes results/ (retrieval + comparison)
streamlit run app.py          # chat UI
```

`python evaluate.py --retrieval` runs only the retrieval metrics (no API key needed).

The documents describe a fictional company handbook created for this project.

## Results (Groq openai/gpt-oss-20b, run on Oct 30, 2026)

| k | Precision@k | Recall@k | Hit rate | MRR |
|---|---|---|---|---|
| 1 | 0.889 | 0.778 | 0.889 | 0.889 |
| 3 | 0.741 | 0.889 | 1.000 | 0.944 |
| 5 | 0.511 | 0.889 | 1.000 | 0.944 |

RAG vs. pure LLM, graded by hand: **RAG 9/10 correct (+1 partial), pure LLM 1/10 correct (+1 partial)**. Full answers: `results/rag_vs_llm.csv`; table: `results/comparison_table.md`. The full evaluation and analysis are in the submitted write-up.
