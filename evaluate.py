"""Evaluate Buddy.

Part 1 - Retrieval quality (no API key needed):
    precision@k, recall@k, hit rate and MRR for each test question,
    using the labelled relevant documents in tests/questions.csv.

Part 2 - RAG vs. pure LLM (needs GROQ_API_KEY):
    asks every question with and without retrieval and checks each answer
    for the expected facts (or, for the out-of-scope question, an honest "I don't know").

Run:  python evaluate.py              (both parts)
      python evaluate.py --retrieval  (part 1 only)
"""
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from rag import TOP_K, build_vectorstore, pure_llm_answer, rag_answer, retrieve, _groq_client

load_dotenv()
ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

ABSTAIN_PHRASES = [
    "don't know", "do not know", "don't have", "do not have", "not mentioned",
    "not in the", "doesn't mention", "does not mention", "no information",
    "not provided", "contact hr", "reach out to hr", "check with hr",
]


def load_questions():
    df = pd.read_csv(ROOT / "tests" / "questions.csv", keep_default_na=False)
    df["relevant"] = df["relevant_docs"].apply(lambda s: [d for d in s.split(";") if d])
    return df


def score_retrieval(store, questions, k):
    rows = []
    for _, q in questions.iterrows():
        results = retrieve(store, q["question"], k=k)
        sources = [doc.metadata["source"] for doc, _ in results]
        top_score = results[0][1] if results else 0.0
        row = {
            "id": q["id"], "question": q["question"], "type": q["type"],
            "retrieved": "; ".join(sources), "top_similarity": round(top_score, 3),
        }
        if q["relevant"]:
            relevant = set(q["relevant"])
            hits = [s in relevant for s in sources]
            first = next((i + 1 for i, h in enumerate(hits) if h), None)
            row.update({
                f"precision@{k}": sum(hits) / k,
                f"recall@{k}": len(relevant & set(sources)) / len(relevant),
                "hit": int(any(hits)),
                "reciprocal_rank": 1 / first if first else 0.0,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def run_retrieval(store, questions):
    print("\n=== Part 1: Retrieval quality ===")
    df = score_retrieval(store, questions, TOP_K)
    df.to_csv(RESULTS / "retrieval_results.csv", index=False)
    show = ["id", "type", "retrieved", f"precision@{TOP_K}", f"recall@{TOP_K}", "top_similarity"]
    print(df[show].to_string(index=False))

    # Summary at several k values, for the analysis (in-scope questions only).
    summary = []
    for k in (1, 3, 5):
        d = score_retrieval(store, questions, k).dropna(subset=["hit"])
        summary.append({
            "k": k,
            "precision@k": round(d[f"precision@{k}"].mean(), 3),
            "recall@k": round(d[f"recall@{k}"].mean(), 3),
            "hit_rate": round(d["hit"].mean(), 3),
            "MRR": round(d["reciprocal_rank"].mean(), 3),
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(RESULTS / "retrieval_summary.csv", index=False)
    print("\nAverages over the 9 in-scope questions:")
    print(summary.to_string(index=False))
    return df, summary


def _normalize(text):
    # The model sometimes uses non-breaking spaces/hyphens and curly quotes.
    import unicodedata
    text = unicodedata.normalize("NFKC", text)
    for ch in ("\u00a0", "\u202f", "\u2009"):
        text = text.replace(ch, " ")
    text = text.replace("\u2011", "-").replace("\u2019", "'").replace("*", "")
    return text.lower()


def is_correct(answer, expected):
    a = _normalize(answer)
    if expected == "ABSTAIN":
        return any(p in a for p in ABSTAIN_PHRASES)
    return all(kw.lower() in a for kw in expected.split("|"))


def run_comparison(store, questions):
    print("\n=== Part 2: RAG vs. pure LLM ===")
    client = _groq_client()
    rows = []
    for _, q in questions.iterrows():
        rag, sources, _ = rag_answer(store, q["question"], client=client)
        time.sleep(1)
        llm = pure_llm_answer(q["question"], client=client)
        time.sleep(1)
        rows.append({
            "id": q["id"], "question": q["question"], "type": q["type"],
            "expected": q["expected_keywords"],
            "rag_answer": rag, "rag_sources": "; ".join(sources),
            "rag_correct": is_correct(rag, q["expected_keywords"]),
            "llm_answer": llm,
            "llm_correct": is_correct(llm, q["expected_keywords"]),
        })
        print(f"{q['id']}: RAG {'✓' if rows[-1]['rag_correct'] else '✗'} | "
              f"LLM {'✓' if rows[-1]['llm_correct'] else '✗'}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "rag_vs_llm.csv", index=False)

    # Markdown comparison table for the write-up.
    lines = ["| ID | Question | Type | RAG answer | RAG ✓ | Pure LLM answer | LLM ✓ |",
             "|---|---|---|---|---|---|---|"]
    clean = lambda s: s.replace("\n", " ").replace("|", "/")
    for r in rows:
        lines.append(f"| {r['id']} | {r['question']} | {r['type']} | {clean(r['rag_answer'])} | "
                     f"{'✓' if r['rag_correct'] else '✗'} | {clean(r['llm_answer'])} | "
                     f"{'✓' if r['llm_correct'] else '✗'} |")
    lines.append(f"\n**Accuracy:** RAG {df['rag_correct'].sum()}/{len(df)} · "
                 f"Pure LLM {df['llm_correct'].sum()}/{len(df)}")
    (RESULTS / "comparison_table.md").write_text("\n".join(lines) + "\n")
    print(f"\nRAG correct: {df['rag_correct'].sum()}/{len(df)} | "
          f"Pure LLM correct: {df['llm_correct'].sum()}/{len(df)}")
    print("Note: correctness is a keyword check. Read the answers in results/rag_vs_llm.csv to confirm.")
    return df


if __name__ == "__main__":
    store, chunks = build_vectorstore()
    print(f"Indexed {len(chunks)} chunks from 10 documents.")
    questions = load_questions()
    run_retrieval(store, questions)
    if "--retrieval" not in sys.argv:
        run_comparison(store, questions)
    print(f"\nResults saved in {RESULTS}/")
