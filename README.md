# Buddy — RAG Q&A for HR Onboarding

**Assignment 2: RAG System Development** · Anushka Naidu · Seattle University MSDS

Buddy is an HR onboarding assistant I built that answers new-employee questions from a company's HR policy documents. Instead of letting the LLM guess, Buddy first looks up the relevant policy text and then answers only from what it found.

This project started from my original [Buddy](https://github.com/anushkanaidu/buddy) app. In that version the whole handbook was one short block of text, so the "retrieval" step always returned everything and never had to choose. For this assignment I split the handbook into 10 separate documents, built a proper retrieval pipeline, and measured how well it works compared to just prompting the LLM.

## Why HR onboarding?

When you start a new job, you have a lot of small but important questions: How many PTO days do I get? Who do I contact if my laptop breaks? When is my training due? The answers are in policy documents, but nobody wants to read ten documents to find one line. This is a good fit for RAG because:

- the answers are company-specific, so a general LLM can't know them
- a wrong answer (for example about leave or benefits) can actually cause problems for an employee
- the documents are short and factual, which makes it easy to check whether an answer is right

## How it works (Technical Component)

```
10 HR documents → split into chunks → embed → store in Chroma
                                                    ↓
           question → embed → find top 3 similar chunks → LLM answers from those chunks only
```

| Part | What I used |
|---|---|
| Framework | Python with LangChain + Chroma |
| Documents | 10 HR policy documents in `documents/` |
| Chunking | `RecursiveCharacterTextSplitter`, splits on section headings first, then 400-character chunks with 50 characters of overlap (29 chunks total). Every chunk keeps the name of the document it came from. |
| Embeddings | `all-MiniLM-L6-v2` (Hugging Face) |
| Vector store | Chroma with cosine similarity, top 3 chunks per question |
| LLM | Groq `openai/gpt-oss-20b`, temperature 0 |
| Prompt | Answer only from the retrieved policy text, and say "I don't know" if the answer isn't there |

**Why this chunking:** my documents are organized into short sections (like "Carryover" or "Sick leave"), so splitting on headings first keeps each rule together in one chunk. The small overlap means a sentence that falls on a chunk boundary isn't lost.

*Note: my original Buddy used Groq's `llama-3.1-8b-instant`, but Groq has since retired it, so I switched to `openai/gpt-oss-20b`.*

## Documents

The 10 documents cover the main parts of onboarding at a fictional company:

| # | Document | Covers |
|---|---|---|
| 1 | PTO policy | 20 days per year, requests, carryover, sick leave |
| 2 | Background verification | Done before Day 1, HireRight (US) and Credence (UK) |
| 3 | Tools and access | Slack, Asana, Figma, Gemini, Google Workspace |
| 4 | IT support | Who to contact, laptops, passwords |
| 5 | Mandatory training | Harassment Prevention, AI Ethics, Tool Onboarding within 14 days |
| 6 | 1-on-1 meetings | Weekly on Friday with your manager |
| 7 | Onboarding stages | The 8 stages from Intake to 90-Day Review |
| 8 | Remote work | Hybrid schedule, home office stipend |
| 9 | Expense reimbursement | Submit within 30 days, receipts |
| 10 | Benefits | When health insurance starts, enrollment, 401(k) |

## Test questions

I wrote 10 questions (`tests/questions.csv`) that test different things, not just simple lookups:

| Type | Example | What it tests |
|---|---|---|
| Simple fact | How many PTO days do I get per year? | Basic retrieval |
| Specific detail | Which company runs background checks in the UK? | Finding one detail among similar ones |
| List + deadline | Which trainings do I have to finish and by when? | Getting a full list right |
| Procedure | My laptop stopped working. Who should I contact? | Practical "what do I do" questions |
| Multi-document | Which tools will I use, and when is Tool Onboarding due? | Combining two documents |
| Reasoning | If I start on March 10, when does my health insurance begin? | Applying a rule to a date |
| Out of scope | How many weeks of paid parental leave do I get? | Whether it admits it doesn't know |

For each question I labelled in advance which document(s) contain the answer, so I could score retrieval.

## Results

### Retrieval quality

| k | Precision@k | Recall@k | Hit rate | MRR |
|---|---|---|---|---|
| 1 | 0.889 | 0.778 | 0.889 | 0.889 |
| 3 | 0.741 | 0.889 | 1.000 | 0.944 |
| 5 | 0.511 | 0.889 | 1.000 | 0.944 |

With the top 3 chunks, the right document showed up for all 9 answerable questions. Going from 3 to 5 chunks didn't find anything new and only added noise, so I kept k = 3.

### RAG vs. pure LLM

I asked every question twice with the same model: once with the retrieved documents (RAG) and once with just the question (pure LLM). I read and graded every answer myself.

| | Correct | Partly correct | Wrong |
|---|---|---|---|
| RAG | 9 | 1 | 0 |
| Pure LLM | 1 | 1 | 8 |

A few examples:

| Question | RAG | Pure LLM |
|---|---|---|
| How many PTO days do I get? | 20 days ✓ | 15 days ✗ |
| Who runs UK background checks? | Credence ✓ | "Backgrounds" ✗ |
| Paid parental leave? (not in the documents) | Says the documents don't cover it and to ask HR ✓ | "12 weeks" (made up) ✗ |

All answers are in `results/rag_vs_llm.csv`, and the full table is in `results/comparison_table.md`.

## What I learned

- **Without retrieval, the LLM sounds confident but makes things up.** It invented PTO days, a vendor name, a meeting day, and a parental leave policy.
- **The pure LLM was only right when the company's policy happened to match common practice**, like health insurance starting the first of the next month. But as a user you can't tell which answers are real and which are guesses.
- **RAG's weak spot was multi-part questions.** On the Tools + Tool Onboarding question, retrieval missed the training document, and instead of saying "I don't know the deadline," the model guessed "by Day 1" (the real answer is 14 days). So RAG handled *no* information well, but not *half* the information.

The full analysis is in my submitted write-up.

## How to run it

```bash
conda create -n buddyrag python=3.11 -y
conda activate buddyrag
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env

python evaluate.py      # retrieval scores + RAG vs. pure LLM comparison, saved to results/
streamlit run app.py    # the Buddy chat app
```

`python evaluate.py --retrieval` runs only the retrieval part (no API key needed).

## Files

```
app.py               Buddy chat app (Streamlit): shows the answer, its sources, and optionally the pure LLM answer
rag.py               The RAG pipeline: load → chunk → embed → Chroma → retrieve → answer
evaluate.py          Retrieval metrics and the RAG vs. pure LLM comparison
documents/           The 10 HR policy documents
tests/questions.csv  The 10 test questions with expected answers
results/             My evaluation results
```

## Limitations

- The documents are short and fictional, and I wrote both the documents and the questions, which probably makes retrieval easier than it would be with real policies.
- Retrieval is scored by document, not by exact chunk.
- I ran each question once, with one model, and graded the answers myself.
