# Buddy — RAG Q&A for HR Onboarding

**Assignment 2: RAG System Development**  
**Anushka Naidu | Seattle University | MS in Data Science**

**Live demo:** [buddy-rag.streamlit.app](https://buddy-rag.streamlit.app)

## 1. Project Overview

Buddy is an AI-powered HR assistant that answers employee questions using company policy documents.

I originally built [Buddy](https://github.com/anushkanaidu/buddy) using Streamlit, Hugging Face embeddings, and Groq. The original version used a short employee handbook stored directly in Python, so retrieval was relatively simple.

For this assignment, I expanded Buddy into a RAG system using 10 separate HR documents, LangChain, and Chroma. I wanted to see whether retrieving company-specific information would help the LLM give more accurate answers compared to using the LLM alone.

I chose HR onboarding because employees often have questions about PTO, training, benefits, and company policies. A regular LLM might give a reasonable answer, but it does not necessarily know the company's actual rules.

## 2. Building the RAG System

I used **Python with LangChain and Chroma**, as suggested in the assignment.

The pipeline works like this:

**Documents → Chunking → Embeddings → Chroma → Retrieval → LLM → Answer**

Buddy loads the HR documents, splits them into smaller chunks, and converts them into embeddings. When someone asks a question, it retrieves the three most similar chunks and sends them to the Groq LLM to generate an answer.

| Component | Implementation |
|---|---|
| Interface | Streamlit |
| Framework | Python + LangChain |
| Vector database | Chroma |
| Embeddings | Hugging Face `all-MiniLM-L6-v2` |
| LLM | Groq `openai/gpt-oss-20b` |
| Chunking | 400 characters, 50 overlap (29 chunks) |
| Retrieval | Top 3 chunks, cosine similarity |
| Temperature | 0 |

I used LangChain's `RecursiveCharacterTextSplitter` to keep related policy information together. It splits on section headings first, so each policy rule (like "Carryover" or "Sick leave") usually stays in one chunk, and the 50-character overlap keeps sentences near a chunk boundary from being lost. Each chunk also keeps its original document name so Buddy can display the sources.

I instructed the LLM to answer only from the retrieved information and acknowledge when the answer is not available.

My original Buddy used `llama-3.1-8b-instant`, but I switched to `openai/gpt-oss-20b` when the earlier model was no longer available.

In the app, Buddy shows its answer, the source documents, and the retrieved chunks with their similarity scores. There is also an option to show the pure LLM answer (no retrieval) next to it for comparison.

## 3. Documents and Testing

I created 10 fictional HR policy documents covering PTO, background verification, tools, IT support, mandatory training, manager meetings, onboarding stages, remote work, expense reimbursement, and benefits.

All documents are in the `documents/` folder.

I tested Buddy with 10 questions, including simple facts, procedures, multi-document questions, reasoning, and one question that could not be answered from the documents.

Some examples were:

- How many PTO days do I get per year?
- Which company runs background checks in the UK?
- Which tools will I use, and when is Tool Onboarding due?
- If I start on March 10, when does my health insurance begin?
- How many weeks of paid parental leave do I get?

I labeled the relevant documents for each question before testing. The complete question set is in `tests/questions.csv`.

## 4. Retrieval Evaluation

I measured retrieval using Precision@k, Recall@k, Hit Rate, and MRR.

Precision measures how many retrieved chunks came from relevant documents, while recall measures how many of the relevant documents were found. Hit Rate checks whether at least one relevant document was retrieved, and MRR considers how highly the first relevant result was ranked.

### Results

| k | Precision@k | Recall@k | Hit Rate | MRR |
|---|---:|---:|---:|---:|
| 1 | 0.889 | 0.778 | 0.889 | 0.889 |
| 3 | 0.741 | 0.889 | 1.000 | 0.944 |
| 5 | 0.511 | 0.889 | 1.000 | 0.944 |

At k = 3, Buddy retrieved at least one relevant document for all 9 answerable questions. Increasing k to 5 did not improve recall and lowered precision, so I kept k = 3.

I evaluated retrieval based on document relevance. Precision counts each retrieved chunk, even when multiple chunks come from the same document, while recall counts unique relevant documents. The unanswerable question was excluded from these metrics.

The full results are available in `results/`.

## 5. RAG vs. Pure LLM

I tested the same 10 questions twice using the same Groq model: once with retrieved documents (RAG) and once without retrieval (pure LLM).

The original automated keyword check scored RAG 8/10 and the pure LLM 1/10. I noticed that it incorrectly marked the health insurance answer as wrong because of formatting around "April 1." I updated the check to handle this. For the saved responses, correcting that issue gives 9/10 for RAG and 2/10 for the pure LLM.

I also reviewed the answers manually.

| System | Correct | Partly Correct | Incorrect |
|---|---:|---:|---:|
| RAG | 9 | 1 | 0 |
| Pure LLM | 1 | 1 | 8 |

### Examples

| Question | RAG | Pure LLM |
|---|---|---|
| PTO days | 20 days — correct | 15 days — incorrect |
| UK background checks | Credence — correct | Backgrounds — incorrect |
| Manager meetings | Friday — correct | Tuesday — incorrect |
| Tool Onboarding deadline | Day 1 — incorrect (actual: 14 days) | 5 business days — incorrect |
| Paid parental leave | Says information is unavailable | Invented 12 weeks |

The full comparison of all 10 questions is in `results/comparison_table.md`.

RAG performed better because it used company-specific information. The pure LLM often gave confident answers that were not supported by the policies.

However, RAG was not perfect. For the question about tools and Tool Onboarding, it listed the correct tools but gave the wrong deadline. Buddy retrieved the tools document but missed the mandatory training document, which contained the actual 14-day deadline.

This showed me that retrieving some relevant information does not always mean the model has everything needed for a correct answer.

### Testing Buddy Live

I also tested Buddy in Streamlit with a question outside my original test set: "What do I need to complete this week?"

RAG answered using the onboarding and manager meeting documents, mentioning IT setup, orientation, meeting the team, and the Friday 1-on-1. The pure LLM mentioned a training module and timesheet that were not in the documents.

This was another example of how retrieval helped Buddy answer questions based on company policies rather than guessing.

## 6. Reflection and Limitations

This assignment helped me understand that RAG is not just about connecting documents to an LLM. The retrieval step is important because the model can only work with the information it receives.

One limitation was multi-document questions. If Buddy retrieved only part of the required information, it could still generate an incorrect answer.

The documents were also fictional and relatively short, which probably made retrieval easier. I tested each question once using one model, and my evaluation used document relevance and keyword matching rather than manually labeling every retrieved chunk.

If I improved Buddy further, I would focus on checking whether the retrieved information fully answers a question before generating a response.

Overall, RAG improved Buddy's answers, but it did not completely eliminate mistakes.

## 7. How to Run

You can try Buddy without installing anything at [buddy-rag.streamlit.app](https://buddy-rag.streamlit.app).

To run it locally:

```bash
conda create -n buddyrag python=3.11 -y
conda activate buddyrag
pip install -r requirements.txt
```

Create a `.env` file with your own API key:

```env
GROQ_API_KEY=your_api_key_here
```

Then run:

```bash
python evaluate.py --retrieval   # retrieval metrics only (no API key needed)
python evaluate.py               # retrieval + RAG vs. pure LLM comparison
streamlit run app.py             # Buddy app
```

The evaluation results are saved in `results/`.

## Repository Files

| File/Folder | Purpose |
|---|---|
| `app.py` | Streamlit chatbot |
| `rag.py` | RAG pipeline |
| `evaluate.py` | Retrieval and LLM evaluation |
| `documents/` | 10 HR policy documents |
| `tests/questions.csv` | Test questions and expected answers |
| `results/` | Evaluation results and comparison tables |

## Conclusion

Building on my original Buddy project helped me understand how retrieval changes an LLM's responses. Buddy performed better with access to HR documents, but the multi-document question showed that RAG can still make mistakes when important information is missing.

My biggest takeaway is that **RAG works best when the system retrieves the right information and the LLM uses it correctly.**
