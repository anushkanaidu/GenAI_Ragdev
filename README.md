# Buddy — RAG Q&A for HR Onboarding

**Assignment 2: RAG System Development**  
**Anushka Naidu | Seattle University | MS in Data Science**

## Project Overview

Buddy is an AI-powered HR onboarding assistant that helps new employees find answers to questions about company policies, training, benefits, and onboarding procedures.

I originally built [Buddy](https://github.com/anushkanaidu/buddy) as an HR assistant using Streamlit, Hugging Face embeddings, and Groq. The original version had a small employee handbook written directly into the Python code. It used embeddings and cosine similarity to retrieve relevant information, but because the handbook was short, the retrieval process was relatively simple.

For this assignment, I wanted to improve the RAG part of Buddy and actually evaluate how well it works. I expanded the handbook into 10 separate HR policy documents, used LangChain and Chroma for document processing and retrieval, and tested whether adding retrieved information helps the LLM give more accurate answers.

My main goal was to understand how retrieval affects the quality of an LLM's response, especially when the answer depends on information the model would not normally know.

## 1. Why I Chose HR Onboarding

I chose HR onboarding because it is something almost every employee experiences when starting a new job.

New employees usually have questions like:

- How many PTO days do I get?
- What training do I need to complete?
- Who should I contact if my laptop stops working?
- When does my health insurance start?
- What do I need to finish before my first day?

The answers are usually somewhere in company documents, but finding the right information can take time.

I thought this would be a good use case for RAG because the information is specific to a company. A regular LLM might give a general answer based on common HR practices, but that does not necessarily mean the answer matches the company's actual policies.

For HR-related questions, incorrect information could also create confusion for employees. I wanted Buddy to retrieve relevant policies first instead of relying only on what the LLM already knows.

## 2. RAG System Development

For the technical component, I used **Python with LangChain and Chroma**, which is one of the approaches provided in the assignment instructions.

### How the System Works

The RAG pipeline follows these steps:

1. **Document loading:** Buddy loads the 10 HR policy documents from the `documents/` folder.
2. **Chunking:** The documents are divided into smaller sections using LangChain's `RecursiveCharacterTextSplitter`.
3. **Embeddings:** Each chunk is converted into a numerical vector using Hugging Face embeddings.
4. **Vector storage:** The chunks and embeddings are stored in Chroma.
5. **Retrieval:** When a user asks a question, Buddy converts the question into an embedding and retrieves the three most similar chunks.
6. **Answer generation:** The retrieved information is passed to the Groq LLM, which generates an answer based on the provided context.

The system also displays the source documents used to generate an answer.

### Tools and Configuration

| Component | What I Used |
|---|---|
| Programming language | Python |
| Framework | LangChain |
| Interface | Streamlit |
| Vector database | Chroma |
| Embedding model | Hugging Face `all-MiniLM-L6-v2` |
| LLM | Groq `openai/gpt-oss-20b` |
| Chunking | RecursiveCharacterTextSplitter |
| Chunk size | 400 characters |
| Chunk overlap | 50 characters |
| Retrieval | Cosine similarity, top 3 chunks |
| Temperature | 0 |

I used a chunk size of 400 characters with an overlap of 50 characters. Since the HR documents contain relatively short policy sections, I wanted to keep related information together instead of splitting individual rules across too many chunks.

The splitter prioritizes section boundaries and paragraph breaks before splitting longer text. The overlap helps preserve context when information falls near a chunk boundary.

Each chunk also keeps the name of its original document so I can identify where the retrieved information came from.

For answer generation, I instructed the LLM to use only the retrieved HR context. If the answer is not available, it should acknowledge that and suggest contacting HR instead of guessing.

My original Buddy used Groq's `llama-3.1-8b-instant`, but that model was no longer available when I worked on this assignment. I switched to `openai/gpt-oss-20b` so I could continue using Groq for answer generation.

## 3. Document Collection

I created 10 fictional HR policy documents for a sample company. Each document focuses on a different part of the employee onboarding experience.

| # | Document | Information Covered |
|---|---|---|
| 1 | PTO Policy | Annual PTO, requests, carryover, sick leave |
| 2 | Background Verification | Background checks before Day 1, US and UK vendors |
| 3 | Tools and Access | Slack, Asana, Figma, Gemini, Google Workspace |
| 4 | IT Support | Laptop problems, password resets, technical support |
| 5 | Mandatory Training | Required training and completion deadlines |
| 6 | 1-on-1 Meetings | Manager meetings and frequency |
| 7 | Onboarding Stages | Employee onboarding from Intake to 90-Day Review |
| 8 | Remote Work | Hybrid work schedule and home office stipend |
| 9 | Expense Reimbursement | Expense submission deadlines and receipts |
| 10 | Benefits | Health insurance, enrollment, and retirement benefits |

All documents are available in the `documents/` folder.

I used fictional documents because they allowed me to create a consistent knowledge base and verify the correct answers. They also made it easier to identify when the LLM provided information that was not present in the documents.

One limitation is that these policies are shorter and more organized than real company documents, so retrieval may be easier than it would be in a real HR environment.

## 4. Testing the RAG System

To evaluate Buddy, I created 10 questions covering different types of information retrieval.

I did not want to test only straightforward questions. I also included questions that required multiple documents, applying a policy rule, and recognizing when information was missing.

| Question Type | Example | Purpose |
|---|---|---|
| Simple fact | How many PTO days do I get per year? | Test basic retrieval |
| Specific detail | Which company runs background checks in the UK? | Retrieve an exact detail |
| List and deadline | Which trainings do I need to finish and by when? | Retrieve multiple related facts |
| Procedure | My laptop stopped working. Who should I contact? | Test practical HR questions |
| Multi-document | Which tools will I use, and when is Tool Onboarding due? | Combine information across documents |
| Reasoning | If I start on March 10, when does my health insurance begin? | Apply a policy to a specific situation |
| Out of scope | How many weeks of paid parental leave do I get? | Check whether the system avoids making up an answer |

The complete set of 10 test questions and their expected answers is in `tests/questions.csv`.

I also labeled the relevant document or documents for each question before running the evaluation. This gave me a reference for measuring whether Buddy retrieved the correct information.

## 5. Retrieval Evaluation

I measured retrieval quality using Precision@k, Recall@k, Hit Rate, and Mean Reciprocal Rank (MRR).

- **Precision@k:** The proportion of retrieved chunks that came from documents labeled as relevant.
- **Recall@k:** The proportion of unique relevant documents found in the top k retrieved chunks.
- **Hit Rate:** Whether at least one relevant document was retrieved.
- **MRR:** How highly the first relevant result appeared in the retrieved results.

I calculated these metrics using the nine questions that had answers in the documents. I excluded the out-of-scope question from retrieval scoring because no document contained its answer.

### Retrieval Results

| k | Precision@k | Recall@k | Hit Rate | MRR |
|---|---:|---:|---:|---:|
| 1 | 0.889 | 0.778 | 0.889 | 0.889 |
| 3 | 0.741 | 0.889 | 1.000 | 0.944 |
| 5 | 0.511 | 0.889 | 1.000 | 0.944 |

### What I Observed

At k = 1, retrieval was fairly precise, but some questions needed information from more than one document.

At k = 3, the hit rate reached 1.0, meaning at least one relevant document was retrieved for every answerable question. However, Recall@3 was still 0.889 because some multi-document questions did not retrieve all the relevant documents.

Increasing k from 3 to 5 did not improve recall. Instead, precision dropped from 0.741 to 0.511 because more less-relevant chunks were included.

Based on this, I kept **k = 3** as the retrieval setting for Buddy.

One thing to note is that I evaluated the retrieved chunks based on their source document. Multiple chunks from the same relevant document were counted separately for precision, while recall measured unique relevant documents. This is useful for evaluating document retrieval, but it does not measure whether each individual chunk contains the exact sentence needed to answer the question.

The full retrieval results are saved in `results/retrieval_results.csv` and `results/retrieval_summary.csv`.

## 6. RAG vs. Pure LLM Comparison

For the comparison, I tested the same 10 questions using the same Groq model in two ways:

1. **RAG:** The LLM received the question along with the top three retrieved document chunks.
2. **Pure LLM:** The LLM received only the question without any company policy documents.

I kept the model and temperature the same so I could focus on how retrieval affected the answers.

The evaluation script checked the generated answers against expected keywords. I also reviewed the actual responses because keyword matching alone does not always reflect whether an answer is correct.

### Automated Evaluation Results

The original automated evaluation marked 8 out of 10 RAG answers and 1 out of 10 pure LLM answers as correct.

| System | Correct | Incorrect |
|---|---:|---:|
| RAG (original automated results) | 8 | 2 |
| Pure LLM (original automated results) | 1 | 9 |

When I reviewed the responses, I noticed that the health insurance question was marked incorrectly because of a formatting issue with the date "April 1."

After noticing this, I updated the keyword check to handle the formatting. With that correction, the existing answers should score 9/10 for RAG and 2/10 for the pure LLM. The actual responses did not change, only how their correctness was checked.

### Manual Review

I also reviewed the complete answers instead of relying only on keyword matching.

| System | Correct | Partly Correct | Incorrect |
|---|---:|---:|---:|
| RAG | 9 | 1 | 0 |
| Pure LLM | 1 | 1 | 8 |

I marked the RAG answer to the tools and onboarding question as partly correct because Buddy listed the tools accurately but gave the wrong training deadline.

For the pure LLM, the expense reimbursement answer included the correct 30-day deadline but also added an unsupported exception, so I considered it partly correct.

### Examples from the Comparison

| Question | RAG Answer | Pure LLM Answer |
|---|---|---|
| How many PTO days do I get? | 20 days — correct | 15 days — incorrect |
| Who handles UK background checks? | Credence — correct | Backgrounds — incorrect |
| When are manager 1-on-1s? | Every Friday — correct | Every other Tuesday — incorrect |
| When is Tool Onboarding due? | Day 1 — incorrect deadline | Five business days — incorrect deadline |
| How much paid parental leave is available? | Says the documents do not specify — appropriate | 12 weeks — unsupported |

### What I Learned from the Comparison

The biggest difference I noticed was that the pure LLM could generate answers that sounded reasonable even when it did not actually know the company's policies.

For example, it gave 15 PTO days instead of 20 and named the wrong background verification company. It also invented a parental leave policy even though that information was not available.

RAG performed better because it had access to the relevant company information.

However, RAG was not perfect.

For the question about tools and Tool Onboarding, Buddy retrieved the tools document but missed the mandatory training document. It correctly listed the tools but gave the wrong deadline.

The actual deadline was 14 days, but Buddy answered Day 1.

I found this interesting because it showed that even when the retrieved information is partly correct, the LLM can still make mistakes when it does not have enough context.

So retrieval improves the quality of the answers, but it does not automatically guarantee that every answer will be accurate.

The complete responses are available in `results/rag_vs_llm.csv` and `results/comparison_table.md`.

## 7. Reflection and Limitations

This assignment helped me understand RAG beyond just connecting documents to an LLM.

Before this, I mostly focused on whether Buddy could find information and generate an answer. By evaluating retrieval separately, I could see that a system might retrieve something relevant without retrieving everything needed for the question.

I also learned that evaluating RAG involves two different things: whether the correct evidence was retrieved and whether the model used that evidence correctly.

One of the biggest limitations was the multi-document question. When Buddy retrieved only part of the needed information, it still generated an answer instead of acknowledging that the deadline was missing.

If I were to improve this system, I would focus on checking whether the retrieved context contains all the information needed before generating an answer. I would also consider testing different chunk sizes or retrieval settings to see if multi-document questions improve.

There are a few other limitations:

- The documents are fictional and relatively short.
- I created both the documents and the test questions, which may make the evaluation easier.
- Retrieval relevance was evaluated at the document level rather than by manually labeling every chunk.
- I used one model and ran each question once.
- The automatic correctness check uses keywords, so manual review is still important.

Overall, I think RAG is useful for applications where answers depend on specific documents, but the quality of the results still depends on document processing, retrieval, and how the LLM handles incomplete information.

## 8. How to Run the Project

The project can be run locally using Python.

### Install Dependencies

```bash
conda create -n buddyrag python=3.11 -y
conda activate buddyrag
pip install -r requirements.txt
```

### Configure the API Key

Create a `.env` file in the project folder and add:

```env
GROQ_API_KEY=your_api_key_here
```

The API key should remain private and should not be uploaded to GitHub.

### Run Retrieval Evaluation

```bash
python evaluate.py --retrieval
```

This evaluates retrieval performance without making LLM API calls.

### Run the Full Evaluation

```bash
python evaluate.py
```

This runs the retrieval evaluation and compares RAG responses against the pure LLM baseline.

The output files are saved in the `results/` folder.

### Run Buddy

```bash
streamlit run app.py
```

This opens the Streamlit application, where users can ask questions, view Buddy's answers, inspect retrieved chunks, and optionally compare answers against a pure LLM.

## 9. Repository Structure

```text
GenAI_Ragdev/
│
├── app.py
├── rag.py
├── evaluate.py
├── requirements.txt
├── README.md
│
├── documents/
│   └── 10 HR policy documents
│
├── tests/
│   └── questions.csv
│
└── results/
    ├── retrieval_results.csv
    ├── retrieval_summary.csv
    ├── rag_vs_llm.csv
    └── comparison_table.md
```

## Conclusion

Through this assignment, I was able to build on my original Buddy project and evaluate how retrieval affects the accuracy of an LLM's answers.

The results showed that RAG performed better than the pure LLM for questions that depended on company-specific HR policies. At the same time, the multi-document question showed that RAG can still generate incorrect information when retrieval misses important evidence.

My biggest takeaway is that **RAG is not just about giving an LLM access to documents. It is also about retrieving the right information and making sure the model uses it correctly.**
