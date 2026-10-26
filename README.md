---
title: Comprehensive Legal AI
emoji: ⚖️
colorFrom: indigo
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Comprehensive Legal AI Platform

Welcome to the **Comprehensive Legal AI Platform**! This is a state-of-the-art **Retrieval-Augmented Generation (RAG)** application designed to analyze legal documents (such as contracts, agreements, statutes, and judgements).

## Core Capabilities

- **Universal Parser**: Extracts clean text from native PDFs, scanned PDFs (via OCR), Word documents (`.docx`), AsciiDoc (`.adoc`), raw text (`.txt`), and HTML files (`.html`/`.htm`).
- **Structured Search & Schema Validation**: Validates metadata fields (parties, dates, governing law, types) using strict Pydantic schemas before indexing, with search filtering over weakness categories and metadata.
- **Rule-Based Risk Engine**: Instantly checks for common contract risks like auto-renewals, unlimited liability clauses, missing governing law, and missing liability limitations.
- **AI-Powered Summarization**: Generates a professional legal summary of uploaded files using Groq-powered LLMs.
- **Interactive Dialogue Q&A**: Lets you chat with the contract, returning exact citations (source file, clause heading, and matching snippet).
- **Claim-Evidence Taxonomy & Grounding Validation**: Implements a post-hoc verification pass checking cited clauses and word-overlap similarity to classify answers as `supported`, `partially supported`, `unsupported`, or `citation error` before returning. Achieves 100% detection (15 of 15) on injected fabricated citations and 80.0% grounded faithfulness.
- **Ingestion Poisoning Detection & Upload Hardening**: Employs anomaly outlier detection on embedding distributions to flag injected or off-topic adversarial text at upload time, backed by a 15MB file size boundary middleware.
- **Rigorous Retrieval & Grounded-Answer Evaluation**: Evaluated over a 120-question benchmark across 60 legal clauses with a strict Dev (40 queries) and Held-Out Test (80 queries) split, comparing BM25, FAISS (dense), and Hybrid with 1,000-resample Bootstrap 95% Confidence Intervals. (See [Retrieval Evaluation Report](reports/retrieval_eval.md)).
- **Automated CI Regression Gate**: Pytest suite includes an automated gate (`tests/test_retrieval_eval_gate.py`) asserting Recall@3 $\ge 0.85$, MRR $\ge 0.80$, and citation error detection on every run.
- **Glassmorphic UI Dashboard**: A sleek, modern dark-themed single-page dashboard for drag-and-drop ingestion and interaction.

---

## Retrieval & Grounding Evaluation Benchmark

The retrieval engine was evaluated over **60 legal contract clauses** (spanning Indian rental leases and commercial contracts aligned with CUAD categories) and **120 questions** split into a **Dev Set ($N=40$)** for hyperparameter tuning and a **Held-Out Test Set ($N=80$)** evaluated once:

### Held-Out Test Set Results ($N=80$, with Bootstrap 95% Confidence Intervals)

| Configuration | Precision@1 (95% CI) | Recall@3 (95% CI) | Recall@5 | MRR (95% CI) | nDCG@10 | Latency (p50 / p95) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25-only ($w=1.0$)** | 0.713 [0.613, 0.812] | 0.825 [0.750, 0.900] | 0.838 | 0.775 [0.700, 0.853] | 0.805 | 28.8ms / 32.8ms |
| **FAISS-only ($w=0.0$)** | 0.887 [0.812, 0.950] | 0.963 [0.912, 1.000] | 0.988 | 0.930 [0.880, 0.970] | 0.947 | 27.9ms / 30.6ms |
| **Hybrid Optimized ($w=0.10$)** | **0.900** [0.837, 0.963] | **0.963** [0.925, 1.000] | **0.975** | **0.938** [0.894, 0.975] | **0.953** | 28.3ms / 30.3ms |
| **Hybrid Legacy ($w=0.40$)** | 0.850 [0.775, 0.925] | 0.912 [0.850, 0.963] | 0.938 | 0.890 [0.831, 0.943] | 0.913 | 29.2ms / 31.9ms |

### Defensible Trade-off by Query Typology ($N=40$ each):
- **Semantic Queries (Synonyms/Paraphrasing):** FAISS-only leads P@1 (**0.825** vs. BM25's 0.500) because dense MiniLM embeddings capture high-level conceptual intent without exact lexical overlap.
- **Exact-Token Queries (Currencies, Dates, Section Codes):** BM25 achieves **0.925** P@1, while Hybrid ($w=0.10$) fuses exact keyword matching with dense ranking to achieve **0.975** Precision@1 and **1.000** Recall@3, outperforming FAISS-only.
- **Production Choice:** Hybrid ($w=0.10$) delivers the highest overall held-out P@1 (0.900), Recall@3 (0.963), and MRR (0.938), safeguarding exact keyword queries without sacrificing dense semantic recall.

### Citation Verification & Faithfulness:
- **Grounded Faithfulness:** 80.0% of evaluated answers fully supported by retrieved context.
- **Hallucinated Citation Detection:** **15 of 15 (100.0%)** intentionally injected fabricated citations caught by the automated verification pass.
- **End-to-End Cost:** ~$0.00015 per grounded query using Groq LLaMA-3.

---

## Step-by-Step Installation

Follow these steps to set up and run the application on your computer:

### 1. Prerequisites
Make sure you have python installed. You can check this by opening a terminal (Command Prompt, PowerShell, or bash) and typing:
```bash
python --version
```
*(Recommended version is Python 3.10 or 3.11).*

### 2. Clone the Repository
Clone the project code to your local machine:
```bash
git clone https://github.com/Pranavsk22/Legal-AI-v2.0.git
cd Legal-AI-v2.0
```

### 3. Create a Virtual Environment
A virtual environment keeps your project dependencies isolated.

- **On Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
- **On macOS / Linux (Terminal)**:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```

### 4. Install Dependencies
Install all required libraries inside the virtual environment:
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
1. Duplicate the file named `.env.example` and rename it to `.env` in the root folder.
2. Open the `.env` file in a text editor.
3. Obtain your free API keys from the **[Groq Console](https://console.groq.com)**.
4. Replace the placeholder values with your keys:
   ```env
   GROQ_API_KEY_SUMMARY=gsk_your_summary_key_here
   GROQ_API_KEY_QA=gsk_your_qa_key_here
   ```
   *(Note: You can use the same key for both variables if desired).*

---

## Running the Platform

Once installed and configured, you can launch the platform:

### 1. Start the FastAPI Server
Run the FastAPI web backend:
```bash
uvicorn backend.api.main:app --reload --port 7860
```
*(The server will reload automatically if you modify any code).*

### 2. Open the Dashboard in your Browser
Navigate to **[http://localhost:7860](http://localhost:7860)** to open the interactive Legal AI Web UI.
- Drag & drop or browse a legal file (e.g., text, Word, HTML, or PDF contract).
- Instantly review the **AI-Generated Summary** and the **Contract Risk Assessment**.
- Use the **Interactive Dialogue** chat panel to ask questions (e.g., *"What is the notice period?"* or *"Who is liable?"*).

---

## Testing the Project

To ensure everything is working correctly, run the full automated test suite:

```bash
# On Windows PowerShell
$env:PYTHONPATH="." ; .venv\Scripts\pytest.exe

# On macOS/Linux Terminal
PYTHONPATH=. pytest
```

---

## Project Architecture

```
Legal-AI-v2.0/
├── backend/                       # Python FastAPI Backend & Core NLP
│   ├── api/                       # API Routing & Web Interface
│   │   ├── main.py                # FastAPI entrypoint (serves index.html UI)
│   │   ├── routes.py              # Ingestion & Ask routers
│   │   └── index.html             # Sleek Glassmorphic Frontend Dashboard
│   └── nlp_modules/               # Document Parsing, Embeddings, Database & LLM
│       ├── html_parser.py         # Parses HTML documents with BeautifulSoup
│       ├── docx_parser.py         # Parses Word documents
│       ├── pdf_parser.py          # Native PDF parsing & pytesseract OCR fallback
│       ├── universal_parser.py    # Routing hub for document ingestion
│       ├── embedder.py            # Sentence-Transformers MiniLM vector generator
│       ├── vector_store.py        # FAISS Index + BM25 Hybrid Search scoring & poisoning check
│       ├── risk_rules.py          # Regex contract risk detection logic
│       └── summarizer.py          # Groq LLaMA-3 client integrations & citation verification
├── tests/                         # Pytest Suite (Unit & Integration tests)
├── scripts/                       # Offline CLI utilities (indexing & searching)
├── Dockerfile                     # Deployment blueprint
└── requirements.txt               # Main Python libraries list
```
