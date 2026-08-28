---
title: Comprehensive Legal AI
emoji: ⚖️
colorFrom: indigo
colorTo: indigo
sdk: docker
sdk_version: "1.0"
app_port: 7860
pinned: false
---

# ⚖️ Comprehensive Legal AI Platform

Welcome to the **Comprehensive Legal AI Platform**! This is a state-of-the-art **Retrieval-Augmented Generation (RAG)** application designed to analyze legal documents (such as contracts, agreements, statutes, and judgements).

It features:
- **Universal Parser**: Extracts clean text from native PDFs, scanned PDFs (via OCR), Word documents (`.docx`), AsciiDoc (`.adoc`), raw text (`.txt`), and HTML files (`.html`/`.htm`).
- **Structured Search & Schema Validation**: Validate metadata fields (parties, dates, governing law, types) using strict Pydantic schemas before indexing, with search filtering over weakness categories and metadata.
- **Rule-based Risk Engine**: Instantly checks for common contract risks like auto-renewals, unlimited liability clauses, missing governing law, and missing liability limitations.
- **AI-Powered Summarization**: Generates a professional legal summary of uploaded files using Groq-powered LLMs.
- **Interactive Dialogue Q&A**: Lets you chat with the contract, returning exact citations (source file, clause heading, and matching snippet).
- **Retrieval Evaluation**: Benchmarked search and QA grounding reports showing retrieval accuracy (Precision/Recall) and RAG accuracy vs. baseline LLM knowledge. (See [Retrieval Evaluation Report](reports/retrieval_eval.md)).
- **Stunning UI Dashboard**: A sleek, glassmorphic dark-themed single-page dashboard for drag-and-drop ingestion and interaction.

---

## 🛠️ Step-by-Step Installation (For Beginners)

Follow these simple steps to set up and run the application on your computer:

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
A virtual environment keeps your project dependencies isolated from the rest of your computer.

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

## 🚀 Running the Platform

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

## 🧪 Testing the Project

To ensure everything is working correctly, we have set up a full suite of automated tests.

Run pytest in your terminal:
```bash
# On Windows PowerShell
$env:PYTHONPATH="." ; .venv\Scripts\pytest.exe

# On macOS/Linux Terminal
PYTHONPATH=. pytest
```

---

## 📂 Project Architecture

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
│       ├── vector_store.py        # FAISS Index + BM25 Hybrid Search scoring
│       ├── risk_rules.py          # Regex contract risk detection logic
│       └── summarizer.py          # Groq LLaMA-3 client integrations (lazy-loaded)
├── tests/                         # Pytest Suite (Unit & Integration tests)
├── scripts/                       # Offline CLI utilities (indexing & searching)
├── Dockerfile                     # Deployment blueprint
└── requirements.txt               # Main Python libraries list
```
