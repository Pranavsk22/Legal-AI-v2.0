"""
summarizer.py
=============

• Two separate Groq keys:
    - GROQ_API_KEY_SUMMARY   → long‑form document summaries
    - GROQ_API_KEY_QA        → interactive Q‑and‑A

Both keys can belong to different Groq accounts to avoid sharing rate‑limits.
Falls back gracefully if you supply only one key.
"""

import os, requests, textwrap

# --------------------------------------------------------------------- #
#  🔑  Load keys (each env var optional, but at least one must exist)
# --------------------------------------------------------------------- #
SUMMARY_KEY = os.getenv("GROQ_API_KEY_SUMMARY")  # summarisation
QA_KEY      = os.getenv("GROQ_API_KEY_QA")       # chat / Q‑A

# Fallback → if only one key provided, use it for both use‑cases
if not SUMMARY_KEY and not QA_KEY:
    raise RuntimeError(
        "🚨 Set at least one of GROQ_API_KEY_SUMMARY or GROQ_API_KEY_QA in your .env"
    )
if not SUMMARY_KEY:
    SUMMARY_KEY = QA_KEY
if not QA_KEY:
    QA_KEY = SUMMARY_KEY

GROQ_URL  = "https://api.groq.com/openai/v1/chat/completions"
MODEL_ID  = "llama3-70b-8192"          # free tier model
TIMEOUT_S = 30


# --------------------------------------------------------------------- #
#  🛠️  Low‑level helper
# --------------------------------------------------------------------- #
def _chat(key: str, system: str, user: str, *, temperature: float = 0.3) -> str:
    """Fire a Groq chat completion and return the assistant content."""
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL_ID,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }

    resp = requests.post(GROQ_URL, json=body, headers=headers, timeout=TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# --------------------------------------------------------------------- #
#  📄  1. Document‑level summary (used in /upload)
# --------------------------------------------------------------------- #
def summarize_document(full_text: str) -> str:
    """Return a concise professional summary of an entire contract / judgement."""
    system = "You are a senior legal analyst. Summarise the document for a busy lawyer."
    user   = textwrap.shorten(full_text, width=15_000, placeholder=" …")  # keep <16k
    try:
        return _chat(SUMMARY_KEY, system, user, temperature=0.25)
    except requests.HTTPError as e:
        raise RuntimeError(f"Groq summary failed: {e}") from e


# --------------------------------------------------------------------- #
#  ❓  2. Answer a question given extracted context chunks
# --------------------------------------------------------------------- #
def answer_question(question: str, context: str) -> str:
    """
    Answer using ONLY the provided context.
    Returns Markdown with short citations – or 'NOT FOUND' if info missing.
    """
    system = (
        "You are a legal assistant. Answer only from the context, "
        "use markdown headings/bullets and cite clause titles in brackets. "
        "Reply 'NOT FOUND' if context lacks the answer."
    )
    user = (
        f"### Question:\n{question}\n\n"
        f"### Context:\n{context}\n\n"
        "### Respond:"
    )
    try:
        return _chat(QA_KEY, system, user, temperature=0.3)
    except requests.HTTPError as e:
        raise RuntimeError(f"Groq QA failed: {e}") from e
