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
from dotenv import load_dotenv
load_dotenv()

# --------------------------------------------------------------------- #
#  🔑  Load keys (each env var optional, but at least one must exist)
# --------------------------------------------------------------------- #
def _get_keys():
    summary_key = os.getenv("GROQ_API_KEY_SUMMARY")
    qa_key = os.getenv("GROQ_API_KEY_QA")
    if not summary_key and not qa_key:
        raise RuntimeError(
            "🚨 Set at least one of GROQ_API_KEY_SUMMARY or GROQ_API_KEY_QA in your .env"
        )
    if not summary_key:
        summary_key = qa_key
    if not qa_key:
        qa_key = summary_key
    return summary_key, qa_key

GROQ_URL  = "https://api.groq.com/openai/v1/chat/completions"
MODEL_ID  = os.getenv("GROQ_MODEL_ID", "groq/compound-mini") # active model ID from environment
TIMEOUT_S = 30


# --------------------------------------------------------------------- #
#  🛠️  Low‑level helper
# --------------------------------------------------------------------- #
def _chat(key: str, system: str, user: str, *, temperature: float = 0.3) -> str:
    """Fire a Groq chat completion and return the assistant content."""
    import time
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

    max_retries = 4
    for attempt in range(max_retries):
        resp = requests.post(GROQ_URL, json=body, headers=headers, timeout=TIMEOUT_S)
        
        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
            try:
                wait_time = float(retry_after) if retry_after else 10.0 * (attempt + 1)
            except ValueError:
                wait_time = 10.0 * (attempt + 1)
            
            # Print to stdout/stderr so developers and backend logs show rate limits
            print(f"⚠️  [Rate Limit] Groq 429 received. Retrying in {wait_time:.2f}s (Attempt {attempt+1}/{max_retries})...")
            time.sleep(wait_time)
            continue
            
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            try:
                err_json = resp.json()
                err_msg = err_json.get("error", {}).get("message", resp.text)
            except Exception:
                err_msg = resp.text
            raise RuntimeError(f"Groq API Error: {err_msg}") from e
        return resp.json()["choices"][0]["message"]["content"].strip()
        
    raise RuntimeError("🚨 Groq API Error: Maximum retries exceeded due to rate limiting (429).")


# --------------------------------------------------------------------- #
#  📄  1. Document‑level summary (used in /upload)
# --------------------------------------------------------------------- #
def summarize_document(full_text: str) -> str:
    """Return a concise professional summary of an entire contract / judgement."""
    summary_key, _ = _get_keys()
    system = "You are a senior legal analyst. Summarise the document for a busy lawyer."
    user   = textwrap.shorten(full_text, width=3_000, placeholder=" …")  # keep <3k
    try:
        return _chat(summary_key, system, user, temperature=0.25)
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
    _, qa_key = _get_keys()
    system = (
        "You are a legal assistant. Answer only from the context, "
        "use markdown headings/bullets and cite clause titles in brackets. "
        "Reply 'NOT FOUND' if context lacks the answer."
    )
    # Truncate context to ~3KB as a safe guard-rail against HTTP 413 Payload Too Large
    context_short = textwrap.shorten(context, width=3_000, placeholder=" …")
    user = (
        f"### Question:\n{question}\n\n"
        f"### Context:\n{context_short}\n\n"
        "### Respond:"
    )
    try:
        return _chat(qa_key, system, user, temperature=0.3)
    except requests.HTTPError as e:
        raise RuntimeError(f"Groq QA failed: {e}") from e


# --------------------------------------------------------------------- #
#  🔑  3. Unified helper function (used across multiple entrypoints)
# --------------------------------------------------------------------- #
def summarize_with_groq(prompt_or_context: str, context: str = None) -> str:
    """
    Unified entrypoint that handles:
      1. Single-parameter: summarize_with_groq(context)
         Runs document summary with SUMMARY_KEY.
      2. Two-parameter: summarize_with_groq(prompt, context)
         Runs question answering with QA_KEY.
    """
    if context is None:
        return summarize_document(prompt_or_context)
    else:
        return answer_question(prompt_or_context, context)

