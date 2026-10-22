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
MODEL_ID = os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-120b")
TIMEOUT_S = 30

CANDIDATE_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "allam-2-7b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

# --------------------------------------------------------------------- #
#  🛠️  Low‑level helper
# --------------------------------------------------------------------- #
def _chat(key: str, system: str, user: str, *, temperature: float = 0.3) -> str:
    """Fire a Groq chat completion with automatic model fallback and retry."""
    import time

    # Build prioritized candidate list with active env var first
    models_to_try = []
    preferred = os.getenv("GROQ_MODEL_ID")
    if preferred:
        models_to_try.append(preferred)
    for m in CANDIDATE_MODELS:
        if m not in models_to_try:
            models_to_try.append(m)

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    last_error = None
    for model_name in models_to_try:
        body = {
            "model": model_name,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.post(GROQ_URL, json=body, headers=headers, timeout=TIMEOUT_S)
            except Exception as e:
                last_error = e
                break

            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
                try:
                    wait_time = float(retry_after) if retry_after else 5.0 * (attempt + 1)
                except ValueError:
                    wait_time = 5.0 * (attempt + 1)
                time.sleep(wait_time)
                continue

            if resp.status_code == 404 or "does not exist" in resp.text:
                # Model not available on this API key, try next candidate
                last_error = RuntimeError(resp.text)
                break

            try:
                resp.raise_for_status()
                # Remember this working model for future calls
                os.environ["GROQ_MODEL_ID"] = model_name
                return resp.json()["choices"][0]["message"]["content"].strip()
            except requests.HTTPError as e:
                try:
                    err_json = resp.json()
                    err_msg = err_json.get("error", {}).get("message", resp.text)
                except Exception:
                    err_msg = resp.text
                last_error = RuntimeError(f"Groq API Error: {err_msg}")
                break

    raise RuntimeError(f"🚨 All Groq model candidates failed. Last error: {str(last_error)}")


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
    user = (
        f"### Question:\n{question}\n\n"
        f"### Context:\n{context}\n\n"
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


def extract_metadata_with_groq(text: str) -> dict:
    """Extract metadata (doc_type, parties, effective_date, governing_law) using Groq LLM."""
    import json
    try:
        summary_key, _ = _get_keys()
    except Exception:
        # Fallback if keys are not set/configured
        return {}

    # Keep only the first 2500 characters for metadata extraction (usually front page has it)
    snippet = text[:2500]
    system = (
        "You are a helpful legal assistant. Extract metadata from the contract text.\n"
        "Respond ONLY with a JSON object containing keys: doc_type, parties, effective_date, governing_law.\n"
        "parties must be a list of strings representing the names of the parties involved.\n"
        "effective_date must be a string (e.g. YYYY-MM-DD or readable text) or null.\n"
        "governing_law must be a string representing the jurisdiction or null.\n"
        "doc_type must be the type of document (e.g. 'Lease Agreement', 'NDA', 'Service Agreement') or null.\n"
        "DO NOT write any explanation, just the raw JSON."
    )
    user = f"Contract Snippet:\n{snippet}"
    try:
        response_text = _chat(summary_key, system, user, temperature=0.1)
        # Try to find JSON block if the model wrapped it in markdown code block
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        data = json.loads(response_text.strip())
        return {
            "doc_type": data.get("doc_type"),
            "parties": data.get("parties"),
            "effective_date": data.get("effective_date"),
            "governing_law": data.get("governing_law")
        }
    except Exception as e:
        # Log to console
        print(f"⚠️ Groq metadata extraction failed or timed out: {str(e)}. Falling back to regex.")
        return {}


def verify_citations(answer: str, top_chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Perform a post-hoc verification check over the Q&A answer:
    1. Extract all citations in bracket formats (e.g. [Clause Name] or 【Clause Name】).
    2. Check if every cited clause exists in the retrieved top_chunks metadata.
    3. Check lexical and semantic overlap between the cited sentences and the actual retrieved chunks.
    4. Categorize validation status as: supported / partially supported / unsupported / citation error.
    
    Returns:
        (validation_status, citation_validation_details)
    """
    import re
    import numpy as np
    from backend.nlp_modules.embedder import embed_chunks

    STOPWORDS = {"the", "is", "at", "which", "on", "and", "a", "an", "of", "to", "in", "for", "with", "that", "this", "it", "by", "from", "as", "shall", "be", "are", "under", "our"}

    # 1. Check if the answer is "NOT FOUND"
    clean_ans = answer.strip().upper()
    if "NOT FOUND" in clean_ans or clean_ans == "NOT FOUND":
        return "supported", []

    # 2. Extract citations from response
    citations = re.findall(r'[\[【]([^\]】]+)[\]】]', answer)
    
    if not citations:
        # No citations provided, but the LLM claims to have answered the question.
        return "unsupported", []

    # Map chunks by clause title for easy retrieval
    chunks_map = {}
    for chunk in top_chunks:
        clause_name = chunk.get("meta", {}).get("clause", "")
        if clause_name:
            chunks_map[clause_name.strip().lower()] = chunk

    # Split answer into sentences and merge trailing citation-only elements back
    raw_sentences = re.split(r'(?<=[.!?])\s+', answer)
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue
        if s.startswith(('[', '【')) and s.endswith((']', '】')) and sentences:
            sentences[-1] = sentences[-1] + " " + s
        else:
            sentences.append(s)
    
    validation_details = []
    has_mismatch = False
    unsupported_count = 0
    partially_supported_count = 0
    supported_count = 0

    for sentence in sentences:
        # Find citations in this specific sentence
        sentence_citations = re.findall(r'[\[【]([^\]】]+)[\]】]', sentence)
        if not sentence_citations:
            continue
        
        # Clean the sentence of the citation brackets for clean semantic/lexical check
        clean_sentence = re.sub(r'[\[【][^\]】]+[\]】]', '', sentence).strip()
        if not clean_sentence:
            continue

        for cited_clause in sentence_citations:
            cited_clause_clean = cited_clause.strip()
            cited_clause_key = cited_clause_clean.lower()
            
            # Match against retrieved chunks (case-insensitive, fallback to partial matching if necessary)
            matched_chunk = None
            if cited_clause_key in chunks_map:
                matched_chunk = chunks_map[cited_clause_key]
            else:
                # Fallback: check if cited clause is a substring of any chunk's clause or vice versa
                for k, chunk in chunks_map.items():
                    if cited_clause_key in k or k in cited_clause_key:
                        matched_chunk = chunk
                        break
            
            if not matched_chunk:
                # Citation doesn't exist in retrieved chunks!
                has_mismatch = True
                validation_details.append({
                    "sentence": sentence,
                    "cited_clause": cited_clause_clean,
                    "status": "citation error",
                    "lexical_overlap": 0.0,
                    "semantic_similarity": 0.0,
                    "reason": "Cited clause was not found in the retrieved context chunks."
                })
                continue
            
            chunk_text = matched_chunk.get("text", "")
            
            # Check Lexical Overlap
            words1 = set(re.findall(r'\w+', clean_sentence.lower())) - STOPWORDS
            words2 = set(re.findall(r'\w+', chunk_text.lower())) - STOPWORDS
            lex_overlap = len(words1.intersection(words2)) / len(words1) if words1 else 1.0
            
            # Check Semantic Similarity (Cosine Similarity of embeddings)
            try:
                emb_sentence = embed_chunks([clean_sentence])[0]
                emb_chunk = embed_chunks([chunk_text])[0]
                sem_similarity = float(np.dot(emb_sentence, emb_chunk))
            except Exception:
                sem_similarity = 0.0

            # Determine sentence support status
            if sem_similarity >= 0.55 or lex_overlap >= 0.4:
                status = "supported"
                supported_count += 1
            elif sem_similarity >= 0.4 or lex_overlap >= 0.2:
                status = "partially supported"
                partially_supported_count += 1
            else:
                status = "unsupported"
                unsupported_count += 1

            validation_details.append({
                "sentence": sentence,
                "cited_clause": cited_clause_clean,
                "status": status,
                "lexical_overlap": float(lex_overlap),
                "semantic_similarity": float(sem_similarity)
            })

    # 3. Aggregate overall taxonomy label
    if has_mismatch:
        overall_status = "citation error"
    elif unsupported_count > 0:
        overall_status = "unsupported"
    elif partially_supported_count > 0:
        overall_status = "partially supported"
    elif supported_count > 0:
        overall_status = "supported"
    else:
        overall_status = "unsupported"

    return overall_status, validation_details



