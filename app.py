import os
import re
from pathlib import Path
from typing import List, Tuple, Optional
import gradio as gr

from backend.nlp_modules import (
    universal_parser,
    embedder,
    vector_store,
    summarizer,
    risk_rules,
)
from backend.nlp_modules.schema import DocumentRecord

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, **kwargs):
            if func is not None:
                return func
            return lambda f: f

# Global or session-based VectorDB instance
DB_INSTANCE = vector_store.VectorDB(dim=384)
PROCESSED_DOCS = set()

@spaces.GPU
def analyze_document(file_obj, api_key_input: Optional[str] = None):
    """
    Parses the uploaded legal document, extracts metadata,
    computes risk flags, validates poisoning anomalies,
    indexes into FAISS/BM25, and generates an executive summary.
    """
    if file_obj is None:
        return (
            "⚠️ Please upload a valid document file.",
            "",
            "",
            "",
            "No document loaded.",
        )

    # Allow setting Groq API key via UI if not present in environment
    if api_key_input and api_key_input.strip():
        os.environ["GROQ_API_KEY_SUMMARY"] = api_key_input.strip()
        os.environ["GROQ_API_KEY_QA"] = api_key_input.strip()

    file_path = file_obj.name
    filename = Path(file_path).name

    try:
        # 1. Universal Extraction
        text = universal_parser.extract_text(file_path)
        if not text or len(text.strip()) < 20:
            return (
                f"❌ Unable to extract readable text from '{filename}'. Please verify file format.",
                "",
                "",
                "",
                "Extraction failed.",
            )

        # 2. Risk Detection
        risks = risk_rules.detect_risks(text)

        # 3. Metadata Extraction
        doc_meta = universal_parser.extract_metadata(text)
        source_format = Path(file_path).suffix.upper().replace(".", "") or "TXT"

        # 4. Chunking
        chunks, basic_metas = universal_parser.split_into_chunks(text)
        if not chunks:
            return (
                f"❌ Document '{filename}' did not yield any indexable clauses.",
                "",
                "",
                "",
                "No chunks created.",
            )

        # 5. Schema Validation
        validated_metas = []
        for i, chunk in enumerate(chunks):
            chunk_risks = risk_rules.detect_risks(chunk)
            try:
                record = DocumentRecord(
                    doc_id=filename,
                    doc_type=doc_meta.get("doc_type", "Agreement"),
                    risk_flags=chunk_risks,
                    parties=doc_meta.get("parties", []),
                    effective_date=doc_meta.get("effective_date"),
                    governing_law=doc_meta.get("governing_law"),
                    source_format=source_format,
                    clause_index=i,
                )
                meta_dict = record.model_dump()
                meta_dict["source"] = filename
                meta_dict["risks"] = risks
                meta_dict["clause"] = basic_metas[i].get("clause", f"Clause {i+1}")
                validated_metas.append(meta_dict)
            except Exception:
                meta_dict = {
                    "source": filename,
                    "risks": risks,
                    "clause": basic_metas[i].get("clause", f"Clause {i+1}"),
                    "doc_id": filename,
                    "doc_type": doc_meta.get("doc_type", "Agreement"),
                    "risk_flags": chunk_risks,
                }
                validated_metas.append(meta_dict)

        # 6. Dense Embeddings
        embeddings = embedder.embed_chunks(chunks)

        # 7. Ingestion Poisoning Outlier Detection
        poisoning_warnings = DB_INSTANCE.detect_poisoning(embeddings)
        flagged_indices = []
        for warn in poisoning_warnings:
            m = re.search(r"Chunk (\d+)", warn)
            if m:
                flagged_indices.append(int(m.group(1)))

        for i in range(len(validated_metas)):
            validated_metas[i]["potential_poisoning"] = (i in flagged_indices)

        # 8. Add to In-Memory Vector Store
        DB_INSTANCE.add(embeddings, chunks, validated_metas)
        PROCESSED_DOCS.add(filename)

        # 9. Generate Executive Summary
        ctx_for_llm = "\n".join(chunks[:3])
        summary = ""
        try:
            summary = summarizer.summarize_with_groq(
                "Provide a concise, professional executive summary of this legal agreement highlighting core rights, obligations, and term.",
                ctx_for_llm,
            )
        except Exception as e:
            summary = f"*(LLM Summary unavailable: {str(e)}. Please configure your GROQ_API_KEY secret in Space Settings)*"

        # Format Status Badge
        status_md = f"### ✅ Successfully Indexed: `{filename}`\n"
        status_md += f"- **Clauses / Chunks**: {len(chunks)}\n"
        status_md += f"- **Format Detected**: {source_format}\n"
        status_md += f"- **Total Active Documents in Memory**: {len(PROCESSED_DOCS)}"

        # Format Metadata Table
        parties_str = ", ".join(doc_meta.get("parties", [])) or "Not detected"
        gov_law = doc_meta.get("governing_law") or "Not specified"
        eff_date = doc_meta.get("effective_date") or "Not specified"
        doc_type = doc_meta.get("doc_type") or "Agreement"

        meta_md = f"""
| Metadata Field | Extracted Value |
| :--- | :--- |
| **Document Type** | `{doc_type}` |
| **Identified Parties** | {parties_str} |
| **Governing Law** | {gov_law} |
| **Effective Date** | {eff_date} |
"""

        # Format Risks
        if risks:
            risk_md = "### ⚠️ Potential Contract Risks Detected\n"
            for r in risks:
                risk_md += f"- 🔴 **{r}**\n"
        else:
            risk_md = "### 🟢 Risk Engine Assessment\nNo standard critical risk flags triggered (e.g. no unlimited liability or auto-renewal red flags detected)."

        # Poisoning / Security Status
        if poisoning_warnings:
            sec_md = f"### ⚠️ Adversarial Poisoning Notice\nDetected {len(poisoning_warnings)} outlier chunk(s) during ingestion vector distribution check."
        else:
            sec_md = "### 🛡️ Ingestion Integrity\nAll vector distributions passed embedding anomaly and distribution checks."

        return status_md, meta_md, risk_md, sec_md, summary

    except Exception as exc:
        return (
            f"❌ Error during document processing: {str(exc)}",
            "",
            "",
            "",
            "Processing error.",
        )


@spaces.GPU
def chat_query(query: str, history: List[Tuple[str, str]], api_key_input: Optional[str] = None):
    """
    Answers questions using hybrid retrieval (FAISS + BM25) and Groq LLM synthesis,
    with claim-evidence citation verification.
    """
    if not query.strip():
        return history, "Please enter a valid legal question.", ""

    if len(DB_INSTANCE.text_chunks) == 0:
        bot_response = "⚠️ No contracts indexed yet. Please upload and analyze a contract in the 'Document Ingestion' tab first."
        history.append((query, bot_response))
        return history, "No documents indexed.", ""

    # Allow setting Groq API key via UI if not present in environment
    if api_key_input and api_key_input.strip():
        os.environ["GROQ_API_KEY_SUMMARY"] = api_key_input.strip()
        os.environ["GROQ_API_KEY_QA"] = api_key_input.strip()

    try:
        q_emb = embedder.embed_chunks([query])[0]
        if hasattr(DB_INSTANCE, "hybrid_search"):
            raw = DB_INSTANCE.hybrid_search(query, q_emb, top_k=15)
        else:
            raw = DB_INSTANCE.search(q_emb, top_k=15)

        # De-duplicate top chunks
        seen, top = set(), []
        for r in raw:
            key = (r["meta"].get("clause", ""), r["meta"].get("source", ""), r["text"][:100])
            if key not in seen:
                seen.add(key)
                top.append(r)
            if len(top) == 5:
                break

        if not top:
            bot_response = "I couldn't locate any relevant clauses in the loaded document(s) matching your inquiry."
            history.append((query, bot_response))
            return history, "No matching clauses found.", ""

        context = "\n\n".join(r["text"] for r in top)
        prompt = f"Answer the user question accurately based on the legal context provided. Reference specific clause titles or terms where applicable.\n\nQuestion: {query}"

        answer = summarizer.summarize_with_groq(prompt, context)

        # Verify Citations
        val_status, val_details = summarizer.verify_citations(answer, top)

        status_badge = f"**Grounding Verification**: `{val_status.upper()}`"
        if val_details.get("avg_overlap"):
            status_badge += f" | Grounding Overlap: `{val_details['avg_overlap']:.1%}`"

        # Format Citations Block
        citations_md = "### 📑 Cited Source Clauses\n"
        for i, r in enumerate(top, 1):
            clause_name = r["meta"].get("clause", f"Clause {i}")
            source = r["meta"].get("source", "Document")
            snippet = r["text"][:220].replace("\n", " ") + "..."
            citations_md += f"**[{i}] {clause_name}** (*{source}*)\n> {snippet}\n\n"

        history.append((query, answer))
        return history, status_badge, citations_md

    except Exception as e:
        bot_response = f"⚠️ Query error: {str(e)}. (Ensure GROQ API keys are configured in Space Settings or in the Key Configuration panel)."
        history.append((query, bot_response))
        return history, "Error occurred", ""


# Build Gradio UI
CUSTOM_CSS = """
.container { max-width: 1200px; margin: auto; }
.card { border-radius: 10px; padding: 15px; margin-bottom: 15px; }
"""

with gr.Blocks(title="Comprehensive Legal AI Platform", css=CUSTOM_CSS) as demo:
    gr.Markdown(
        """
        # ⚖️ Comprehensive Legal AI Platform
        **Enterprise Contract Analysis, Hybrid RAG, Risk Engine & Cited Grounding**
        """
    )

    with gr.Accordion("⚙️ Groq API Configuration (Optional if already set in Space Secrets)", open=False):
        api_key_box = gr.Textbox(
            label="Groq API Key (gsk_...)",
            placeholder="Enter Groq API Key if not set in Space Secrets",
            type="password",
        )
        gr.Markdown("*Tip: You can add `GROQ_API_KEY_SUMMARY` and `GROQ_API_KEY_QA` in your Space's Settings > Variables and secrets to avoid typing this.*")

    with gr.Tabs():
        # TAB 1: INGESTION & RISK ASSESSMENT
        with gr.Tab("📄 Document Analysis & Risk Engine"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(
                        label="Upload Legal Agreement (PDF, DOCX, TXT, HTML)",
                        file_types=[".pdf", ".docx", ".txt", ".html", ".htm", ".adoc"],
                    )
                    analyze_btn = gr.Button("🔍 Analyze & Index Contract", variant="primary")
                    status_output = gr.Markdown("Ready to ingest document.")
                    poisoning_output = gr.Markdown("")

                with gr.Column(scale=2):
                    summary_output = gr.Markdown("### 📋 Executive Summary\n*Upload a document to generate a summary.*")
                    meta_output = gr.Markdown("")
                    risk_output = gr.Markdown("")

            analyze_btn.click(
                fn=analyze_document,
                inputs=[file_input, api_key_box],
                outputs=[status_output, meta_output, risk_output, poisoning_output, summary_output],
            )

        # TAB 2: INTERACTIVE GROUNDED Q&A
        with gr.Tab("💬 Interactive Cited Q&A"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="Legal Consultation History", height=450)
                    with gr.Row():
                        query_input = gr.Textbox(
                            label="Ask about the contract (e.g. 'What is the termination notice period?' or 'Who is liable for damages?')",
                            placeholder="Type your legal query here...",
                            scale=4,
                        )
                        ask_btn = gr.Button("Ask", variant="primary", scale=1)

                with gr.Column(scale=1):
                    grounding_badge = gr.Markdown("### 🛡️ Grounding Status\n*Ask a question to see verification.*")
                    citations_box = gr.Markdown("### 📑 Citations\n*Relevant contract clauses will appear here.*")

            ask_btn.click(
                fn=chat_query,
                inputs=[query_input, chatbot, api_key_box],
                outputs=[chatbot, grounding_badge, citations_box],
            )
            query_input.submit(
                fn=chat_query,
                inputs=[query_input, chatbot, api_key_box],
                outputs=[chatbot, grounding_badge, citations_box],
            )

        # TAB 3: SYSTEM ARCHITECTURE
        with gr.Tab("ℹ️ Architecture & Methodology"):
            gr.Markdown(
                """
                ### Core System Architecture
                1. **Universal Parser**: Extracts text streams from native PDFs (PyMuPDF), scanned image PDFs (Tesseract OCR + OpenCV), Word `.docx`, and HTML.
                2. **Rule-Based Risk Engine**: Evaluates 8 deterministic contract risk patterns (unlimited liability, unilateral termination, missing governing law, auto-renewals).
                3. **Schema Validation**: Enforces structured Pydantic schemas on all document chunks and metadata.
                4. **Hybrid Retrieval**: Combines dense semantic representations (FAISS `IndexFlatL2` + `all-MiniLM-L6-v2`) with sparse keyword matching (`rank-bm25`).
                5. **Post-Hoc Claim Grounding**: Validates generated answers against retrieved clause text using n-gram overlap and citation attribution.
                """
            )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="indigo"))
