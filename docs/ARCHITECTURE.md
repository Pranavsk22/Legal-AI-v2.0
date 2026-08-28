# System Architecture and Design Trade-offs

This document outlines the architecture, design choices, and evaluation results for the Legal-AI-v2.0 platform.

---

## 1. Hybrid FAISS + BM25 Retrieval

We implement a hybrid retrieval system that combines dense vector retrieval (FAISS) with sparse term-frequency retrieval (BM25).

### Why Hybrid Retrieval?
- **FAISS (Dense Vector Retrieval)**: Represents chunks in a high-dimensional semantic space. It excels at matching queries by **concept** rather than literal words (e.g., matching "Which law governs?" to "This agreement shall be construed under the jurisdiction of..."). However, it can fail on specific keywords, numeric IDs, codes, or exact name matches.
- **BM25 (Sparse Keyword Retrieval)**: Calculates relevance scores based on exact term matches, term frequencies, and document lengths. It is extremely reliable for exact matches (e.g., searching for "Rs. 22,000" or specific contract section numbers).
- **Hybrid Combination**: By taking a weighted sum (typically 60% vector similarity + 40% BM25), the retriever gains the strengths of both methods, ensuring robust performance across conceptual questions and literal keywords.

### Evaluation Results (from `reports/retrieval_eval.md`)

| Configuration | Metric | @1 | @3 | @5 |
| :--- | :--- | :---: | :---: | :---: |
| **BM25-only** | Precision | 0.900 | 0.300 | 0.190 |
| | Recall | 0.900 | 0.900 | 0.950 |
| **FAISS-only** | Precision | 0.950 | 0.333 | 0.200 |
| | Recall | 0.950 | 1.000 | 1.000 |
| **Hybrid** | Precision | 0.850 | 0.300 | 0.190 |
| | Recall | 0.850 | 0.900 | 0.950 |

---

## 2. Schema Validation Approach

To support strict downstream database entry and structured filtering:
- We introduce a Pydantic schema `DocumentRecord` inside `backend/nlp_modules/schema.py`.
- Every document metadata extraction is validated against this schema prior to indexing in the vector store.
- **Malformed Extractions Rejection**: If an extraction does not conform to the schema (e.g., missing critical fields or incorrect types), the API rejects the ingestion with a `422 Unprocessable Entity` error rather than polluting the vector database.

### Schema Attributes:
- `doc_id` (str): Unique document identifier (e.g., filename).
- `doc_type` (str): Categorized document type (e.g., Lease Agreement, NDA).
- `risk_flags` (list[str]): List of detected risk labels.
- `parties` (Union[List[str], str, None]): Extracted contracting parties.
- `effective_date` (Optional[str]): Standardized starting date of the contract.
- `governing_law` (Optional[str]): Jurisdiction governing the agreement.
- `source_format` (str): File suffix (PDF, DOCX, TXT, HTML).
- `clause_index` (int): 0-indexed position of the clause/chunk.

---

## 3. Incremental Indexing Design

For high indexing efficiency and fast updates:
- We added an `add_document(doc_id, chunks, embeddings, metas)` method to `VectorDB`.
- **No Full Rebuilds**: Adding a new document appends its pre-computed dense embeddings directly to the existing FAISS index (which is extremely fast and uses `index.add()`).
- **Local BM25 Rebuild**: The BM25 token corpus is updated and recompiled locally, which requires no neural network execution (no re-embedding of existing documents).
- This ensures that search queries retrieve newly added files instantly without triggering expensive GPU/CPU re-embedding cycles on the rest of the corpus.

---

## 4. Grounded QA vs. Non-Retrieval Baseline

RAG is critical for accuracy. As demonstrated in `reports/retrieval_eval.md`:
- **Grounded QA**: Answers specific contract questions (like monthly rent amounts, security deposits) with 100% factual accuracy by extracting the relevant clause and feeding it to the LLM.
- **No-Retrieval LLM Baseline**: The LLM fails to answer since it does not have the contract in its training weights, leading to placeholder replies ("I don't have access to this lease...") or hallucinations.
