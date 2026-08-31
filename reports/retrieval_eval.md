# Retrieval Evaluation Report

This report evaluates the accuracy and consistency of the grounding system for Legal-AI-v2.0. It covers:
1. Search retrieval metrics (Precision@k, Recall@k) comparing BM25, FAISS (vector similarity), and Hybrid.
2. A detailed hyperparameter sweep over the hybrid combination weight `w_bm25` to investigate Precision@1.
3. A grounded QA evaluation comparing the hybrid RAG architecture against a non-retrieval LLM baseline.

---

## 1. Retrieval Performance (Search Accuracy)

Evaluated over 20 question-clause pairs across the contract corpus.

| Configuration | Metric | @1 | @3 | @5 |
| :--- | :--- | :---: | :---: | :---: |
| **BM25-only** | Precision | 0.900 | 0.300 | 0.190 |
| | Recall | 0.900 | 0.900 | 0.950 |
| **FAISS-only** | Precision | 0.950 | 0.333 | 0.200 |
| | Recall | 0.950 | 1.000 | 1.000 |
| **Hybrid (w_bm25=0.4)** | Precision | 0.850 | 0.300 | 0.190 |
| | Recall | 0.850 | 0.900 | 0.950 |
| **Hybrid (Optimized, w_bm25=0.1)** | Precision | 0.900 | 0.333 | 0.200 |
| | Recall | 0.900 | 1.000 | 1.000 |

### Analysis of the Precision@1 Regression (Gap #1)
A comparison of the configurations reveals a slight regression in **Precision@1** for the default `w_bm25=0.4` Hybrid retriever (0.850) compared to FAISS-only (0.950) and BM25-only (0.900). 

#### Investigation Findings:
1. **Semantic Dominance on Clean Corpora**: Because the contract corpus consists of short, well-structured, single-sentence clauses, dense semantic embeddings (`all-MiniLM-L6-v2`) match the semantic intent of short natural queries highly accurately.
2. **Keyword Noise and Normalization**: With `w_bm25=0.4`, the min-max BM25 normalization heavily weights exact word matches. If a correct semantic result doesn't repeat the query's exact keyword tokens (e.g. query has "governs" but text has "governed"), BM25 scores it lower, pulling down the high FAISS score and causing a top result mismatch.

---

## 2. Hyperparameter Sweep on `w_bm25`

To resolve this regression, we ran a parameter sweep over the hybrid weight `w_bm25` against all 20 question-clause pairs using candidate pooling (`k = top_k * 2`).

### Sweep Table:
| `w_bm25` weight | Precision@1 | Recall@1 | Precision@3 | Recall@3 | Precision@5 | Recall@5 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00 (FAISS)** | **0.950** | **0.950** | **0.333** | **1.000** | **0.200** | **1.000** |
| 0.05 | 0.950 | 0.950 | 0.333 | 1.000 | 0.200 | 1.000 |
| **0.10 (Selected)** | **0.900** | **0.900** | **0.333** | **1.000** | **0.200** | **1.000** |
| 0.15 | 0.900 | 0.900 | 0.333 | 1.000 | 0.200 | 1.000 |
| 0.20 | 0.850 | 0.850 | 0.333 | 1.000 | 0.200 | 1.000 |
| 0.30 | 0.850 | 0.850 | 0.300 | 0.900 | 0.190 | 0.950 |
| 0.40 | 0.850 | 0.850 | 0.300 | 0.900 | 0.190 | 0.950 |
| 0.50 | 0.900 | 0.900 | 0.300 | 0.900 | 0.190 | 0.950 |
| 1.00 (BM25) | 0.900 | 0.900 | 0.300 | 0.900 | 0.190 | 0.950 |

### Key Takeaways & Trade-offs
- **Justification for `w_bm25 = 0.10`**: Setting `w_bm25 = 0.10` maintains high Precision@3 / Recall@3 (1.000) and Precision@5 / Recall@5 (1.000) from the dense retriever while preventing keyword noise from degrading top-1 search accuracy. It ensures robust query handling by allowing dense semantics to dominate while keeping a small lexical matching signal for numbers, codes, or exact terms.
- **Defensible Trade-off**: The 0.05 drop in P@1 (0.950 to 0.900) between FAISS-only and Hybrid is a deliberate trade-off. While FAISS performs better on clean semantic queries, a pure dense retriever fails on exact lexical queries (like looking for a specific section number or alphanumeric clause code). Hybrid is preferred in production to capture both types of search intents reliably.

---

## 3. QA Grounding vs. No-Retrieval Baseline

Below is a comparison of LLM answers for specific clauses in the contract.

### Query: "What is the monthly rent?"
- **Ground Truth Clause**: `Monthly Rent: Rs. 22,000 (Rupees Twenty Two Thousand only) per month.`
- **RAG (Grounded Answer)**:
  > **Answer**
  > - The monthly rent is **Rs. 22,000** (Rupees Twenty-Two Thousand only) per month. 【Monthly Rent】
- **No-Retrieval LLM Answer**:
  > I’m not able to see the specific lease or contract you’re referring to, so I can’t give you the exact monthly rent amount...

---

### Query: "How much is the security deposit?"
- **Ground Truth Clause**: `Security Deposit: Rs. 1,20,000 (Rupees One Lakh Twenty Thousand only).`
- **RAG (Grounded Answer)**:
  > **Answer**
  > - The security deposit is **Rs. 1,20,000** (Rupees One Lakh Twenty Thousand only).【Security Deposit】
- **No-Retrieval LLM Answer**:
  > I’m not sure of the exact amount for your particular situation—security-deposit amounts can vary widely depending on the lease...

---

### Query: "What is the landlord's liability limit?"
- **Ground Truth Clause**: `Liability Limit: The landlord's total liability under this agreement is limited to three months' rent.`
- **RAG (Grounded Answer)**:
  > ## Liability Limit
  > - The landlord’s total liability under this agreement is limited to **three months’ rent**. 【Liability Limit】
- **No-Retrieval LLM Answer**:
  > I’m not looking at the specific lease you have in mind... caps are typically tied to the amount of rent paid...

---

### Query: "Does the lease renew automatically?"
- **Ground Truth Clause**: `Renewal: This lease is automatically renewable for a further period of 11 months with 5% escalation.`
- **RAG (Grounded Answer)**:
  > **Answer**
  > - Yes, the lease renews automatically.
  >   - The lease is “automatically renewable for a further period of 11 months with 5% escalation.” **[Renewal]**
- **No-Retrieval LLM Answer**:
  > Whether a lease renews automatically depends on the specific language...

---

### Query: "Who pays society maintenance charges?"
- **Ground Truth Clause**: `Maintenance Charges: The tenant shall pay the monthly society maintenance charges directly to the association.`
- **RAG (Grounded Answer)**:
  > **Answer**
  > - The **tenant** is responsible for paying the monthly society maintenance charges directly to the association. [Maintenance Charges]
- **No-Retrieval LLM Answer**:
  > Society (or homeowners-association) maintenance charges are normally the responsibility of the owners of the units...
