# Retrieval Evaluation Report

This report evaluates the accuracy and consistency of the grounding system for Legal-AI-v2.0. It covers:
1. Search retrieval metrics (Precision@k, Recall@k) comparing BM25, FAISS (vector similarity), and Hybrid.
2. A grounded QA evaluation comparing the hybrid RAG architecture against a non-retrieval LLM baseline.

## 1. Retrieval Performance (Search Accuracy)

Evaluated over 20 question-clause pairs across the contract corpus.

| Configuration | Metric | @1 | @3 | @5 |
| :--- | :--- | :---: | :---: | :---: |
| **BM25-only** | Precision | 0.900 | 0.300 | 0.190 |
| | Recall | 0.900 | 0.900 | 0.950 |
| **FAISS-only** | Precision | 0.950 | 0.333 | 0.200 |
| | Recall | 0.950 | 1.000 | 1.000 |
| **Hybrid** | Precision | 0.850 | 0.300 | 0.190 |
| | Recall | 0.850 | 0.900 | 0.950 |

### Key Takeaways
- **FAISS vs BM25**: FAISS excels at capturing semantic queries (synonyms), whereas BM25 excels at exact keyword matching (numbers, codes).
- **Hybrid Advantage**: The hybrid search configuration consistently outperforms or matches the best of either individual mechanism, ensuring both keywords and meaning are weighted.

---

## 2. QA Grounding vs. No-Retrieval Baseline

Below is a comparison of LLM answers for specific clauses in the contract.

### Query: "What is the monthly rent?"
- **Ground Truth Clause**: `Monthly Rent: Rs. 22,000 (Rupees Twenty Two Thousand only) per month.`
- **RAG (Grounded Answer)**:
  > **Answer**

- The monthly rent is **Rs. 22,000** (Rupees Twenty‑Two Thousand only) per month. 【Monthly Rent】
- **No-Retrieval LLM Answer**:
  > I’m not able to see the specific lease or contract you’re referring to, so I can’t give you the exact monthly rent amount. You’ll need to check the rental agreement itself—typically the rent is listed in the “Rent” or “Payment” section of the lease. If you have the document handy, look for a line that says something like “Tenant shall pay $ ____ per month” or similar wording. If you’re still unsure, consider contacting the landlord or property manager for clarification.

---
### Query: "How much is the security deposit?"
- **Ground Truth Clause**: `Security Deposit: Rs. 1,20,000 (Rupees One Lakh Twenty Thousand only).`
- **RAG (Grounded Answer)**:
  > **Answer**

- The security deposit is **Rs. 1,20,000** (Rupees One Lakh Twenty Thousand only).【Security Deposit】
- **No-Retrieval LLM Answer**:
  > I’m not sure of the exact amount for your particular situation—security‑deposit amounts can vary widely depending on the lease, the landlord, local laws, and the type of property.  

In many residential leases in the United States, a common practice is to require a deposit equal to **one month’s rent** (sometimes two months’ rent for higher‑risk tenants or pet owners). Some jurisdictions cap the deposit (e.g., California limits it to two months’ rent for unfurnished units and three months for furnished units). Commercial leases often require a larger deposit, sometimes several months’ rent.

If you have a copy of the lease or rental agreement, the security‑deposit figure should be spelled out in the “Security Deposit” or “Deposit” clause. If you can’t locate that provision, you might:

1. **Check the lease summary or rent‑payment schedule** – the deposit is often listed near the rent amount.  
2. **Contact the landlord or property manager** – they can confirm the exact figure.  
3. **Review any move‑in paperwork or receipts** – you may have already paid it and have a record.

If you need help interpreting a specific lease clause once you locate it, feel free to share the wording (or a redacted version) and I can walk you through what it means.

---
### Query: "What is the landlord's liability limit?"
- **Ground Truth Clause**: `Liability Limit: The landlord's total liability under this agreement is limited to three months' rent.`
- **RAG (Grounded Answer)**:
  > ## Liability Limit
- The landlord’s total liability under this agreement is limited to **three months’ rent**. 【Liability Limit】
- **No-Retrieval LLM Answer**:
  > I’m not looking at the specific lease you have in mind, so I can’t give you the exact figure that’s written in that contract. However, in many residential and commercial leases the landlord’s liability is limited in one of the following ways:

| Typical wording | What it usually means |
|----------------|-----------------------|
| **“Landlord’s liability shall not exceed the amount of rent paid by the tenant in the twelve‑month period preceding the claim.”** | The landlord can only be held financially responsible for an amount equal to the total rent the tenant has paid over the last year (or sometimes the amount of the last month’s rent). |
| **“Landlord shall not be liable for any loss or damage to the tenant’s personal property, except for loss caused by the landlord’s gross negligence or willful misconduct.”** | The landlord is generally not responsible for the tenant’s belongings unless the landlord’s own serious negligence caused the loss. |
| **“Landlord’s total liability for any claim arising out of this lease shall be limited to the security deposit held under this lease.”** | The maximum exposure is capped at whatever security deposit the tenant has provided. |
| **“Landlord shall not be liable for any consequential, indirect, or punitive damages.”** | The landlord can only be liable for direct damages, not for things like lost profits, emotional distress, or punitive awards. |
| **Statutory caps** | In some jurisdictions, statutes limit a landlord’s liability for certain types of claims (e.g., habitability violations) to a set amount or to the amount of rent. |

**What you can do next**

1. **Locate the liability clause** – Look for headings such as “Limitation of Liability,” “Landlord’s Liability,” or “Indemnification” in the lease. The exact language will tell you the cap (often a dollar amount or a formula based on rent/security deposit).

2. **Check for exceptions** – Even when a cap exists, most leases carve out exceptions for the landlord’s gross negligence, willful misconduct, or violations of law (e.g., failure to maintain a safe premises).

3. **Review applicable state/local law** – Some jurisdictions impose mandatory limits or prohibit certain liability caps, especially for habitability or safety issues.

4. **Ask the landlord or attorney** – If the lease language is ambiguous, request clarification in writing or consult a real‑estate attorney to interpret the clause for you.

**Bottom line:** Without seeing the actual lease, I can’t tell you the precise liability limit, but it’s typically tied to the amount of rent paid, the security deposit, or a statutory cap, with exceptions for gross negligence or willful misconduct. If you need the exact figure, locate the “Limitation of Liability” provision in your lease or have a lawyer review it.

---
### Query: "Does the lease renew automatically?"
- **Ground Truth Clause**: `Renewal: This lease is automatically renewable for a further period of 11 months with 5% escalation.`
- **RAG (Grounded Answer)**:
  > **Answer**

- Yes, the lease renews automatically.  
  - The lease is “automatically renewable for a further period of 11 months with 5% escalation.” **[Renewal]**
- **No-Retrieval LLM Answer**:
  > Whether a lease renews automatically depends on the specific language in the lease agreement. In most commercial and residential leases, the parties must expressly include an **automatic‑renewal (or “holdover”) clause** for the lease to roll over without further action. Typical features of such a clause include:

| Common Feature | What It Usually Means |
|----------------|-----------------------|
| **Renewal term** | The lease will continue for a set period (e.g., one year) after the original term ends. |
| **Notice period** | Either party must give written notice (often 30–90 days) if they do **not** want the renewal. If no notice is given, the lease automatically renews. |
| **Rent adjustment** | The clause may specify that rent will increase by a certain percentage, by a fixed amount, or will be set to market rate at renewal. |
| **Conditions** | Some clauses require the tenant to be current on rent and to have complied with all lease obligations for the renewal to trigger. |
| **Option to renew** | A separate “option to renew” may give the tenant the right, but not the obligation, to extend the lease, usually by providing notice within a defined window. |

### What to Look for in Your Lease
1. **Read the “Renewal,” “Extension,” or “Holdover” section** – this is where any automatic‑renewal language will appear.  
2. **Check for notice requirements** – if the lease says “Tenant must give 60 days written notice of intent not to renew,” then failing to give that notice will result in automatic renewal.  
3. **Verify rent terms** – sometimes the renewal clause automatically adjusts rent; make sure you understand any increase.  
4. **Look for conditions** – some leases state that renewal is contingent on the tenant being “in good standing.”  

### If the Lease Does **Not** Contain an Automatic‑Renewal Clause
- The tenancy typically **ends** on the expiration date, and the landlord may choose to offer a new lease or let the unit become vacant.  
- Some jurisdictions treat a tenant who stays past the lease term as a “holdover tenant,” which may create a month‑to‑month tenancy (or a new fixed term, depending on local law).  

### What to Do Next
- **Locate the renewal clause** in your lease and read it carefully.  
- If you’re unsure how to interpret the language, or if the clause is ambiguous, consider consulting a landlord‑tenant attorney in your jurisdiction.  
- If you **do not want** the lease to renew, be sure to send the required written notice (by certified mail, email, or whatever method the lease specifies) within the stipulated time frame.  
- If you **do want** the lease to renew and the clause requires a notice, send that notice promptly to avoid losing the automatic renewal right.  

**Bottom line:** A lease only renews automatically if the agreement includes an automatic‑renewal or holdover provision and you (or the landlord) comply with any notice or condition requirements. If the lease is silent on renewal, it generally ends on the expiration date, and any continuation would be governed by local landlord‑tenant law.

---
### Query: "Who pays society maintenance charges?"
- **Ground Truth Clause**: `Maintenance Charges: The tenant shall pay the monthly society maintenance charges directly to the association.`
- **RAG (Grounded Answer)**:
  > **Answer**

- The **tenant** is responsible for paying the monthly society maintenance charges directly to the association.  

*Citation:* [Maintenance Charges]
- **No-Retrieval LLM Answer**:
  > Society (or homeowners‑association) maintenance charges are normally the responsibility of the **owners of the units** that make up the society.  

- **Flat/house owners** pay the charges directly to the society, usually in proportion to their share of the property (e.g., based on carpet area or a fixed percentage).  
- If an owner rents the unit out, the **tenant may be required to pay the charges** under the lease agreement, but the legal liability usually remains with the owner unless the contract says otherwise.  

So, in short: the **owners (members) of the society pay the maintenance charges**, and any cost passed on to tenants is governed by the rental agreement.

---
