# Comprehensive Legal AI — Improvement Tasks
Repo: https://github.com/Pranavsk22/Legal-AI-v2.0

## Why these changes
This repo is already the closest architectural match on the list: FastAPI backend, hybrid
FAISS + BM25 retrieval, multi-format parsing (PDF/OCR/DOCX/HTML), and clause-level cited Q&A.
That maps almost directly onto three separate Mitacs picks:

- **Project 1 — LLM-Assisted Discovery of Emotion Datasets (D4ED)**: wants a two-workflow
  system (form-driven record creation + PDF-upload-to-LLM-extraction-to-human-review), schema
  validation before writing to a database, and structured/keyword/filtered search. Legal AI
  already has the PDF-to-structured-data half; it's missing schema validation and filtered
  search.
- **Project 3 — LLM-Powered Vulnerability Detection with RAG + Vector DBs**: wants embeddings +
  vector DB + metadata filtering by category/severity, evaluated against a non-retrieval
  baseline, with incremental indexing. Legal AI's `vector_store.py` (FAISS+BM25 hybrid) is the
  same architecture applied to a different domain — the risk engine's categories are your
  "weakness categories."
- **Project 6 — More efficient indexing for vector databases**: wants indexing/build-time and
  retrieval-quality trade-off analysis, which you can produce cheaply from the eval harness
  below.
- **Project 9 — Course-Specific LLMs for Business Education**: wants RAG grounded in a defined
  corpus with evaluation of accuracy/consistency — the same skeleton, different documents.

## Tasks (execute in order, commit after each numbered task)

### 1. Add a metadata schema + validator
Create `backend/nlp_modules/schema.py` with a `pydantic` (or `jsonschema`) model
`DocumentRecord { doc_id, doc_type, risk_flags: list[str], parties, effective_date,
governing_law, source_format, clause_index }`. In `universal_parser.py`, populate this model
after parsing and validate it before anything is written to `vector_store.py`. Reject/flag
malformed extractions instead of silently indexing them.

### 2. Add a structured filter/search endpoint
In `backend/api/routes.py`, add `GET /api/search` accepting query params
`risk_type`, `doc_type`, `governing_law`, `date_from`, `date_to`. Combine metadata filtering
(over the validated records from Task 1) with the existing hybrid FAISS+BM25 semantic search,
so a query can be narrowed by category *and* meaning — this is the "metadata filtering by
weakness category and severity" pattern from Project 3 and the "structured search, keyword
search, and filtering" pattern from Project 1.

### 3. Add incremental indexing
Extend `vector_store.py` with an `add_document(doc_id, chunks, embeddings)` method that appends
to the existing FAISS index and BM25 corpus without a full rebuild. Add a small test that
indexes 10 docs, then incrementally adds 1 more, and confirms search results include it without
re-embedding the first 10.

### 4. Add a retrieval evaluation harness
Create `scripts/eval_retrieval.py`: build a small held-out set (~20 question/expected-clause
pairs) from your own sample contracts, then report precision@k / recall@k for three
configurations — BM25-only, FAISS-only, and your hybrid — plus a "no-retrieval" baseline where
the LLM answers from the question alone. Write results to `reports/retrieval_eval.md` as a
table. This single script gives you a defensible number to cite for Projects 1, 3, and 6 alike.

### 5. Add the "review before finalize" workflow
Currently `universal_parser.py` → `risk_rules.py`/`summarizer.py` runs straight through to an
answer. Add a `POST /api/ingest/draft` endpoint that returns the extracted `DocumentRecord`
(Task 1) as an editable draft (not yet indexed), and a separate `POST /api/ingest/confirm`
that validates the human-edited version and only then writes to the vector store. This mirrors
D4ED's "draft record → human review → finalized record" flow from Project 1 almost exactly.

### 6. Document architecture and trade-offs
Add `ARCHITECTURE.md` covering: why hybrid FAISS+BM25 over either alone, the schema validation
approach, and the incremental-indexing design. Reference the eval numbers from Task 4.

### 7. Update the README
Add a "Structured Search & Schema Validation" feature bullet and a "Retrieval Evaluation"
bullet linking to `reports/retrieval_eval.md`, under the existing feature list — keep
installation instructions unchanged.

### UI DESIGN CHANGES
Avoid these UI designs. if any of the below UI designs are already existing, change them immediately:
Visual/Styling

Purple-to-blue gradient backgrounds everywhere (the default AI "premium" gradient)
Excessive glassmorphism/backdrop-blur on every card
Every button has a glow/shadow effect regardless of context
Overuse of rounded-2xl or rounded-3xl on literally everything
Emoji used as icons instead of a proper icon set
Inconsistent spacing that "mostly" aligns but isn't on a real grid
Default Tailwind color palette used raw (indigo-600, violet-500) with no custom theme
Drop shadows applied inconsistently in intensity/direction across components
Hero sections with a centered headline, subheadline, and two CTA buttons — every single time
Icon + heading + paragraph "feature card" grids in sets of 3 or 4, repeated for every section

Typography
11. Font weight jumps straight from 400 to 700/800 with nothing in between
12. Overly large, bold headlines with little hierarchy differentiation elsewhere
13. Inter/system-ui as the only font, no pairing or personality
14. Gradient text on headlines

Layout/Structure
15. Every page follows the exact same template: hero → features → testimonials → CTA → footer
16. Sidebar nav with icon + label pairs that all look identical in visual weight
17. Cards with identical padding/shadow/radius stacked with no visual rhythm
18. Dashboard with 4 stat cards in a row, each with an icon top-left and a big number
19. Empty states that are overly cute (illustration + friendly copy) but generic
20. Modals that pop up centered with a blurred overlay for literally every interaction, even trivial ones

Interaction/Motion
21. Every element fades/slides in on scroll, applied uniformly with no intentional variation
22. Hover states that just scale(1.05) everything
23. Loading skeletons that don't match the actual content shape
24. Toast notifications for every single action, including non-critical ones
25. Overuse of "smooth" easing/transition on all elements, making UI feel sluggish rather than snappy

Content/Copy
26. Placeholder-y microcopy that's technically correct but generic ("Manage your tasks efficiently")
27. Testimonial sections with obviously fake names, stock avatars, and vague praise
28. Feature descriptions that restate the feature name rather than explain a benefit
29. Pricing tables with "Starter / Pro / Enterprise" tiers copied almost verbatim from SaaS templates
30. Excessive use of checkmark icons (✓) in green next to every list item, regardless of relevance


## Git workflow
```bash
git checkout -b feature/schema-validated-search
git add -A
git commit -m "Add schema-validated extraction, filtered search, incremental indexing, and retrieval eval harness"
git push origin feature/schema-validated-search
git checkout main && git merge feature/schema-validated-search && git push origin main
```
