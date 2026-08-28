from rank_bm25 import BM25Okapi
import faiss
import numpy as np

class VectorDB:
    def __init__(self, dim=384):
        self.index = faiss.IndexFlatL2(dim)
        self.text_chunks = []
        self.meta_chunks = []
        self.bm25 = None # initialize later when data is available

    def _matches_filters(self, meta: dict, risk_type=None, doc_type=None, governing_law=None, date_from=None, date_to=None) -> bool:
        if risk_type:
            r_flags = meta.get("risk_flags") or meta.get("risks") or []
            if not any(risk_type.lower() == rf.lower() for rf in r_flags):
                return False
        
        if doc_type:
            d_type = meta.get("doc_type") or ""
            if doc_type.lower() != d_type.lower():
                return False
                
        if governing_law:
            gov_law = meta.get("governing_law") or ""
            if governing_law.lower() not in gov_law.lower():
                return False
                
        if date_from or date_to:
            eff_date = meta.get("effective_date")
            if not eff_date:
                return False
            import re
            from datetime import datetime
            date_match = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", eff_date)
            if date_match:
                try:
                    dt = datetime(int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3)))
                except ValueError:
                    return False
            else:
                parsed = False
                for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y"):
                    try:
                        dt = datetime.strptime(eff_date, fmt)
                        parsed = True
                        break
                    except ValueError:
                        continue
                if not parsed:
                    return False
            
            if date_from:
                try:
                    df_match = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", date_from)
                    if df_match:
                        dt_from = datetime(int(df_match.group(1)), int(df_match.group(2)), int(df_match.group(3)))
                    else:
                        dt_from = datetime.strptime(date_from, "%Y-%m-%d")
                    if dt < dt_from:
                        return False
                except Exception:
                    pass
            if date_to:
                try:
                    dt_match = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", date_to)
                    if dt_match:
                        dt_to = datetime(int(dt_match.group(1)), int(dt_match.group(2)), int(dt_match.group(3)))
                    else:
                        dt_to = datetime.strptime(date_to, "%Y-%m-%d")
                    if dt > dt_to:
                        return False
                except Exception:
                    pass
        return True

    def hybrid_search(self, query: str, query_emb=None, top_k=8, w_bm25=0.4, risk_type=None, doc_type=None, governing_law=None, date_from=None, date_to=None):
        has_filters = any([risk_type, doc_type, governing_law, date_from, date_to])
        valid_indices = []
        if has_filters:
            for idx, meta in enumerate(self.meta_chunks):
                if self._matches_filters(meta, risk_type, doc_type, governing_law, date_from, date_to):
                    valid_indices.append(idx)
            if not valid_indices:
                return []

        # If we have no query, just return top_k from the filtered indices
        if not query or query_emb is None:
            target_indices = valid_indices if has_filters else list(range(len(self.text_chunks)))
            final = target_indices[:top_k]
            return [
                {"text": self.text_chunks[i], "meta": self.meta_chunks[i]}
                for i in final
            ]

        if not self.bm25:
            self.bm25 = BM25Okapi([t.split() for t in self.text_chunks])

        # 1) Vector similarity scores
        k = len(self.text_chunks) if has_filters else (top_k * 2)
        k = min(k, self.index.ntotal)
        if k == 0:
            return []

        D, I = self.index.search(np.array([query_emb]).astype("float32"), k)
        vec_scores = 1 / (1 + D[0])             # distance → similarity
        vec_score_map = {idx: score for idx, score in zip(I[0], vec_scores)}

        # 2) BM25 scores (Normalized to [0, 1] range)
        bm25_raw = np.array(self.bm25.get_scores(query.split()))
        max_b = np.max(bm25_raw) if len(bm25_raw) > 0 else 0.0
        min_b = np.min(bm25_raw) if len(bm25_raw) > 0 else 0.0
        
        if max_b - min_b > 0:
            bm25_scores = (bm25_raw - min_b) / (max_b - min_b)
        else:
            bm25_scores = np.zeros_like(bm25_raw)

        # 3) Combine scores
        combined = []
        target_indices = valid_indices if has_filters else I[0]
        for idx in target_indices:
            if 0 <= idx < len(self.text_chunks):
                vscore = vec_score_map.get(idx, 0.0)
                score = (1 - w_bm25) * vscore + w_bm25 * bm25_scores[idx]
                combined.append((idx, score))

        # 4) Return top‑k
        combined.sort(key=lambda x: x[1], reverse=True)
        final = combined[:top_k]
        return [
            {"text": self.text_chunks[i], "meta": self.meta_chunks[i]}
            for i, _ in final
        ]

    def add(self, embeddings, texts, metas):
        self.index.add(np.array(embeddings).astype("float32"))
        self.text_chunks.extend(texts)
        self.meta_chunks.extend(metas)
        self._rebuild_bm25() 

    def search(self, query_emb, top_k=5):
        D, I = self.index.search(np.array([query_emb]).astype("float32"), top_k)
        return [
            {
                "text": self.text_chunks[i],
                "meta": self.meta_chunks[i],
            }
            for i in I[0]
            if 0 <= i < len(self.text_chunks)
        ]

    def save(self, path):
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path, dim, texts, metas):
        obj = cls(dim)
        obj.index = faiss.read_index(str(path))
        obj.text_chunks = texts
        obj.meta_chunks = metas
        obj._rebuild_bm25()                     # NEW
        return obj
    
    def add_document(self, doc_id: str, chunks: list[str], embeddings, metas: list[dict] = None):
        """
        Appends new document chunks and embeddings to the FAISS index and BM25 corpus
        without a full rebuild of the FAISS index (i.e. without re-embedding existing documents).
        """
        if metas is None:
            metas = []
            for i, chunk in enumerate(chunks):
                metas.append({
                    "doc_id": doc_id,
                    "doc_type": "Unknown",
                    "risk_flags": [],
                    "parties": "Unknown",
                    "effective_date": None,
                    "governing_law": "Unknown",
                    "source_format": "TXT",
                    "clause_index": i,
                    "source": doc_id,
                    "clause": "Unknown"
                })
        self.add(embeddings, chunks, metas)

    def _rebuild_bm25(self):
        """Re‑compute BM25 index on current text_chunks."""
        if not self.text_chunks:
            self.bm25 = None
            return
        tokenized = [t.lower().split() for t in self.text_chunks]
        self.bm25 = BM25Okapi(tokenized)

