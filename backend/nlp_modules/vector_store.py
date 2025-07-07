from rank_bm25 import BM25Okapi
import faiss
import numpy as np

class VectorDB:
    def __init__(self, dim=384):
        self.index = faiss.IndexFlatL2(dim)
        self.text_chunks = []
        self.meta_chunks = []
        self.bm25 = None # initialize later when data is available

    def hybrid_search(self, query: str, query_emb, top_k=8, w_bm25=0.4):

        if not self.bm25:
            self.bm25 = BM25Okapi([t.split() for t in self.text_chunks])

        # 1) Vector similarity scores
        D, I = self.index.search(np.array([query_emb]).astype("float32"), top_k * 2)
        vec_scores = 1 / (1 + D[0])             # distance → similarity

        # 2) BM25 scores
        bm25_scores = self.bm25.get_scores(query.split())

        # 3) Combine scores
        combined = []
        for idx, vscore in zip(I[0], vec_scores):
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
    
    def _rebuild_bm25(self):
        """Re‑compute BM25 index on current text_chunks."""
        if not self.text_chunks:
            self.bm25 = None
            return
        tokenized = [t.lower().split() for t in self.text_chunks]
        self.bm25 = BM25Okapi(tokenized)

