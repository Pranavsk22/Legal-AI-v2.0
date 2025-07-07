# scripts/index_contracts.py
"""
Walk through data/legal_docs/contracts, extract text from *any* supported
format (PDF, DOCX, ADOC, scanned PDF), chunk & embed it, then store vectors
into a FAISS index on disk so later scripts can query instantly.
"""
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


import pickle
from pathlib import Path
from backend.nlp_modules import universal_parser, embedder, vector_store


DATA_DIR   = Path("data/legal_docs/contracts")
INDEX_PATH = (Path("vector_db") / "contracts.faiss").resolve()
META_PATH  = (Path("vector_db") / "contracts_text.pkl").resolve()


def build_index():
    vdb = vector_store.VectorDB()
    for file in DATA_DIR.iterdir():
        if not file.is_file():
            continue

        print(f"📄 Parsing {file.name}")
        try:
            text = universal_parser.extract_text(str(file))
            chunks, metas = universal_parser.split_into_chunks(text)
            if not chunks:
                print(f"⚠️  No text extracted from {file.name}")
                continue

            # Add file‑name to metadata so we can cite it later
            for m in metas:
                m["source"] = file.name

            embeddings = embedder.embed_chunks(chunks)
            vdb.add(embeddings, chunks, metas)       # 👈 now passes metas
        except Exception as e:
            print(f"❌ Failed on {file.name}: {e}")

    # ─── persist index + metadata ──────────────────────────────────────────
    INDEX_PATH.parent.mkdir(exist_ok=True)          # ensure vector_db/
    vdb.save(INDEX_PATH)

    # 🔐 store BOTH text + meta
    pickle.dump({"texts": vdb.text_chunks, "metas": vdb.meta_chunks},
                META_PATH.open("wb"))

    print(f"✅ Indexed {len(vdb.text_chunks)} chunks → {INDEX_PATH}")


if __name__ == "__main__":
    Path("vector_db").mkdir(exist_ok=True)
    build_index()
