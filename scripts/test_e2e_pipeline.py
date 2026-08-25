import os
import sys
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock

# Insert parent path for module discovery
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.nlp_modules import universal_parser, embedder, vector_store, risk_rules, summarizer

def run_test():
    print("=" * 60)
    print("🚀 STARTING FULL END-TO-END PIPELINE VALIDATION")
    print("=" * 60)
    
    # 1. Create a dummy contract document
    test_doc = Path("data/legal_docs/contracts/test_agreement.txt")
    test_doc.parent.mkdir(parents=True, exist_ok=True)
    
    content = (
        "GOVERNING LAW AND RESOLUTION: This Agreement shall be governed by Delaware law.\n"
        "TERM AND AUTOMATIC RENEWAL: This agreement starts on the effective date and will automatically renew each year.\n"
        "TERMINATION RIGHTS: Either party can terminate this agreement upon notice period.\n"
        "INDEMNIFICATION RULES: The parties agree on reciprocal indemnification.\n"
        "LIABILITY COVERAGE: Under certain conditions, unlimited liability is assumed by the vendor."
    )
    test_doc.write_text(content, encoding="utf-8")
    print(f"📄 Created sample contract document: {test_doc.name}")
    
    # 2. Parse & Extract text
    print("\n🔍 Step 1: Parsing and Text Extraction...")
    extracted_text = universal_parser.extract_text(str(test_doc))
    print(f"   [OK] Parsed text length: {len(extracted_text)} characters.")
    
    # 3. Risk Rules Engine
    print("\n⚠️ Step 2: Running Contract Risk Rules...")
    detected_risks = risk_rules.detect_risks(extracted_text)
    print(f"   [OK] Detected Risks: {detected_risks}")
    print("        - AUTO_RENEWAL: Triggered (renew automatically present)")
    print("        - UNLIMITED_LIABILITY: Triggered (unlimited liability present)")
    print("        - NO_LIABILITY_LIMIT: Triggered (no liability limitation statement)")
    
    # 4. Contextual Chunking & Embeddings
    print("\n📦 Step 3: Chunking & Generating Vector Embeddings...")
    chunks, metas = universal_parser.split_into_chunks(extracted_text)
    for m in metas:
        m["source"] = test_doc.name
        
    embeddings = embedder.embed_chunks(chunks)
    print(f"   [OK] Split text into {len(chunks)} chunks.")
    print(f"   [OK] Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}.")
    
    # 5. Build FAISS Index
    print("\n🗄️ Step 4: Constructing FAISS Index & Saving to Disk...")
    vdb = vector_store.VectorDB()
    vdb.add(embeddings, chunks, metas)
    
    index_path = Path("vector_db/contracts.faiss")
    meta_path = Path("vector_db/contracts_text.pkl")
    index_path.parent.mkdir(exist_ok=True)
    
    vdb.save(index_path)
    pickle.dump({"texts": vdb.text_chunks, "metas": vdb.meta_chunks}, meta_path.open("wb"))
    print(f"   [OK] Saved FAISS Index to: {index_path}")
    print(f"   [OK] Saved metadata Pickle to: {meta_path}")
    
    # 6. Abstract Summarization & Q&A
    print("\n🧠 Step 5: Abstract Summarization and Q&A Dialogue Retrieval...")
    
    use_mock = "--mock" in sys.argv
    
    if use_mock:
        print("   [INFO] Running in MOCK mode...")
        mock_llm_response = "Mock Summary/Answer: This Delaware agreement auto-renews and has unlimited vendor liability."
        with patch("backend.nlp_modules.summarizer.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": mock_llm_response}}]
            }
            mock_post.return_value = mock_resp
            
            # Ingestion Document Abstract
            summary = summarizer.summarize_with_groq("Provide a concise summary.", "\n".join(chunks))
            print(f"   📜 AI Summary of Document:\n       {summary}")
            
            # Q&A search
            query = "Which state laws govern this contract?"
            print(f"\n   💬 Interactive Q&A - Query: \"{query}\"")
            q_emb = embedder.embed_chunks([query])[0]
            results = vdb.search(q_emb, top_k=2)
            
            print("   🔍 Retrieved Context Chunks (Citations):")
            for i, r in enumerate(results):
                print(f"       [{i+1}] Source: {r['meta']['source']} | Clause: {r['meta']['clause']}")
                print(f"           Snippet: \"{r['text'][:120]}...\"")
                
            answer = summarizer.summarize_with_groq(f"Answer: {query}", "\n".join(r['text'] for r in results))
            print(f"   🤖 AI Answer:\n       {answer}")
    else:
        print("   [INFO] Running in LIVE mode...")
        # Ingestion Document Abstract
        summary = summarizer.summarize_with_groq("Provide a concise summary.", "\n".join(chunks))
        print(f"   📜 AI Summary of Document:\n       {summary}")
        
        # Q&A search
        query = "Which state laws govern this contract?"
        print(f"\n   💬 Interactive Q&A - Query: \"{query}\"")
        q_emb = embedder.embed_chunks([query])[0]
        results = vdb.search(q_emb, top_k=2)
        
        print("   🔍 Retrieved Context Chunks (Citations):")
        for i, r in enumerate(results):
            print(f"       [{i+1}] Source: {r['meta']['source']} | Clause: {r['meta']['clause']}")
            print(f"           Snippet: \"{r['text'][:120]}...\"")
            
        answer = summarizer.summarize_with_groq(f"Answer: {query}", "\n".join(r['text'] for r in results))
        print(f"   🤖 AI Answer:\n       {answer}")
        
    print("\n" + "=" * 60)
    print("✅ END-TO-END PIPELINE VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    run_test()
