import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from retrieval.fusion import reciprocal_rank_fusion

def test_rrf_disjoint_lists():
    dense_results = [
        {"id": "c1", "content": "Chunk 1 - Dense Only (Rank 1)"},
        {"id": "c2", "content": "Chunk 2 - In Both (Dense Rank 2)"}
    ]
    
    bm25_results = [
        {"id": "c3", "content": "Chunk 3 - BM25 Only (Rank 1)"},
        {"id": "c2", "content": "Chunk 2 - In Both (BM25 Rank 2)"}
    ]
    
    fused = reciprocal_rank_fusion(dense_results, bm25_results, k=60, top_n=5)
    
    print("--- Fused Results ---")
    for rank, doc in enumerate(fused, 1):
        print(f"Rank {rank}: ID={doc['id']}, Score={doc['rrf_score']:.4f} | {doc['content']}")
        
if __name__ == "__main__":
    test_rrf_disjoint_lists()
