import sys
from pathlib import Path
import yaml

# Ensure src/ is in the python path
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from retrieval.reranker import get_reranker

def test_reranker_order():
    print("Initializing Reranker...")
    # Load config to get model name
    with open("configs/ingestion.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_name = config.get("retrieval", {}).get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    reranker = get_reranker(model_name)
    
    query = "How do I train a machine learning model?"
    
    # 3 fake candidates with obvious relevance ordering
    candidates = [
        {
            "id": "c3",
            "content": "The sky is blue and grass is green. This has nothing to do with programming.",
            "expected_rank": 3
        },
        {
            "id": "c1",
            "content": "To train a machine learning model, you typically call the .fit() method on your training data.",
            "expected_rank": 1
        },
        {
            "id": "c2",
            "content": "Models are mathematical representations of data patterns.",
            "expected_rank": 2
        }
    ]
    
    print("\n[Input Candidates]")
    for c in candidates:
        print(f"ID: {c['id']} | Content: {c['content']}")
        
    print(f"\n[Query] {query}")
    
    print("\nRunning reranker...")
    # rerank() should return a list of dicts sorted by 'score' descending
    ranked = reranker.rerank(query, candidates)
    
    print("\n[Output Ranked Candidates]")
    for i, c in enumerate(ranked):
        score = c.get('score', 'N/A')
        if isinstance(score, float):
            score_str = f"{score:.4f}"
        else:
            score_str = str(score)
        print(f"Rank {i+1}: ID: {c['id']} | Score: {score_str} | Content: {c['content']}")
        
    # Check the order is c1, c2, c3
    actual_ids = [c["id"] for c in ranked]
    expected_ids = ["c1", "c2", "c3"]
    
    print("\n--- RESULTS ---")
    if actual_ids == expected_ids:
        print("✅ SUCCESS! The reranker sorted the candidates correctly.")
    else:
        print(f"❌ FAILURE! Expected order {expected_ids} but got {actual_ids}.")
        
    # Check for silent truncation
    if len(ranked) == len(candidates):
        print("✅ SUCCESS! No silent truncation occurred.")
    else:
        print(f"❌ FAILURE! Expected {len(candidates)} candidates but got {len(ranked)}.")

if __name__ == "__main__":
    test_reranker_order()
