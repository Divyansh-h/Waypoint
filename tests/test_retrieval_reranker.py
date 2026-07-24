import sys
from pathlib import Path

# Ensure src/ is in the python path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "src"))

from retrieval.reranker import Reranker


def test_reranker_reordering_hard_negatives():
    """
    Tests that a cross-encoder can correctly reorder a list containing
    hard negatives that a pure Bi-Encoder might struggle to separate.
    """
    reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    query = "How do I initialize the penalty parameter for Logistic Regression?"
    
    # A candidate pool where a naive dense retriever might get confused,
    # placing the exact match at Rank 2 and a "hard negative" at Rank 1.
    candidates = [
        {"id": "hard_negative_ridge", "content": "class RidgeClassifier: def __init__(self, penalty='l2'): ..."},
        {"id": "true_positive_logistic", "content": "class LogisticRegression: def __init__(self, penalty='l2'): ..."}
    ]
    
    ranked = reranker.rerank(query, candidates)
    
    assert len(ranked) == 2
    
    # Verify the cross-encoder accurately promotes the true positive over the hard negative.
    assert ranked[0]["id"] == "true_positive_logistic", "The cross-encoder should identify the exact conceptual match."
    
    # Check that it assigned real scores
    assert "reranker_score" in ranked[0]
    assert ranked[0]["reranker_score"] > ranked[1]["reranker_score"]
