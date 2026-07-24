import sys
from pathlib import Path

import pytest

# Ensure we can import from scripts and src
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from eval.metrics import cluster_failures
from ingestion.models import EvalExample
from scripts.run_eval import evaluate_example


def create_mock_example(ground_truth_chunk_ids: list[str]) -> EvalExample:
    """Helper to quickly scaffold a valid EvalExample."""
    return EvalExample(
        id="test_001",
        question="What is this?",
        ground_truth_chunk_ids=ground_truth_chunk_ids,
        difficulty_tag="easy",
        question_type="factual",
        metadata={}
    )

def test_single_chunk_hit_rank_1():
    example = create_mock_example(["chunk_A"])
    retrieved = ["chunk_A", "chunk_B", "chunk_C"]
    
    result = evaluate_example(example, retrieved)
    
    assert result["hit"] is True
    assert result["rank"] == 1
    assert result["reciprocal_rank"] == 1.0

def test_single_chunk_hit_rank_3():
    example = create_mock_example(["chunk_A"])
    retrieved = ["chunk_X", "chunk_Y", "chunk_A"]
    
    result = evaluate_example(example, retrieved)
    
    assert result["hit"] is True
    assert result["rank"] == 3
    assert result["reciprocal_rank"] == pytest.approx(1.0 / 3.0)

def test_single_chunk_miss():
    example = create_mock_example(["chunk_A"])
    retrieved = ["chunk_X", "chunk_Y", "chunk_Z"]
    
    result = evaluate_example(example, retrieved)
    
    assert result["hit"] is False
    assert result["rank"] is None
    assert result["reciprocal_rank"] == 0.0

def test_multi_hop_takes_worst_rank():
    # Ground Truth says we MUST find BOTH A and B to solve the query
    example = create_mock_example(["chunk_A", "chunk_B"])
    # A is found at rank 1, B is found at rank 3.
    # Actual max rank is 3. Path length is 2. Effective rank = 3 - 2 + 1 = 2.
    retrieved = ["chunk_A", "chunk_X", "chunk_B", "chunk_Y"]
    
    result = evaluate_example(example, retrieved)
    
    assert result["hit"] is True
    assert result["rank"] == 2
    assert result["reciprocal_rank"] == 0.5

def test_multi_hop_miss():
    # Ground truth needs BOTH A and B
    example = create_mock_example(["chunk_A", "chunk_B"])
    # Only A is found. B is missing. This is a total failure for a multi-hop query.
    retrieved = ["chunk_A", "chunk_X", "chunk_Y", "chunk_Z"]
    
    result = evaluate_example(example, retrieved)
    
    assert result["hit"] is False
    assert result["rank"] is None
    assert result["reciprocal_rank"] == 0.0

def test_cluster_failures_standard():
    misses = [
        {"id": "q1", "difficulty": "hard", "type": "conceptual", "hit": False},
        {"id": "q2", "difficulty": "hard", "type": "conceptual", "hit": False},
        {"id": "q3", "difficulty": "medium", "type": "factual", "hit": False},
        {"id": "q4", "difficulty": "easy", "type": "conceptual", "hit": False},
    ]
    
    diff_counts, type_counts = cluster_failures(misses)
    
    assert diff_counts == {"HARD": 2, "MEDIUM": 1, "EASY": 1}
    assert type_counts == {"CONCEPTUAL": 3, "FACTUAL": 1}

def test_cluster_failures_empty():
    diff_counts, type_counts = cluster_failures([])
    assert diff_counts == {}
    assert type_counts == {}
    
def test_cluster_failures_missing_keys():
    misses = [
        {"id": "q1", "hit": False} # missing difficulty and type
    ]
    diff_counts, type_counts = cluster_failures(misses)
    assert diff_counts == {"UNKNOWN": 1}
    assert type_counts == {"UNKNOWN": 1}
