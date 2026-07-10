import sys
from pathlib import Path
import pytest

# Ensure we can import from scripts and src
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from scripts.run_eval import evaluate_example
from ingestion.models import EvalExample, GroundTruth

def create_mock_example(acceptable_paths: list[list[str]]) -> EvalExample:
    """Helper to quickly scaffold a valid EvalExample."""
    return EvalExample(
        id="test_001",
        question="What is this?",
        ground_truth=GroundTruth(acceptable_paths=acceptable_paths),
        difficulty_tag="easy",
        question_type="factual",
        metadata={}
    )

def test_single_chunk_hit_rank_1():
    example = create_mock_example([["chunk_A"]])
    retrieved = ["chunk_A", "chunk_B", "chunk_C"]
    
    result = evaluate_example(example, retrieved)
    
    assert result["hit"] is True
    assert result["rank"] == 1
    assert result["reciprocal_rank"] == 1.0

def test_single_chunk_hit_rank_3():
    example = create_mock_example([["chunk_A"]])
    retrieved = ["chunk_X", "chunk_Y", "chunk_A"]
    
    result = evaluate_example(example, retrieved)
    
    assert result["hit"] is True
    assert result["rank"] == 3
    assert result["reciprocal_rank"] == pytest.approx(1.0 / 3.0)

def test_single_chunk_miss():
    example = create_mock_example([["chunk_A"]])
    retrieved = ["chunk_X", "chunk_Y", "chunk_Z"]
    
    result = evaluate_example(example, retrieved)
    
    assert result["hit"] is False
    assert result["rank"] is None
    assert result["reciprocal_rank"] == 0.0

def test_logical_or_takes_best_rank():
    # Ground Truth says finding either A or B is a success
    example = create_mock_example([["chunk_A"], ["chunk_B"]])
    # B is at rank 2, A is at rank 4. The metric should take rank 2.
    retrieved = ["chunk_X", "chunk_B", "chunk_Y", "chunk_A"]
    
    result = evaluate_example(example, retrieved)
    
    assert result["hit"] is True
    assert result["rank"] == 2
    assert result["reciprocal_rank"] == 0.5

def test_logical_and_multi_hop_takes_worst_rank():
    # Ground Truth says we MUST find BOTH A and B to solve the query
    example = create_mock_example([["chunk_A", "chunk_B"]])
    # A is found at rank 1, B is found at rank 3.
    # The agent doesn't have the full context until rank 3, so rank should be 3.
    retrieved = ["chunk_A", "chunk_X", "chunk_B", "chunk_Y"]
    
    result = evaluate_example(example, retrieved)
    
    assert result["hit"] is True
    assert result["rank"] == 3
    assert result["reciprocal_rank"] == pytest.approx(1.0 / 3.0)

def test_logical_and_multi_hop_miss():
    # Ground truth needs BOTH A and B
    example = create_mock_example([["chunk_A", "chunk_B"]])
    # Only A is found. B is missing. This is a total failure for a multi-hop query.
    retrieved = ["chunk_A", "chunk_X", "chunk_Y", "chunk_Z"]
    
    result = evaluate_example(example, retrieved)
    
    assert result["hit"] is False
    assert result["rank"] is None
    assert result["reciprocal_rank"] == 0.0

def test_complex_hybrid_or_and():
    # Ground Truth: Needs (A AND B) OR (C)
    example = create_mock_example([["chunk_A", "chunk_B"], ["chunk_C"]])
    
    # Scenario 1: A and B are found at rank 4 and 5. C is not found.
    # Result -> hit at rank 5.
    retrieved_1 = ["chunk_X", "chunk_Y", "chunk_Z", "chunk_A", "chunk_B"]
    res_1 = evaluate_example(example, retrieved_1)
    assert res_1["hit"] is True
    assert res_1["rank"] == 5
    
    # Scenario 2: C is found at rank 2. A is found at rank 5. (B missing).
    # Result -> hit at rank 2 via the ["chunk_C"] OR path.
    retrieved_2 = ["chunk_X", "chunk_C", "chunk_Y", "chunk_Z", "chunk_A"]
    res_2 = evaluate_example(example, retrieved_2)
    assert res_2["hit"] is True
    assert res_2["rank"] == 2
