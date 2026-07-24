from training.filtering.quality import filter_training_pairs
from training.schema import PositiveChunk, TrainingPair


def make_pair(
    anchor: str, 
    concept: str = "generic concept", 
    content: str = "generic code implementation content"
) -> TrainingPair:
    return TrainingPair(
        anchor=anchor,
        positive=PositiveChunk(chunk_id="chunk1", content=content, file_path="test.py"),
        negatives=[],
        source="mined",
        metadata={"core_concept": concept}
    )

def test_filter_too_short():
    """Ensure questions with fewer than min_words are dropped."""
    pairs = [
        make_pair("Too short"), # 2 words -> fail
        make_pair("This one is perfectly fine") # 5 words -> pass
    ]
    filtered = filter_training_pairs(pairs, min_words=4)
    assert len(filtered) == 1
    assert filtered[0].anchor == "This one is perfectly fine"

def test_filter_near_duplicate():
    """Ensure near-duplicate semantic questions are dropped based on TF-IDF threshold."""
    pairs = [
        make_pair("How do I initialize the main random forest classifier?"),
        make_pair("How do I initialize the main random forest classifier?"), # exact duplicate -> fail
        make_pair("How do I initialize the main random forest regression?"), # near duplicate -> fail (high tf-idf overlap)
        make_pair("Where is the database connection string located?") # completely distinct -> pass
    ]
    
    # Run with a slightly lower threshold to guarantee the near-duplicate is caught
    # (In a tiny 4-document corpus, the single distinct word gets huge IDF weight, lowering similarity)
    filtered = filter_training_pairs(pairs, min_words=4, max_similarity=0.60)
    
    assert len(filtered) == 2
    assert filtered[0].anchor == "How do I initialize the main random forest classifier?"
    assert filtered[1].anchor == "Where is the database connection string located?"

def test_relevance_flagging():
    """Ensure hallucinatory synthetic concepts flag properly."""
    pairs = [
        # Valid: concept "database" exists in code
        make_pair(
            anchor="How to connect to database?",
            concept="database connection",
            content="def connect_db(): return database()"
        ),
        # Invalid: concept "spaceship" doesn't exist in a simple math function
        make_pair(
            anchor="How to fly to the moon?",
            concept="spaceship launch rocket",
            content="def sum(a, b): return a + b"
        )
    ]
    filtered = filter_training_pairs(pairs, min_words=4)
    assert len(filtered) == 2
    assert filtered[0].metadata["answer_match_flag"] == "PASS"
    assert filtered[1].metadata["answer_match_flag"] == "WARNING_HALLUCINATION"
