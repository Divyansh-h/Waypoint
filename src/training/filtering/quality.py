from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

from training.schema import TrainingPair


def filter_training_pairs(
    pairs: List[TrainingPair], 
    min_words: int = 4,
    max_similarity: float = 0.85
) -> List[TrainingPair]:
    """
    Applies the Phase 1 filtering checklist to the synthetic training dataset:
    1. Drops questions that are too short (under min_words).
    2. Drops near-duplicate questions using TF-IDF cosine similarity.
    3. Flags questions where the LLM's concept heavily hallucinates
       or doesn't match the source chunk.
    
    Args:
        pairs: The list of raw synthetic TrainingPairs.
        min_words: Minimum word count for a valid question.
        max_similarity: Cosine similarity threshold for dropping duplicates (0.0 to 1.0).
        
    Returns:
        A pristine list of filtered TrainingPairs.
    """
    if not pairs:
        return []
        
    import json
    import os
    os.makedirs("data/training", exist_ok=True)
    
    dropped_log = []
    
    # 1. Length Filter (Drop questions under N words)
    length_filtered = []
    for pair in pairs:
        word_count = len(pair.anchor.split())
        if word_count >= min_words:
            length_filtered.append(pair)
        else:
            dropped_log.append({
                "reason": "TOO_SHORT",
                "anchor": pair.anchor,
                "source": pair.source
            })
            
    if not length_filtered:
        if dropped_log:
            with open("data/training/dropped_pairs.jsonl", "w") as f:
                for d in dropped_log:
                    f.write(json.dumps(d) + "\
")
        return []
        
    # 2. Near-Duplicate Filter (Embedding Similarity)
    anchors = [p.anchor for p in length_filtered]
    
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(anchors)
        cosine_sim_matrix = cosine_similarity(tfidf_matrix)
    except ValueError:
        return length_filtered

    drop_indices = set()
    for i in range(len(anchors)):
        if i in drop_indices:
            continue
        for j in range(i + 1, len(anchors)):
            if cosine_sim_matrix[i, j] > max_similarity:
                drop_indices.add(j) 
                
    deduped_pairs = []
    for i, pair in enumerate(length_filtered):
        if i not in drop_indices:
            deduped_pairs.append(pair)
        else:
            dropped_log.append({
                "reason": "NEAR_DUPLICATE",
                "anchor": pair.anchor,
                "source": pair.source
            })
            
    if dropped_log:
        with open("data/training/dropped_pairs.jsonl", "w") as f:
            for d in dropped_log:
                f.write(json.dumps(d) + "\
")
    
    # 3. Relevance Flagging (LLM cited answer vs source chunk)
    filtered_pairs = []
    for pair in deduped_pairs:
        # Check if the LLM's "core_concept" actually shares any
        # semantic/lexical overlap with the code chunk
        concept = pair.metadata.get("core_concept", "").lower()
        chunk_text = pair.positive.content.lower()
        
        # Simple heuristic: Do any meaningful words (len > 3) from the concept exist in the code?
        concept_words = set([w for w in concept.split() if len(w) > 3])
        
        if concept_words:
            overlap = len([w for w in concept_words if w in chunk_text])
            if overlap == 0:
                # The LLM hallucinated a concept completely unrelated to the code chunk text
                pair.metadata["answer_match_flag"] = "WARNING_HALLUCINATION"
            else:
                pair.metadata["answer_match_flag"] = "PASS"
        else:
            pair.metadata["answer_match_flag"] = "PASS_NO_CONCEPT"
            
        filtered_pairs.append(pair)
        
    return filtered_pairs
