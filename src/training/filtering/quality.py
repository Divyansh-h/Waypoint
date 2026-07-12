import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
    3. Flags questions where the LLM's concept heavily hallucinates or doesn't match the source chunk.
    
    Args:
        pairs: The list of raw synthetic TrainingPairs.
        min_words: Minimum word count for a valid question.
        max_similarity: Cosine similarity threshold for dropping duplicates (0.0 to 1.0).
        
    Returns:
        A pristine list of filtered TrainingPairs.
    """
    if not pairs:
        return []
        
    # 1. Length Filter (Drop questions under N words)
    length_filtered = []
    for pair in pairs:
        word_count = len(pair.anchor.split())
        if word_count >= min_words:
            length_filtered.append(pair)
            
    if not length_filtered:
        return []
        
    # 2. Near-Duplicate Filter (Embedding Similarity)
    # Using TF-IDF as an ultra-fast, local heuristic for exact/near-exact question overlap
    anchors = [p.anchor for p in length_filtered]
    
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(anchors)
        cosine_sim_matrix = cosine_similarity(tfidf_matrix)
    except ValueError:
        # Fallback if vocabulary is entirely empty (e.g. all stop words)
        return length_filtered

    # Identify duplicates
    drop_indices = set()
    for i in range(len(anchors)):
        if i in drop_indices:
            continue
        # Compare current question with all subsequent questions
        for j in range(i + 1, len(anchors)):
            if cosine_sim_matrix[i, j] > max_similarity:
                drop_indices.add(j) # Mark the duplicate for deletion
                
    deduped_pairs = [pair for i, pair in enumerate(length_filtered) if i not in drop_indices]
    
    # 3. Relevance Flagging (LLM cited answer vs source chunk)
    filtered_pairs = []
    for pair in deduped_pairs:
        # Check if the LLM's "core_concept" actually shares any semantic/lexical overlap with the code chunk
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
