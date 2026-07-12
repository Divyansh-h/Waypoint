from training.mining.docstring_pairs import mine_docstring_pairs

def test_mine_docstring_pairs(sample_chunks):
    pairs = mine_docstring_pairs(sample_chunks)
    
    assert len(pairs) > 0
    
    # Verify the trivial docstring was skipped ("predict" chunk has 4 words)
    chunk_names_mined = [p.metadata["extraction_method"] for p in pairs]
    
    for pair in pairs:
        print(f"\\n--- ANCHOR ---\\n{pair.anchor}")
        print(f"--- POSITIVE (Stripped) ---\\n{pair.positive.content}")
