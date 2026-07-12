import argparse
import sys
import os

# Add src to python path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import yaml
from training.mining.docstring_pairs import mine_docstring_pairs
from training.synthetic.generate_qa import generate_question_from_chunk
from training.synthetic.cost_tracker import CostTracker
from training.mining.hard_negatives import mine_embedding_neighbors, mine_from_failure_log
from training.filtering.quality import filter_training_pairs
from dotenv import load_dotenv
import time

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Orchestrate Phase 2 synthetic training data generation.")
    parser.add_argument("--config", type=str, default="configs/training_data.yaml", help="Path to config file.")
    args = parser.parse_args()

    # Load YAML Configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    llm_model = config["synthetic_generation"]["llm_model"]
    prompt_template = config["synthetic_generation"]["prompt_template_path"]
    
    cost_tracker = CostTracker(model_name=llm_model)

    print("🚀 Starting Phase 2 Training Data Generation Pipeline...\n")

    # ---------------------------------------------------------
    # 1. Mine Positives
    # ---------------------------------------------------------
    print("STEP 1: Mining positive chunks (docstrings & implementations)...")
    
    # Connect to the DB and fetch candidate chunks
    conn_str = config.get("database", {}).get("connection_string", "postgresql://user:password@localhost:5432/rag_db")
    try:
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(conn_str)
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT id as chunk_id, file_path, chunk_type, content, function_name as name FROM sklearn_code LIMIT 5000;")
            raw_chunks = [dict(row) for row in cur.fetchall()]
        conn.close()
    except Exception as e:
        print(f"   -> DB Error: {e}")
        raw_chunks = []
        
    # Use the docstring miner to extract clean chunks that actually have docstrings
    docstring_pairs = mine_docstring_pairs(raw_chunks)
    
    # Cap at 5 for the synthetic sample to avoid API rate limits during this test run
    sample_pairs = docstring_pairs[:5]
    print(f"   -> Mined {len(docstring_pairs)} valid docstring chunks. Sampling {len(sample_pairs)} for synthetic generation.\n")

    # ---------------------------------------------------------
    # 2. Generate Synthetic Queries
    # ---------------------------------------------------------
    print("STEP 2: Generating synthetic QA queries via LLM API...")
    synthetic_pairs = []
    
    batch_size = config["synthetic_generation"]["batch_size"]
    print(f"   -> Submitting chunks in batches of {batch_size} to {llm_model}...")
    
    # Ensure output directory exists
    os.makedirs("data/training/synthetic", exist_ok=True)
    raw_output_path = "data/training/synthetic/raw_questions.jsonl"
    
    import json
    
    with open(raw_output_path, 'w', encoding='utf-8') as out_f:
        # We generate synthetic questions based on the stripped implementation chunks
        for pair in sample_pairs:
            # We mock the chunk dict since the generator expects it
            chunk = {
                "chunk_id": pair.positive.chunk_id, 
                "content": pair.positive.content,
                "file_path": pair.positive.file_path
            }
            syn_pair = generate_question_from_chunk(chunk, prompt_template)
            synthetic_pairs.append(syn_pair)
            
            usage = syn_pair.metadata.get("usage", {})
            cost_tracker.add_usage(usage.get("input_tokens", 0), usage.get("output_tokens", 0))
            
            # Save raw output with chunk ID
            out_f.write(json.dumps({
                "chunk_id": pair.positive.chunk_id,
                "generated_question": syn_pair.anchor,
                "core_concept": syn_pair.metadata.get("core_concept", ""),
                "llm_model": llm_model
            }) + "\n")
            out_f.flush()
            
            # Simple rate limiting
            time.sleep(0.5)
            
    print(f"   -> Generated {len(synthetic_pairs)} intent-based synthetic questions.")
    print(f"   -> Saved raw output to: {raw_output_path}")
    print(f"   -> 💸 API Usage: {cost_tracker.summary()}\n")

    # ---------------------------------------------------------
    # 3. Mine Hard Negatives
    # ---------------------------------------------------------
    print("STEP 3: Mining hard negatives (Automated ANN & Failure Log Curation)...")
    print("   -> Extracting nearest-neighbor collisions via pgvector...")
    for pair in synthetic_pairs + docstring_pairs:
        # Mine 5 hard negatives for each query
        negatives = mine_embedding_neighbors(query=pair.anchor, true_positive_id=pair.positive.chunk_id, k=5)
        pair.negatives.extend(negatives)
        
    print("   -> Curating multi-hop/abstraction failures from Phase 1 telemetry...")
    curated_failures = mine_from_failure_log("results/week3/dummy_log.json")
    
    all_pairs = synthetic_pairs + docstring_pairs + curated_failures
    print(f"   -> Hard negatives attached. Total dataset pool: {len(all_pairs)} triplets.\n")

    # ---------------------------------------------------------
    # 4. Filter Quality / Anti-Leakage
    # ---------------------------------------------------------
    print("STEP 4: Applying Anti-Leakage & Human-Realism filtering checklists...")
    
    min_words = config.get("filtering", {}).get("min_query_length_words", 4)
    
    filtered_pairs = filter_training_pairs(
        pairs=all_pairs,
        min_words=min_words,
        max_similarity=0.85
    )
    
    dropped_count = len(all_pairs) - len(filtered_pairs)
    hallucination_count = sum(1 for p in filtered_pairs if p.metadata.get("answer_match_flag") == "WARNING_HALLUCINATION")
    
    print(f"   -> Dropped {dropped_count} low-quality or duplicate pairs.")
    print(f"   -> Flagged {hallucination_count} pairs for LLM hallucination review.")
    print(f"   -> {len(filtered_pairs)} pristine pairs remain.\n")

    # ---------------------------------------------------------
    # 5. Stratified Train/Val Split (File-Level Isolation)
    # ---------------------------------------------------------
    print("STEP 5: Splitting into train/val sets (Module-level isolation)...")
    
    import random
    from collections import defaultdict
    
    # Group pairs by their source file to prevent data leakage
    file_to_pairs = defaultdict(list)
    for pair in filtered_pairs:
        # Default to "unknown" if file_path is missing (e.g. from failure logs stub)
        fpath = pair.positive.file_path or "unknown"
        file_to_pairs[fpath].append(pair)
        
    unique_files = list(file_to_pairs.keys())
    # Sort for deterministic shuffling (with seed)
    unique_files.sort()
    random.seed(42)
    random.shuffle(unique_files)
    
    split_idx = int(len(unique_files) * 0.8)
    train_files = set(unique_files[:split_idx])
    val_files = set(unique_files[split_idx:])
    
    train_pairs = []
    val_pairs = []
    for fpath, pairs in file_to_pairs.items():
        if fpath in train_files:
            train_pairs.extend(pairs)
        else:
            val_pairs.extend(pairs)
            
    os.makedirs("data/training/train", exist_ok=True)
    os.makedirs("data/training/val", exist_ok=True)
    
    with open("data/training/train/train.jsonl", 'w', encoding='utf-8') as out_f:
        for pair in train_pairs:
            out_f.write(pair.model_dump_json() + "\n")
            
    with open("data/training/val/val.jsonl", 'w', encoding='utf-8') as out_f:
        for pair in val_pairs:
            out_f.write(pair.model_dump_json() + "\n")
            
    print(f"   -> Training set: {len(train_pairs)} pairs across {len(train_files)} files (Saved to data/training/train/train.jsonl)")
    print(f"   -> Validation set: {len(val_pairs)} pairs across {len(val_files)} files (Saved to data/training/val/val.jsonl)\\n")
    
    print("✅ Pipeline orchestration complete. Ready for Sentence-Transformers MNRL fine-tuning!")

if __name__ == "__main__":
    main()
