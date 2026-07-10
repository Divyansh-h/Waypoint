import argparse
import json
import os
import sys
import uuid
from pathlib import Path

import psycopg2
from pgvector.psycopg2 import register_vector
import yaml

# Ensure src/ is in the python path
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ingestion.embed import get_jina_embeddings
from ingestion.models import EvalExample, GroundTruth


def load_db_config(config_path="configs/ingestion.yaml"):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config["database"]["connection_string"], config["database"].get("collection_name", "chunks")
    except Exception as e:
        print(f"Error loading config: {e}")
        return "postgresql://user:password@localhost:5432/rag_db", "chunks"


def get_db_connection(conn_str):
    try:
        conn = psycopg2.connect(conn_str)
        register_vector(conn)
        return conn
    except Exception as e:
        print(f"Failed to connect to database. Ensure Postgres is running. Error: {e}")
        sys.exit(1)


def search_chunks_vector(conn, table_name, query_text, limit=5):
    """Embeds the query and performs a vector similarity search."""
    print(f"\n[🔍] Embedding query via Jina API...")
    try:
        # get_jina_embeddings returns a list of embeddings, we want the first one
        embeddings = get_jina_embeddings([query_text])
        if not embeddings:
            return []
        query_vector = embeddings[0]
    except Exception as e:
        print(f"Embedding failed: {e}")
        return []

    print(f"[🔍] Querying PostgreSQL ({table_name})...")
    with conn.cursor() as cur:
        # Using cosine distance (<=>) for similarity
        cur.execute(
            f"""
            SELECT id, file_path, function_name, line_start, content 
            FROM {table_name} 
            ORDER BY embedding <=> %s::vector 
            LIMIT %s
            """,
            (query_vector, limit)
        )
        return cur.fetchall()


def search_chunks_keyword(conn, table_name, keyword, limit=5):
    """Performs a basic ILIKE text search if vector search missed the GT."""
    print(f"[🔍] Searching database for keyword: '{keyword}'...")
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, file_path, function_name, line_start, content 
            FROM {table_name} 
            WHERE content ILIKE %s OR function_name ILIKE %s OR file_path ILIKE %s
            LIMIT %s
            """,
            (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", limit)
        )
        return cur.fetchall()


def print_candidates(results):
    print("\n" + "="*80)
    print(" CANDIDATE CHUNKS ".center(80))
    print("="*80)
    if not results:
        print("No results found. (Is the database populated?)")
        return

    for i, row in enumerate(results):
        chunk_id, file_path, func_name, line_start, content = row
        print(f"\n[{i}] ID: {chunk_id}")
        print(f"    File: {file_path}:{line_start}")
        print(f"    Func: {func_name}")
        snippet = content[:150].replace('\n', ' ') + "..." if len(content) > 150 else content.replace('\n', ' ')
        print(f"    Preview: {snippet}")


def main():
    parser = argparse.ArgumentParser(description="Interactive tool to add eval questions.")
    parser.add_argument("--eval-file", type=str, default="data/eval/eval_set_draft.jsonl")
    args = parser.parse_args()

    eval_file = Path(args.eval_file)
    eval_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Connect to DB
    conn_str, table_name = load_db_config()
    conn = get_db_connection(conn_str)

    print("\n=== RAG Evaluation Set Builder ===")
    print("Type 'quit' or 'exit' at the question prompt to stop.")

    while True:
        # 2. Get Question
        print("\n" + "-"*50)
        question = input("Enter a new eval question:\n> ").strip()
        if question.lower() in ['quit', 'exit']:
            break
        if not question:
            continue

        # 3. Candidate Search Loop
        while True:
            candidates = search_chunks_vector(conn, table_name, question, limit=10)
            print_candidates(candidates)

            print("\n" + "-"*50)
            print("Select ground truth chunks.")
            print("Options:")
            print(" - Enter comma-separated IDs for a Logical AND path (e.g., 'id1,id2')")
            print(" - Enter 'search: <keyword>' if the chunk you want isn't in the list")
            print(" - Enter 'skip' to discard this question entirely")
            
            choice = input("> ").strip()
            
            if choice.lower() == 'skip':
                print("Skipping question.")
                break
            
            if choice.lower().startswith("search:"):
                keyword = choice.split(":", 1)[1].strip()
                candidates = search_chunks_keyword(conn, table_name, keyword, limit=10)
                print_candidates(candidates)
                # Re-prompt for selection
                choice = input("Enter comma-separated IDs (or 'skip'):\n> ").strip()
                if choice.lower() == 'skip':
                    break

            if choice:
                # 4. Process IDs
                selected_ids = [c.strip() for c in choice.split(",") if c.strip()]
                
                # 5. Metadata
                diff = ""
                while diff not in ["easy", "medium", "hard", "adversarial"]:
                    diff = input("Difficulty [easy/medium/hard/adversarial]: ").strip().lower()
                    
                qtype = ""
                while qtype not in ["factual", "debugging", "conceptual", "api_usage", "out_of_scope"]:
                    qtype = input("Type [factual/debugging/conceptual/api_usage/out_of_scope]: ").strip().lower()

                # 6. Build and save Example
                # Storing as a single path for now. To add multiple OR paths, you could edit the JSONL manually later.
                example = EvalExample(
                    id=f"eval_{uuid.uuid4().hex[:8]}",
                    question=question,
                    ground_truth=GroundTruth(acceptable_paths=[selected_ids]),
                    difficulty_tag=diff,
                    question_type=qtype
                )
                
                with open(eval_file, "a", encoding="utf-8") as f:
                    f.write(example.model_dump_json() + "\n")
                
                print(f"\n✅ Saved to {eval_file}!")
                break

    conn.close()
    print("Exiting.")

if __name__ == "__main__":
    main()
