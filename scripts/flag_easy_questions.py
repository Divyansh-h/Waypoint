import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

# Ensure src/ and scripts/ are in the python path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))

from eval.loader import load_eval_set
from ingestion.embed import get_jina_embeddings
from scripts.run_eval import get_db_connection, load_db_config


def main():
    parser = argparse.ArgumentParser(description="Flag overly easy evaluation questions.")
    parser.add_argument("--eval-file", type=str, default="data/eval/eval_set.jsonl")
    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    try:
        examples = load_eval_set(eval_path)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)

    if not examples:
        print(f"No examples found in {eval_path.name}.")
        sys.exit(0)

    # We need the database to look up the actual chunk metadata (function names, etc.)
    conn_str, table_name = load_db_config()
    conn = get_db_connection(conn_str)
    
    console = Console()
    table = Table(title=f"Flagged 'Too Easy' Questions in {eval_path.name}", show_lines=True)
    table.add_column("Q ID", style="cyan")
    table.add_column("Question", style="yellow")
    table.add_column("Leaky Reason", style="red")

    flagged_count = 0

    with conn.cursor() as cur:
        for ex in examples:
            # Flatten all required chunk IDs across all paths
            all_ids = set()
            for path in ex.ground_truth.acceptable_paths:
                all_ids.update(path)
            
            reasons = []
            
            if all_ids:
                try:
                    q_embedding = get_jina_embeddings([ex.question])[0]
                except Exception as e:
                    print(f"Failed to embed question '{ex.id}': {e}")
                    q_embedding = [0.0] * 768  # Assuming 768, but DB query will fail if dimension mismatch, so handle gracefully
                    
                # Query the database for these specific chunks to check their metadata and distance
                cur.execute(
                    f"SELECT id, function_name, file_path, (embedding <=> %s::vector) as distance FROM {table_name} WHERE id = ANY(%s)", 
                    (q_embedding, list(all_ids))
                )
                rows = cur.fetchall()
                
                q_text = ex.question.lower()
                
                for row in rows:
                    chunk_id, func_name, file_path, distance = row
                    
                    # Flag 1: Question explicitly states the target function name
                    if func_name:
                        # Ensure we are checking whole words to avoid false positives (e.g. "fit" matching "fitness")
                        # A simple heuristic for now
                        if func_name.lower() in q_text:
                            reasons.append(f"Contains exact target function: '{func_name}'")
                    
                    # Flag 2: Question explicitly states the target file name
                    if file_path:
                        filename = Path(file_path).stem.lower()
                        # Ignore short generic names like "base" or "utils"
                        if filename in q_text and len(filename) > 4 and filename not in ["base", "utils"]:
                            reasons.append(f"Contains exact target file: '{filename}'")
                            
                    # Flag 3: Dense Leakage - Question is a near-duplicate of the chunk content itself
                    # Cosine distance < 0.2 means Cosine Similarity > 0.8 (extremely high overlap)
                    if distance is not None and distance < 0.2:
                        reasons.append(f"Dense Leakage: Near-duplicate of chunk content (distance: {distance:.3f})")
                            
            if reasons:
                flagged_count += 1
                table.add_row(ex.id, ex.question, "\n".join(set(reasons)))

    conn.close()

    if flagged_count > 0:
        console.print(table)
        console.print(f"\n[bold red]⚠️  Found {flagged_count} potentially leaky questions![/bold red]")
        console.print("Tip: If the answer is restated in the question, BM25 will instantly cheat and find it.")
        console.print("Tip: If the dense distance is < 0.2, the question is basically just copy-pasted chunk code.")
        console.print("Consider rewording these to test semantic/conceptual understanding rather than exact keyword matching.")
    else:
        console.print("\n[bold green]✅ No leaky questions found! Good job writing realistic queries.[/bold green]")


if __name__ == "__main__":
    main()
