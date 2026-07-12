import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import psycopg2
from pgvector.psycopg2 import register_vector
import yaml

from rich.console import Console
from rich.table import Table

# Ensure src/ is in the python path
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from eval.loader import load_eval_set
from ingestion.models import EvalExample
from ingestion.embed import get_jina_embeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("run_eval")

# Global override for CLI sweeping
GLOBAL_POOL_SIZE_OVERRIDE = None


def load_db_config(config_path="configs/ingestion.yaml"):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config["database"]["connection_string"], config["database"].get("collection_name", "chunks")
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return "postgresql://user:password@localhost:5432/rag_db", "chunks"


def get_db_connection(conn_str):
    try:
        conn = psycopg2.connect(conn_str)
        register_vector(conn)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)


RETRIEVAL_METHODS = {
    "dense": None,
    "bm25": None,
    "hybrid": None,
    "reranked_hybrid": None,
}

_pipeline = None

def retrieve_chunks(conn, table_name: str, query: str, method: str, k: int = 10) -> List[str]:
    """
    Retrieves top-K chunk IDs using the requested preset via RetrievalPipeline.
    """
    global _pipeline
    if _pipeline is None:
        from retrieval.pipeline import RetrievalPipeline
        global GLOBAL_POOL_SIZE_OVERRIDE, GLOBAL_RRF_K_OVERRIDE
        pool_size = GLOBAL_POOL_SIZE_OVERRIDE if 'GLOBAL_POOL_SIZE_OVERRIDE' in globals() else None
        rrf_k = GLOBAL_RRF_K_OVERRIDE if 'GLOBAL_RRF_K_OVERRIDE' in globals() else None
        
        _pipeline = RetrievalPipeline(conn, table_name, rrf_k_override=rrf_k, pool_size_override=pool_size)
        
    return _pipeline.retrieve(query, method, k)


def evaluate_example(example: EvalExample, retrieved_chunks: List[str]) -> dict:
    """
    Evaluates a single question's retrieval results against its ground truth.
    Supports Logical OR (multiple acceptable paths) and Logical AND (multi-hop).
    """
    best_rank = float('inf')
    is_hit = False

    # Check each acceptable path (Logical OR)
    for path in example.ground_truth.acceptable_paths:
        # Check if ALL chunks in this path were retrieved (Logical AND / Multi-hop)
        path_ranks = []
        path_satisfied = True
        
        for required_chunk_id in path:
            if required_chunk_id in retrieved_chunks:
                # 1-indexed rank
                rank = retrieved_chunks.index(required_chunk_id) + 1
                path_ranks.append(rank)
            else:
                path_satisfied = False
                break
                
        if path_satisfied:
            is_hit = True
            # The rank of a multi-hop path is determined by the LAST chunk you have to read.
            # However, if a path requires N chunks, the absolute best max rank is N.
            # To avoid penalizing multi-hop queries, we normalize the rank.
            # Effective Rank = Actual Max Rank - N + 1
            # A perfect 3-hop retrieval at ranks 1,2,3 gives Max Rank 3 -> Effective Rank 1 (RR 1.0)
            actual_max_rank = max(path_ranks)
            effective_rank = actual_max_rank - len(path) + 1
            
            if effective_rank < best_rank:
                best_rank = effective_rank

    return {
        "id": example.id,
        "difficulty": example.difficulty_tag,
        "type": example.question_type,
        "hit": is_hit,
        "rank": best_rank if is_hit else None,
        "reciprocal_rank": 1.0 / best_rank if is_hit else 0.0
    }


def main():
    parser = argparse.ArgumentParser(description="Run Retrieval Evaluation")
    parser.add_argument("--eval-file", type=str, default="data/eval/eval_set.jsonl", help="Path to the JSONL eval set")
    parser.add_argument("--k", type=int, default=10, help="Number of chunks to retrieve (Top-K)")
    parser.add_argument("--method", type=str, choices=list(RETRIEVAL_METHODS.keys()), default="dense", 
                        help="The retrieval strategy to evaluate.")
    parser.add_argument("--out-dir", type=str, default="results/week3", help="Directory to save the run JSON")
    parser.add_argument("--rrf-k", type=int, default=None, help="Override the k parameter for Reciprocal Rank Fusion")
    args = parser.parse_args()
    
    if args.rrf_k is not None:
        global GLOBAL_RRF_K_OVERRIDE
        GLOBAL_RRF_K_OVERRIDE = args.rrf_k

    eval_file_path = Path(args.eval_file)
    try:
        examples = load_eval_set(eval_file_path)
    except Exception as e:
        logger.error(f"Failed to load evaluation dataset: {e}")
        sys.exit(1)

    if not examples:
        logger.warning("Evaluation dataset is empty.")
        sys.exit(0)

    conn_str, table_name = load_db_config()
    conn = get_db_connection(conn_str)

    logger.info(f"Running retrieval for {len(examples)} questions (Method: {args.method.upper()}, K={args.k})...")
    
    results = []
    for example in examples:
        retrieved = retrieve_chunks(conn, table_name, example.question, method=args.method, k=args.k)
        result = evaluate_example(example, retrieved)
        results.append(result)
        
    conn.close()

    # Compute aggregate metrics
    total = len(results)
    hits = sum(1 for r in results if r["hit"])
    mrr = sum(r["reciprocal_rank"] for r in results) / total
    recall = (hits / total) * 100

    # Group by difficulty
    diff_stats = {}
    for r in results:
        diff = r["difficulty"]
        if diff not in diff_stats:
            diff_stats[diff] = {"total": 0, "hits": 0, "mrr_sum": 0.0}
        diff_stats[diff]["total"] += 1
        if r["hit"]:
            diff_stats[diff]["hits"] += 1
            diff_stats[diff]["mrr_sum"] += r["reciprocal_rank"]

    # Print Results Table using Rich
    console = Console()
    
    table = Table(title="RAG Retrieval Evaluation Results", show_header=True, header_style="bold magenta")
    table.add_column("Difficulty", style="cyan", width=15)
    table.add_column("Count", justify="right", style="green")
    table.add_column(f"Recall@{args.k}", justify="right", style="yellow")
    table.add_column("MRR", justify="right", style="blue")

    for diff, stats in diff_stats.items():
        d_count = stats["total"]
        d_recall = (stats["hits"] / d_count) * 100
        d_mrr = stats["mrr_sum"] / d_count
        table.add_row(
            diff.upper(), 
            str(d_count), 
            f"{d_recall:.1f}%", 
            f"{d_mrr:.4f}"
        )
    
    # Add a summary row
    table.add_section()
    table.add_row("OVERALL", str(total), f"{recall:.1f}%", f"{mrr:.4f}", style="bold")

    console.print()
    console.print(f"[bold]Retrieval Method:[/bold] {args.method.upper()}")
    console.print(f"[bold]Retrieval Depth:[/bold]  Top-{args.k}")
    console.print(table)
    console.print()

    # Save Results
    results_dir = Path(__file__).parent.parent / args.out_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"eval_run_{timestamp}.json"
    
    run_summary = {
        "timestamp": datetime.now().isoformat(),
        "eval_file": str(eval_file_path),
        "method": args.method,
        "top_k": args.k,
        "total_questions": total,
        "overall_recall": recall,
        "overall_mrr": mrr,
        "difficulty_breakdown": diff_stats,
        "results": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
        
    logger.info(f"💾 Full evaluation run saved to: {output_file}")


if __name__ == "__main__":
    main()
