import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

# Ensure src/ and root are in the python path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))

from eval.loader import load_eval_set
from eval.metrics import cluster_failures
from scripts.run_eval import load_db_config, get_db_connection, retrieve_chunks, evaluate_example, RETRIEVAL_METHODS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("compare_methods")


def evaluate_method(examples, conn, table_name, method, verbose=False):
    results_k5 = []
    results_k10 = []
    
    for example in examples:
        # Retrieve top 10 for the method
        retrieved_10 = retrieve_chunks(conn, table_name, example.question, method=method, k=10)
        
        # Evaluate for K=5
        res_5 = evaluate_example(example, retrieved_10[:5])
        results_k5.append(res_5)
        
        # Evaluate for K=10
        res_10 = evaluate_example(example, retrieved_10[:10])
        results_k10.append(res_10)
        
        if verbose:
            logger.info(f"Q: {example.question}")
            logger.info(f"  Ground Truth: {example.ground_truth.acceptable_paths}")
            logger.info(f"  Retrieved (Top 10): {retrieved_10}")
            hit_status = f"✅ HIT at Rank {res_10['rank']}" if res_10['hit'] else "❌ MISS"
            logger.info(f"  Status: {hit_status}\n")
        
    # Compute aggregate metrics
    total = len(examples)
    
    hits_5 = sum(1 for r in results_k5 if r["hit"])
    recall_5 = (hits_5 / total) * 100
    
    hits_10 = sum(1 for r in results_k10 if r["hit"])
    recall_10 = (hits_10 / total) * 100
    
    # MRR is typically computed over the full retrieved list (K=10)
    mrr = sum(r["reciprocal_rank"] for r in results_k10) / total
    
    return {
        "recall_5": recall_5,
        "recall_10": recall_10,
        "mrr": mrr,
        "raw_results": results_k10
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Retrieval Methods")
    parser.add_argument("--eval-file", type=str, default="data/eval/eval_set.jsonl")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print per-question retrieval details")
    args = parser.parse_args()

    eval_file_path = Path(args.eval_file)
    try:
        examples = load_eval_set(eval_file_path)
    except Exception as e:
        logger.error(f"Failed to load evaluation dataset: {e}")
        sys.exit(1)

    if not examples:
        logger.warning("Evaluation dataset is empty.")
        sys.exit(0)

    # Sanity check for the user
    if len(examples) < 50:
        logger.warning(f"⚠️ Note: You only have {len(examples)} question(s) in {eval_file_path.name}.")
        logger.warning("You mentioned ~50 earlier. Please populate the JSONL file to get statistically significant results!\n")

    conn_str, table_name = load_db_config()
    conn = get_db_connection(conn_str)
    
    methods = list(RETRIEVAL_METHODS.keys())
    comparison = {}
    
    console = Console()
    with console.status("[bold green]Running retrieval across all presets..."):
        for method in methods:
            if args.verbose:
                logger.info(f"\n{'='*50}\nEvaluating method: {method.upper()}\n{'='*50}")
            else:
                logger.info(f"Evaluating method: {method.upper()}...")
            comparison[method] = evaluate_method(examples, conn, table_name, method, verbose=args.verbose)

    conn.close()

    # Print Comparison Table
    table = Table(title="RAG Retrieval Preset Comparison", show_header=True, header_style="bold magenta")
    table.add_column("Method", style="cyan", width=15)
    table.add_column("Recall@5", justify="right", style="yellow")
    table.add_column("Recall@10", justify="right", style="green")
    table.add_column("MRR (Top 10)", justify="right", style="blue")

    for method in methods:
        stats = comparison[method]
        table.add_row(
            method.upper(), 
            f"{stats['recall_5']:.1f}%", 
            f"{stats['recall_10']:.1f}%", 
            f"{stats['mrr']:.4f}"
        )

    console.print()
    console.print(table)
    console.print()

    # --- Failure Clustering Helper ---
    has_misses = False
    for method in methods:
        misses = [r for r in comparison[method]["raw_results"] if not r["hit"]]
        if misses:
            has_misses = True
            break
            
    if has_misses:
        console.print("[bold red]Failure Clustering Analysis (Misses @ K=10)[/bold red]")
        for method in methods:
            misses = [r for r in comparison[method]["raw_results"] if not r["hit"]]
            if not misses:
                console.print(f"[bold]{method.upper()} Misses:[/bold] 0")
                continue
                
            diff_counts, type_counts = cluster_failures(misses)
                
            console.print(f"[bold]{method.upper()} Misses ({len(misses)} total):[/bold]")
            
            diff_str = ", ".join(f"{k}: {v}" for k, v in sorted(diff_counts.items()))
            type_str = ", ".join(f"{k}: {v}" for k, v in sorted(type_counts.items()))
            
            console.print(f"  By Difficulty: {diff_str}")
            console.print(f"  By Type:       {type_str}\n")
    else:
        console.print("[bold green]No misses detected! All questions answered correctly in top 10.[/bold green]\n")

    # Save detailed JSON output for manual inspection of mismatches
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"compare_run_{timestamp}.json"
    
    export_payload = {
        "timestamp": datetime.now().isoformat(),
        "eval_file": str(eval_file_path),
        "methods_compared": methods,
        "comparison_stats": {
            m: {k: v for k, v in stats.items() if k != "raw_results"} 
            for m, stats in comparison.items()
        },
        "raw_results_per_method": {
            m: stats["raw_results"] for m, stats in comparison.items()
        }
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)
        
    logger.info(f"💾 Full comparison run with raw per-question results saved to: {output_file}")


if __name__ == "__main__":
    main()
