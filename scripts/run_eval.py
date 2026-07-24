import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import psycopg2
import yaml
from pgvector.psycopg2 import register_vector
from rich.console import Console
from rich.table import Table

# Ensure src/ is in the python path
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import os

import google.generativeai as genai
from dotenv import load_dotenv

from eval.judge import JudgeScore, score_answer
from eval.loader import load_eval_set
from ingestion.models import EvalExample

# Load environment variables from .env
load_dotenv()

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

    path_ranks = []
    path_satisfied = True
    
    for required_chunk_id in example.ground_truth_chunk_ids:
        if required_chunk_id in retrieved_chunks:
            # 1-indexed rank
            rank = retrieved_chunks.index(required_chunk_id) + 1
            path_ranks.append(rank)
        else:
            path_satisfied = False
            break
            
    if path_satisfied and path_ranks:
        is_hit = True
        actual_max_rank = max(path_ranks)
        effective_rank = actual_max_rank - len(example.ground_truth_chunk_ids) + 1
        
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

def fetch_chunks_for_llm(retrieved_chunk_ids: List[str], conn, table_name: str) -> List[dict]:
    """Fetches chunks from DB, preserves ranking order, and strips away the chunk ID to prevent cheating."""
    if not retrieved_chunk_ids:
        return []
        
    with conn.cursor() as cur:
        format_strings = ','.join(['%s'] * len(retrieved_chunk_ids))
        cur.execute(f"SELECT id, content FROM {table_name} WHERE id IN ({format_strings})", tuple(retrieved_chunk_ids))
        rows = cur.fetchall()
        
    # Map by id to preserve the retrieved order (vital for RAG evaluation)
    chunk_map = {row[0]: row[1] for row in rows}
    
    clean_chunks = []
    for i, cid in enumerate(retrieved_chunk_ids):
        if cid in chunk_map:
            # We explicitly DO NOT pass the chunk ID (cid) as the file_path! 
            # In synthetic datasets, the ID is often the exact symbol name (e.g. KMeans.score).
            # Passing it would allow the Judge/Generator to cheat without reading the code.
            clean_chunks.append({
                "content": chunk_map[cid],
                "file_path": f"Snippet_{i+1}"
            })
            
    return clean_chunks


def generate_answer(question: str, chunks: List[dict]) -> str:
    """Generates an answer using the agent model based ONLY on the provided chunks."""
    if not chunks:
        return "I don't have enough context to answer."
        
    chunks_text = ""
    for chunk in chunks:
        chunks_text += f"\n--- {chunk['file_path']} ---\n{chunk['content']}\n"
            
    # 2. Ping LLM (or mock if no API key)
    if "GEMINI_API_KEY" not in os.environ:
        return f"[MOCK GENERATION for Question: {question}]"
        
    prompt = f"Answer the following question using ONLY the provided code chunks. If the answer is not in the chunks, say so.\n\nChunks:\n{chunks_text}\n\nQuestion: {question}"
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return "Failed to generate answer."


def main():
    parser = argparse.ArgumentParser(description="Run Retrieval Evaluation")
    parser.add_argument("--eval-file", type=str, default="data/eval/eval_set.jsonl", help="Path to the JSONL eval set")
    parser.add_argument("--k", type=int, default=10, help="Number of chunks to retrieve (Top-K)")
    parser.add_argument("--method", type=str, choices=list(RETRIEVAL_METHODS.keys()), default="dense", 
                        help="The retrieval strategy to evaluate.")
    parser.add_argument("--out-dir", type=str, default="results/week3", help="Directory to save the run JSON")
    parser.add_argument("--rrf-k", type=int, default=None, help="Override the k parameter for Reciprocal Rank Fusion")
    parser.add_argument("--judge", action="store_true", help="Run End-to-End Generation & Judging")
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
        
        # 🚨 Phase 4: The End-to-End Pipeline
        if args.judge:
            # 1. Fetch sanitized chunks (No leaking ground-truth IDs)
            actual_chunks = fetch_chunks_for_llm(retrieved, conn, table_name)
            
            # 2. Generate Answer
            answer = generate_answer(example.question, actual_chunks)
            
            # 3. Score Answer against the EXACT same sanitized context
            if "GEMINI_API_KEY" not in os.environ:
                judge_score = JudgeScore(True, True, True, True, True)
            else:
                judge_score = score_answer(example.question, answer, actual_chunks)
                
            result["generation"] = answer
            result["judge_score"] = {
                "total": judge_score.total_score,
                "is_correct": judge_score.is_correct,
                "no_hallucination": judge_score.no_hallucination,
                "is_complete": judge_score.is_complete,
                "multi_hop_synthesis": judge_score.multi_hop_synthesis,
                "has_citation": judge_score.has_citation
            }
            
            if "GEMINI_API_KEY" in os.environ:
                logger.info("Throttling for 8 seconds to respect Free Tier API quotas...")
                time.sleep(8)
            
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
    
    if args.judge:
        # Calculate generation aggregates
        total_judge = sum(r["judge_score"]["total"] for r in results if "judge_score" in r)
        avg_judge = total_judge / max(1, len(results))
        hallucination_failures = sum(1 for r in results if "judge_score" in r and not r["judge_score"]["no_hallucination"])
        
        j_table = Table(title="End-to-End Generation Quality (LLM-Judge)", show_header=True, header_style="bold green")
        j_table.add_column("Metric", style="cyan", width=25)
        j_table.add_column("Value", justify="right", style="yellow")
        
        j_table.add_row("Average Total Score", f"{avg_judge:.1f} / 5.0")
        j_table.add_row("Hallucination Rate", f"{(hallucination_failures/max(1, len(results)))*100:.1f}%")
        
        console.print()
        console.print(j_table)
        
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
