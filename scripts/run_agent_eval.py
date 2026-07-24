import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Ensure src/ is in the python path
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dotenv import load_dotenv

from agent.orchestrator import AgentOrchestrator
from eval.loader import load_eval_set

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("run_agent_eval")

def evaluate_agent_hit(ground_truth_ids: list, context_used_ids: list) -> bool:
    """
    Checks if the Agent successfully retrieved and utilized all required chunks.
    For multi-hop queries, it must find ALL ground truth IDs.
    """
    if not ground_truth_ids:
        return False
    return all(gt_id in context_used_ids for gt_id in ground_truth_ids)

def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 Agent Evaluation")
    parser.add_argument("--eval-file", type=str, default="data/eval/eval_set.jsonl")
    parser.add_argument("--out-dir", type=str, default="results/phase3")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to evaluate")
    args = parser.parse_args()
    
    eval_file_path = Path(args.eval_file)
    try:
        examples = load_eval_set(eval_file_path)
    except Exception as e:
        logger.error(f"Failed to load evaluation dataset: {e}")
        sys.exit(1)

    if args.limit:
        examples = examples[:args.limit]

    logger.info("🤖 Initializing Agent Orchestrator...")
    orchestrator = AgentOrchestrator()
    
    results = []
    total_hits = 0
    
    logger.info(f"🚀 Running full agent eval on {len(examples)} questions...")
    
    for i, example in enumerate(examples):
        logger.info(f"--- Q{i+1}: {example.question} ---")
        
        # Execute the full Agent loop
        agent_result = orchestrator.run(example.question)
        
        # Evaluate if the final gathered context contains the ground truth
        # context_used is a Dict[snippet_id, raw_code_content]
        # The keys aren't the exact DB IDs anymore since they are formatted as "step_X_tool"
        # However, we can check if the ground truth IDs were successfully passed to the LLM during the process.
        # Let's extract the actual snippet IDs from the context values if possible, 
        # or we just rely on a text substring match for now since the agent formats it as "--- SNIPPET ID: X ---".
        
        is_hit = False
        context_string = "\n".join(agent_result.context_used.values())
        if all(f"SNIPPET ID: {gt_id}" in context_string for gt_id in example.ground_truth_chunk_ids):
            is_hit = True
            total_hits += 1
            
        results.append({
            "id": example.id,
            "difficulty": example.difficulty_tag,
            "type": example.question_type,
            "hit": is_hit,
            "success": agent_result.success,
            "termination_reason": agent_result.termination_reason,
            "agent_iterations": agent_result.iterations
        })
        
    recall = (total_hits / len(examples)) * 100
    
    logger.info("\n====================================")
    logger.info("✅ Agent Eval Stub Complete")
    logger.info(f"📊 Simulated 1-Step Recall: {recall:.2f}%")
    logger.info("====================================\n")
    
    # Save Results
    results_dir = Path(__file__).parent.parent / args.out_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"agent_eval_run_{timestamp}.json"
    
    run_summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": "full_agent_loop",
        "total_questions": len(examples),
        "overall_recall": recall,
        "results": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
        
    logger.info(f"💾 Eval trace saved to: {output_file}")

if __name__ == "__main__":
    main()
