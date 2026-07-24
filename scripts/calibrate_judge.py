import argparse
import json
import logging
import random
import sys
from pathlib import Path

# Ensure src/ is in the python path
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from rich.console import Console
from rich.prompt import IntPrompt

from eval.judge import score_answer

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    print("scikit-learn is required for Cohen's Kappa. Please install it.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("calibrate_judge")
console = Console()

def load_calibration_set(filepath: str) -> list:
    examples = []
    with open(filepath, "r") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-mock", action="store_true", help="Simulate human and LLM grades.")
    parser.add_argument("--eval-file", type=str, default="data/eval/blind_test_set.jsonl", help="Path to evaluation JSONL")
    args = parser.parse_args()

    console.print("[bold cyan]🚀 Starting LLM-as-Judge Calibration[/bold cyan]\n")
    
    # 1. Load Calibration Examples
    calib_file = args.eval_file
    try:
        examples = load_calibration_set(calib_file)
    except FileNotFoundError:
        logger.error(f"Calibration file {calib_file} not found.")
        sys.exit(1)
        
    console.print(f"Loaded {len(examples)} examples for calibration.\n")
    
    llm_scores = []
    human_scores = []
    
    # 2. Interactive Grading Loop
    for idx, ex in enumerate(examples):
        question = ex.get("question", "Unknown Question")
        
        # MOCK: In reality, we'd retrieve chunks and generate an answer here
        mock_retrieved_chunks = [{"text": "Mock Chunk..."}]
        mock_generated_answer = f"Mock Answer to: {question}"
        
        if args.auto_mock:
            # Simulate a realistic human grade
            h_score = random.choices([5, 4, 3, 2, 0], weights=[0.5, 0.2, 0.1, 0.1, 0.1])[0]
            # Simulate an LLM that agrees ~70% of the time but leans slightly optimistic
            if random.random() < 0.7:
                llm_score = h_score
            else:
                llm_score = min(5, h_score + random.choice([1, 2]))
        else:
            # Get LLM Score (Using the stubbed score_answer in src/eval/judge.py)
            judge_result = score_answer(question, mock_generated_answer, mock_retrieved_chunks)
            llm_score = judge_result.total_score
        
        llm_scores.append(llm_score)
        
        # Prompt Human for their Score
        console.print(f"[bold yellow]--- Example {idx+1}/{len(examples)} ---[/bold yellow]")
        console.print(f"[bold]Question:[/bold] {question}")
        console.print(f"[bold]Generated Answer:[/bold] {mock_generated_answer}")
        console.print(f"\n[italic](LLM judged this internally, yielding a total score of: {llm_score}/5)[/italic]\n")
        
        if args.auto_mock:
            console.print(f"[bold cyan]> Simulated Human Grade:[/bold cyan] {h_score}\n")
        else:
            h_score = IntPrompt.ask("Enter your human grade (0-5) based on the Binary Checklist", choices=[str(i) for i in range(6)])
            
        human_scores.append(h_score)
        console.print("\n")
        
        # Save to human_labels.jsonl
        label_record = {
            "id": ex.get("id", f"eval_{idx}"),
            "question": question,
            "human_score": h_score,
            "llm_score": llm_score,
            "llm_raw_output": "MOCK_OUTPUT_UNTIL_API_WIRED"
        }
        with open("data/eval/human_labels.jsonl", "a") as f:
            f.write(json.dumps(label_record) + "\n")
        
    # 3. Compute Agreement Statistics
    console.print("[bold green]✅ Calibration Complete! Computing Agreement...[/bold green]\n")
    
    # Simple Percent Agreement
    exact_matches = sum(1 for h, l in zip(human_scores, llm_scores) if h == l)
    percent_agreement = (exact_matches / len(examples)) * 100
    
    # Cohen's Kappa
    # We use a linear weight if desired, but default quadratic/unweighted is standard. 
    # For a 0-5 categorical scale, unweighted treats any difference as a total mismatch.
    kappa = cohen_kappa_score(human_scores, llm_scores)
    
    console.print(f"Total Examples Graded: {len(examples)}")
    console.print(f"Simple Percent Agreement: {percent_agreement:.1f}%")
    console.print(f"[bold magenta]Cohen's Kappa (κ): {kappa:.3f}[/bold magenta]\n")
    
    if kappa < 0.60:
        console.print("\n[bold red]🚨 DISAGREEMENT ANALYSIS 🚨[/bold red]")
        console.print("Here are the specific queries where you and the LLM disagreed:\n")
        
        disagreements = 0
        for h, l, ex in zip(human_scores, llm_scores, examples):
            if h != l:
                disagreements += 1
                q_text = ex.get("question", "Unknown")
                console.print(f"[bold red]❌ Disagreement #{disagreements}[/bold red]")
                console.print(f"   [bold]Eval ID:[/bold] {ex.get('id', 'Unknown')}")
                console.print(f"   [bold]Question:[/bold] {q_text}")
                console.print(f"   [bold]Human:[/bold] {h}/5  |  [bold]LLM:[/bold] {l}/5  (Delta: {abs(h-l)})\n")
    
    if kappa < 0.20:
        console.print("[bold red]Verdict: Slight/No Agreement. The LLM is hallucinating scores or heavily biased. You MUST rewrite the prompt.[/bold red]")
    elif kappa < 0.40:
        console.print("[bold red]Verdict: Fair Agreement. The LLM is still failing to capture human nuance.[/bold red]")
    elif kappa < 0.60:
        console.print("[bold yellow]Verdict: Moderate Agreement. Getting closer, but fails on edge cases.[/bold yellow]")
    else:
        console.print("[bold green]Verdict: Substantial Agreement! (κ ≥ 0.60) The LLM Judge is officially calibrated and trustworthy![/bold green]")

if __name__ == "__main__":
    main()
