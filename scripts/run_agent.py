import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is in the python path
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from agent.orchestrator import AgentOrchestrator
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Configure standard logging to output the Orchestrator's internal step-by-step trace
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("run_agent")

def main():
    parser = argparse.ArgumentParser(description="Run the Phase 3 RAG Agent on a specific query.")
    parser.add_argument("question", type=str, help="The question to ask the Agent.")
    parser.add_argument("--max-steps", type=int, default=6, help="Hard kill-switch limit for routing iterations.")
    
    args = parser.parse_args()
    console = Console()
    
    console.print(Panel.fit(f"[bold cyan]Question:[/bold cyan] {args.question}", title="Phase 3 Agent CLI"))
    
    orchestrator = AgentOrchestrator(max_iterations=args.max_steps)
    
    try:
        # Currently, the orchestrator is hardcoded with a stub pass-through 
        # (PLANNING -> RETRIEVING -> SYNTHESIZING) for testing.
        result = orchestrator.run(args.question)
        
        console.print("\n[bold green]Agent Execution Complete![/bold green]")
        
        # Display the final answer
        console.print(Panel(Markdown(result.answer), title="Final Synthesized Answer", border_style="green"))
        
        # Display metadata
        console.print(f"[bold]Total Iterations:[/bold] {result.iterations} / {args.max_steps}")
        console.print(f"[bold]Success Status:[/bold] {'✅' if result.success else '❌'}")
        console.print(f"[bold]Chunks Utilized:[/bold] {list(result.context_used.keys())}")
        
    except KeyboardInterrupt:
        console.print("\n[bold red]Execution interrupted by user.[/bold red]")
        sys.exit(1)
    except Exception as e:
        logger.exception("A fatal error occurred during Agent execution.")
        sys.exit(1)

if __name__ == "__main__":
    main()
