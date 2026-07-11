import argparse
import json
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table

def load_run(path: Path):
    if not path.exists():
        print(f"Error: {path} does not exist.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def format_delta(val, is_pct=False):
    sign = "+" if val > 0 else ""
    color = "green" if val > 0 else "red" if val < 0 else "dim"
    fmt = f"{sign}{val:.2f}%" if is_pct else f"{sign}{val:.4f}"
    # If delta is essentially zero, format as dim neutral
    if abs(val) < 0.0001:
        return f"[dim]0.00{'%' if is_pct else '00'}[/dim]"
    return f"[{color}]{fmt}[/{color}]"

def main():
    parser = argparse.ArgumentParser(description="Compare two evaluation runs to see metrics and per-question deltas.")
    parser.add_argument("run1", type=str, help="Path to the first run JSON (Baseline)")
    parser.add_argument("run2", type=str, help="Path to the second run JSON (Experiment)")
    parser.add_argument("--method", type=str, help="Filter per-question deltas to a specific method (e.g., reranked_hybrid)")
    args = parser.parse_args()

    run1 = load_run(Path(args.run1))
    run2 = load_run(Path(args.run2))

    console = Console()

    methods1 = set(run1.get("methods_compared", []))
    methods2 = set(run2.get("methods_compared", []))
    common_methods = sorted(list(methods1.intersection(methods2)))

    if not common_methods:
        console.print("[bold red]No common methods found between the two runs.[/bold red]")
        sys.exit(1)

    console.print(f"[bold]Comparing Runs:[/bold]")
    console.print(f"Run 1 (Baseline):   {args.run1}")
    console.print(f"Run 2 (Experiment): {args.run2}\n")

    # --- Macro Metrics Table ---
    table = Table(title="Aggregate Metrics Delta (Run 2 - Run 1)", show_header=True, header_style="bold magenta")
    table.add_column("Method", style="cyan")
    table.add_column("Recall@5 Delta", justify="right")
    table.add_column("Recall@10 Delta", justify="right")
    table.add_column("MRR Delta", justify="right")
    table.add_column("P50 Lat Delta", justify="right")
    table.add_column("P95 Lat Delta", justify="right")

    for m in common_methods:
        s1 = run1["comparison_stats"][m]
        s2 = run2["comparison_stats"][m]
        
        d_r5 = s2["recall_5"] - s1["recall_5"]
        d_r10 = s2["recall_10"] - s1["recall_10"]
        d_mrr = s2["mrr"] - s1["mrr"]
        
        # We might not have latency if run1 is old
        d_p50 = s2.get("p50_latency_ms", 0) - s1.get("p50_latency_ms", 0)
        d_p95 = s2.get("p95_latency_ms", 0) - s1.get("p95_latency_ms", 0)
        
        def format_lat_delta(val):
            sign = "+" if val > 0 else ""
            color = "red" if val > 0 else "green" if val < 0 else "dim" # For latency, less is better
            if abs(val) < 0.1:
                return "[dim]0.0ms[/dim]"
            return f"[{color}]{sign}{val:.1f}ms[/{color}]"
        
        table.add_row(
            m.upper(),
            format_delta(d_r5, True),
            format_delta(d_r10, True),
            format_delta(d_mrr, False),
            format_lat_delta(d_p50),
            format_lat_delta(d_p95)
        )

    console.print(table)
    console.print()

    # --- Per-Question Delta ---
    target_methods = [args.method] if args.method else common_methods
    
    for m in target_methods:
        if m not in common_methods:
            console.print(f"[bold red]Method '{m}' not found in both runs.[/bold red]")
            continue
            
        r1_q = {q["id"]: q for q in run1["raw_results_per_method"][m]}
        r2_q = {q["id"]: q for q in run2["raw_results_per_method"][m]}
        
        common_qs = set(r1_q.keys()).intersection(set(r2_q.keys()))
        
        gained = []
        lost = []
        rank_improved = []
        rank_worsened = []
        
        for qid in common_qs:
            q1 = r1_q[qid]
            q2 = r2_q[qid]
            
            # Binary Hit Changes
            if not q1["hit"] and q2["hit"]:
                gained.append(q2)
            elif q1["hit"] and not q2["hit"]:
                lost.append(q2)
            # Rank Shifts (only if both were hits)
            elif q1["hit"] and q2["hit"]:
                if q2["rank"] < q1["rank"]:
                    rank_improved.append((q2, q1["rank"], q2["rank"]))
                elif q2["rank"] > q1["rank"]:
                    rank_worsened.append((q2, q1["rank"], q2["rank"]))
                    
        # Print if there were any changes
        if gained or lost or rank_improved or rank_worsened:
            console.print(f"[bold underline]Per-Question Changes for {m.upper()}:[/bold underline]")
            
            if gained:
                console.print(f"[bold green]Gained Hits (Was MISS, now HIT): {len(gained)}[/bold green]")
                for q in gained:
                    console.print(f"  + {q['id']} (Rank {q['rank']})")
                    
            if lost:
                console.print(f"[bold red]Lost Hits (Was HIT, now MISS): {len(lost)}[/bold red]")
                for q in lost:
                    console.print(f"  - {q['id']}")
                    
            if rank_improved:
                console.print(f"[bold green]Rank Improved: {len(rank_improved)}[/bold green]")
                for q, old, new in rank_improved:
                    console.print(f"  ^ {q['id']} (Rank {old} -> {new})")
                    
            if rank_worsened:
                console.print(f"[bold yellow]Rank Worsened: {len(rank_worsened)}[/bold yellow]")
                for q, old, new in rank_worsened:
                    console.print(f"  v {q['id']} (Rank {old} -> {new})")
                    
            console.print()
        else:
            if args.method:
                console.print(f"[dim]No per-question changes detected for {m.upper()} between the two runs.[/dim]\n")

if __name__ == "__main__":
    main()
