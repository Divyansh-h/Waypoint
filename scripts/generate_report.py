import argparse
import json
from pathlib import Path


def generate_markdown(json_path: Path, output_path: Path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    methods = data.get("methods_compared", [])
    if not methods:
        # Fallback for older run_eval.py single-method JSONs
        method = data.get("method", "dense")
        methods = [method]
        data["comparison_stats"] = {method: {
            "recall_5": data.get("overall_recall", 0), # mock
            "recall_10": data.get("overall_recall", 0),
            "mrr": data.get("overall_mrr", 0)
        }}
        data["raw_results_per_method"] = {method: data.get("results", [])}

    md = [
        "# RAG Evaluation Report",
        f"**Run Timestamp:** {data.get('timestamp')}",
        f"**Evaluation File:** {data.get('eval_file')}",
        "\n## 1. Metrics Comparison\n",
        "| Method | Recall@5 | Recall@10 | MRR |",
        "|---|---|---|---|"
    ]
    
    for m in methods:
        stats = data["comparison_stats"][m]
        r5 = stats.get("recall_5", 0)
        r10 = stats.get("recall_10", 0)
        mrr = stats.get("mrr", 0)
        md.append(f"| {m.upper()} | {r5:.1f}% | {r10:.1f}% | {mrr:.4f} |")
        
    md.append("\n## 2. Failure Clustering (Misses @ K=10)\n")
    
    for m in methods:
        results = data["raw_results_per_method"][m]
        misses = [r for r in results if not r["hit"]]
        md.append(f"### {m.upper()} Failures: {len(misses)}")
        
        if misses:
            diff_counts = {}
            type_counts = {}
            for r in misses:
                d = r.get("difficulty", "unknown").upper()
                t = r.get("type", "unknown").upper()
                diff_counts[d] = diff_counts.get(d, 0) + 1
                type_counts[t] = type_counts.get(t, 0) + 1
                
            md.append("**By Difficulty:** " + ", ".join(f"{k}: {v}" for k, v in sorted(diff_counts.items())))
            md.append("\n**By Type:** " + ", ".join(f"{k}: {v}" for k, v in sorted(type_counts.items())) + "\n")
        else:
            md.append("*No misses!* \n")

    md.append("\n## 3. Worst 5 Questions by Rank\n")
    
    for m in methods:
        results = data["raw_results_per_method"][m]
        # Sort criteria: misses first (rank is None), then highest rank number
        def sort_key(r):
            if not r["hit"] or r["rank"] is None:
                return 999999
            return r["rank"]
            
        worst = sorted(results, key=sort_key, reverse=True)[:5]
        md.append(f"### {m.upper()} Worst 5")
        if not worst:
            md.append("*No data.*\n")
            continue
            
        md.append("| Question ID | Difficulty | Type | Status | Rank |")
        md.append("|---|---|---|---|---|")
        for w in worst:
            status = "❌ MISS" if not w["hit"] else "✅ HIT"
            rank = w["rank"] if w["hit"] else "N/A"
            md.append(f"| {w['id']} | {w.get('difficulty', '').upper()} | {w.get('type', '').upper()} | {status} | {rank} |")
        md.append("\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
        
    print(f"Report generated successfully at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the JSON results file")
    parser.add_argument("--output", type=str, required=True, help="Path to output markdown file")
    args = parser.parse_args()
    
    generate_markdown(Path(args.input), Path(args.output))
