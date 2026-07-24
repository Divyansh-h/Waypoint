import json

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")

# Load data
with open("results/week3/compare_run_20260712_131625.json", "r") as f:
    data = json.load(f)

raw_results = data["raw_results_per_method"]
methods = ["dense", "bm25", "hybrid", "reranked_hybrid"]
method_labels = {"dense": "Dense", "bm25": "BM25", "hybrid": "Hybrid (RRF)", "reranked_hybrid": "Hybrid + Reranker"}
k_values = [1, 3, 5, 10]

# Compute recall@k
recall_data = {m: [] for m in methods}

for method in methods:
    results = raw_results[method]
    total_q = len(results)
    
    for k in k_values:
        # A hit at rank <= k counts
        hits = sum(1 for item in results if item["hit"] and item["rank"] is not None and item["rank"] <= k)
        recall = (hits / total_q) * 100
        recall_data[method].append(recall)

# Plotting
plt.figure(figsize=(10, 6))

colors = {"dense": "#3498db", "bm25": "#e74c3c", "hybrid": "#2ecc71", "reranked_hybrid": "#9b59b6"}
markers = {"dense": "o", "bm25": "s", "hybrid": "^", "reranked_hybrid": "D"}

for method in methods:
    plt.plot(k_values, recall_data[method], 
             label=method_labels[method], 
             color=colors[method], 
             marker=markers[method],
             linewidth=2.5,
             markersize=8)

plt.title("Recall@K Comparison (Phase 1 Benchmark)", fontsize=16, pad=15)
plt.xlabel("Top-K Retrieved Chunks", fontsize=12)
plt.ylabel("Recall (%)", fontsize=12)
plt.xticks(k_values)
plt.yticks(range(0, 51, 10))
plt.ylim(-2, 50)  # BM25 is at 0, max is 44, so 0-50 is good
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)

# Save to artifact dir
output_path = "/Users/divyansh/.gemini/antigravity-cli/brain/1d0e9281-187b-4c9c-a192-88eb3cbc23f7/recall_curve.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Chart saved to {output_path}")
