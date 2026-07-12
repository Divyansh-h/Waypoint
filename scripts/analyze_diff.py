import json

with open("data/eval/eval_set.jsonl", "r") as f:
    questions = {}
    for line in f:
        obj = json.loads(line)
        questions[obj["id"]] = obj["question"]

with open("results/week3/compare_run_20260712_131625.json", "r") as f:
    data = json.load(f)

baseline = data["raw_results_per_method"]["hybrid"]
reranked = data["raw_results_per_method"]["reranked_hybrid"]

b_dict = {item["id"]: item["reciprocal_rank"] for item in baseline}
r_dict = {item["id"]: item["reciprocal_rank"] for item in reranked}

improved = []
degraded = []

for q_id, b_mrr in b_dict.items():
    r_mrr = r_dict.get(q_id, 0.0)
    q_text = questions.get(q_id, "Unknown question")
    
    if r_mrr > b_mrr:
        improved.append({"q": q_text, "b_mrr": b_mrr, "r_mrr": r_mrr})
    elif r_mrr < b_mrr:
        degraded.append({"q": q_text, "b_mrr": b_mrr, "r_mrr": r_mrr})

print("=== IMPROVED BY RERANKER ===")
for i, item in enumerate(improved[:3]):
    print(f"[+] Q: {item['q']}\n    Hybrid MRR: {item['b_mrr']:.2f} -> Reranked MRR: {item['r_mrr']:.2f}\n")
print(f"Total Improved: {len(improved)}")

print("\n=== DEGRADED BY RERANKER ===")
for i, item in enumerate(degraded[:3]):
    print(f"[-] Q: {item['q']}\n    Hybrid MRR: {item['b_mrr']:.2f} -> Reranked MRR: {item['r_mrr']:.2f}\n")
print(f"Total Degraded: {len(degraded)}")
