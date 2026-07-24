import glob
import json

runs = []
for f in glob.glob("results/**/eval_run_*.json", recursive=True):
    with open(f, 'r') as file:
        data = json.load(file)
        runs.append({
            "run_id": f.split('/')[-1].replace('.json', ''),
            "date": data.get("timestamp", "").split('T')[0] if "timestamp" in data else "Unknown",
            "phase": f"Method: {data.get('method', 'Unknown')}",
            "recall_10": data.get("overall_recall", 0.0),
            "mrr_10": data.get("overall_mrr", 0.0),
            "timestamp": data.get("timestamp", "")
        })
runs.sort(key=lambda x: x["timestamp"])
for r in runs: print(r["run_id"], r["date"], r["recall_10"])
