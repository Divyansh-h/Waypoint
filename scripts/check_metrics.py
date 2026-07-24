import glob
import json

for f in glob.glob("results/**/*.json", recursive=True):
    with open(f, 'r') as file:
        try:
            d = json.load(file)
            if "overall_recall" in d:
                print(f"{f}: Recall={d['overall_recall']}, MRR={d.get('overall_mrr')}")
        except: pass
