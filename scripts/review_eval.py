import json
import re


def review_questions():
    with open("data/eval/eval_set.jsonl", "r") as f:
        lines = f.readlines()
        
    # The new 50 are presumably the last 50
    new_lines = lines[-50:]
    
    flagged = []
    
    # Regex to detect code-like artifacts in the question
    # 1. Filenames (e.g., something.py, _base.py)
    file_pattern = re.compile(r'\b\w+\.py\b')
    
    # 2. Fully qualified names or method paths (e.g., LinearModel.fit, module._internal)
    path_pattern = re.compile(r'\b[A-Z]\w*\.[a-z]\w*\b|\b\w+\._\w+\b')
    
    for i, line in enumerate(new_lines):
        obj = json.loads(line)
        q = obj["question"]
        q_id = obj["id"]
        
        reasons = []
        if file_pattern.search(q):
            reasons.append("Contains exact filename (.py)")
        
        if path_pattern.search(q):
            reasons.append("Contains exact Class.method or module._internal path")
            
        if "what arguments are required" in q.lower():
            reasons.append("Formulaic/Templated question structure")
            
        if reasons:
            flagged.append({
                "id": q_id,
                "question": q,
                "reasons": reasons
            })
            
    print(f"Total newly added questions checked: {len(new_lines)}")
    print(f"Flagged for data-leakage/easy-question issues: {len(flagged)}\n")
    
    for f in flagged:
        print(f"[-] ID: {f['id']}")
        print(f"    Q:  {f['question']}")
        print(f"    Flag: {', '.join(f['reasons'])}\n")
        
if __name__ == "__main__":
    review_questions()
