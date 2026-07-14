import json

count = 0
with open('data/eval/eval_set.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if len(data.get('ground_truth_chunk_ids', [])) > 1:
            count += 1
            print(f"--- Example {count} ---")
            print(f"Q: {data['question']}")
            print(f"Type: {data.get('question_type')} | Diff: {data.get('difficulty_tag')}")
            print(f"Ground Truth Chunks: {data['ground_truth_chunk_ids']}")
            print("-" * 50)
            if count >= 5:
                break
