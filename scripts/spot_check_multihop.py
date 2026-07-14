import json

count = 0
with open('data/eval/eval_set.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data.get('question_type') == 'multi_hop' or 'multi_hop' in str(data):
            count += 1
            print(f"--- Example {count} ---")
            print(f"Q: {data['question']}")
            print(f"Ground Truth Chunks: {data['ground_truth_chunk_ids']}")
            print("-" * 50)
            if count >= 5:
                break
