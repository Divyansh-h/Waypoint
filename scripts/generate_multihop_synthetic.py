import argparse
import ast
import json
import logging
import os
import sys
import time
import random
from typing import List, Dict, Tuple
from dotenv import load_dotenv

import psycopg2
import psycopg2.extras
import google.generativeai as genai

# Ensure src/ is in the python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from ingestion.models import EvalExample

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("multihop_gen")

def extract_function_calls(code: str) -> List[str]:
    """Uses Python's AST to extract all function/method names called within the code."""
    calls = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)
    except SyntaxError:
        # Some chunks might not be perfectly valid isolated Python modules
        pass
    return list(set(calls))

def get_ast_linked_pairs(limit: int = 150) -> List[Tuple[dict, dict]]:
    """
    Finds real AST dependencies by extracting function calls from a 'Caller' chunk
    and finding the corresponding 'Callee' chunk in the database.
    """
    logger.info("Connecting to database to extract AST-linked chunk pairs...")
    conn_str = "postgresql://user:password@localhost:5432/rag_db"
    
    try:
        conn = psycopg2.connect(conn_str)
    except Exception as e:
        logger.error(f"Failed to connect to DB: {e}")
        return []

    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        # Fetch a pool of functions/methods
        cur.execute("SELECT id, file_path, chunk_type, content, function_name FROM sklearn_code WHERE chunk_type IN ('function', 'method');")
        all_chunks = [dict(row) for row in cur.fetchall()]
        
    conn.close()
    
    # Map chunks by their function name for fast callee lookup
    # We only map chunks that have a defined function_name
    callee_map = {}
    for chunk in all_chunks:
        fname = chunk.get("function_name")
        if fname:
            # Simple collision handling: just keep the first one or a list. We'll keep a list.
            if fname not in callee_map:
                callee_map[fname] = []
            callee_map[fname].append(chunk)

    pairs = []
    # Find dependencies
    for caller in all_chunks:
        calls = extract_function_calls(caller["content"])
        for call in calls:
            # Prevent self-loops
            if call == caller.get("function_name"):
                continue
                
            if call in callee_map:
                for callee in callee_map[call]:
                    # Ensure they are different chunks
                    if caller["id"] != callee["id"]:
                        pairs.append((caller, callee))
                        
    # Shuffle and limit
    random.seed(42)
    random.shuffle(pairs)
    logger.info(f"Found {len(pairs)} AST-linked pairs in the codebase.")
    return pairs[:limit]


def generate_multihop_question(caller: dict, callee: dict, model) -> dict:
    """Uses the Gemini LLM to write a complex multi-hop question requiring both chunks."""
    prompt = f"""You are an expert Python engineer and dataset creator.
I will give you two code chunks that have an AST dependency (Chunk A calls Chunk B).
Your job is to write a realistic, highly technical, and complex "Multi-Hop" question that a developer would ask.
Crucially, the question MUST require reading and understanding BOTH chunks to answer correctly. 

Chunk A (Caller: {caller.get('function_name')}):
```python
{caller['content']}
```

Chunk B (Callee: {callee.get('function_name')}):
```python
{callee['content']}
```

Output ONLY a JSON object with this exact schema:
{{
  "reasoning": "Explain why this question strictly requires both chunks to answer.",
  "question": "The actual technical question string."
}}
"""
    try:
        resp = model.generate_content(prompt)
        text = resp.text.strip()
        if text.startswith("```json"): text = text[7:]
        if text.startswith("```"): text = text[3:]
        if text.endswith("```"): text = text[:-3]
        return json.loads(text.strip())
    except Exception as e:
        logger.error(f"Failed to generate/parse LLM response: {e}")
        return None


def main():
    load_dotenv()
    if "GEMINI_API_KEY" not in os.environ:
        logger.error("GEMINI_API_KEY is required.")
        sys.exit(1)

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    
    # We use a lower temperature for consistent formatting, but slight variance for creativity
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=genai.GenerationConfig(
            temperature=0.4,
            response_mime_type="application/json"
        )
    )

    TARGET_COUNT = 150
    pairs = get_ast_linked_pairs(limit=TARGET_COUNT * 2) # Get extra in case of LLM failures
    
    if not pairs:
        logger.error("No valid AST pairs found.")
        sys.exit(1)

    output_file = "data/eval/synthetic_multihop.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    generated_count = 0
    logger.info(f"Generating ~{TARGET_COUNT} questions. This will take a few minutes...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for caller, callee in pairs:
            if generated_count >= TARGET_COUNT:
                break
                
            result = generate_multihop_question(caller, callee, model)
            if not result or "question" not in result:
                continue
                
            example_id = f"eval_mh_{generated_count:03d}"
            
            # Create EvalExample model
            example = EvalExample(
                id=example_id,
                question=result["question"],
                ground_truth_chunk_ids=[caller["id"], callee["id"]],
                difficulty_tag="hard",
                question_type="multi_hop",
                metadata={
                    "caller_function": caller.get("function_name"),
                    "callee_function": callee.get("function_name"),
                    "reasoning": result.get("reasoning", "")
                }
            )
            
            f.write(example.model_dump_json() + "\n")
            f.flush()
            generated_count += 1
            
            if generated_count % 10 == 0:
                logger.info(f"Generated {generated_count}/{TARGET_COUNT} questions...")
                
            # Respect Free-Tier RPM (15 RPM -> 4 seconds per request)
            time.sleep(4.1)

    logger.info(f"✅ Successfully generated {generated_count} Multi-Hop Eval questions.")
    logger.info(f"Saved to: {output_file}")
    logger.info("Next Step: Run `cat data/eval/synthetic_multihop.jsonl >> data/eval/eval_set.jsonl` to backfill the master dataset.")

if __name__ == "__main__":
    main()
