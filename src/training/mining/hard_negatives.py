import json
from typing import List
from training.schema import TrainingPair, PositiveChunk

import psycopg2

def mine_embedding_neighbors(query: str, true_positive_id: str, k: int = 20) -> List[str]:
    """
    Mines automated hard negatives by querying the pgvector database.
    We use the true_positive's embedding as a proxy to find the hardest structural negatives
    already in the vector space, explicitly excluding the true positive itself.
    """
    conn_str = "postgresql://user:password@localhost:5432/rag_db"
    hard_negatives = []
    
    try:
        conn = psycopg2.connect(conn_str)
        with conn.cursor() as cur:
            sql = """
                SELECT id 
                FROM sklearn_code 
                WHERE id != %s
                ORDER BY embedding <=> (SELECT embedding FROM sklearn_code WHERE id = %s LIMIT 1) 
                LIMIT %s;
            """
            cur.execute(sql, (true_positive_id, true_positive_id, k))
            rows = cur.fetchall()
            hard_negatives = [row[0] for row in rows]
        conn.close()
    except Exception as e:
        print(f"   -> [Mining Error] for chunk {true_positive_id}: {e}")
        
    return hard_negatives


def mine_from_failure_log(eval_results_path: str) -> List[TrainingPair]:
    """
    Parses a Phase 1 evaluation log (JSON) to find queries where the retrieval system failed.
    Extracts the incorrectly highly-ranked chunks and explicitly labels them as hard negatives.
    
    Args:
        eval_results_path: Path to the JSON evaluation log (e.g., compare_run_...json).
        
    Returns:
        A list of highly curated TrainingPairs explicitly targeting system failures.
    """
    curated_pairs: List[TrainingPair] = []
    
    # ---------------------------------------------------------
    # TODO: Implement Failure Log Parsing
    # ---------------------------------------------------------
    # 1. Load the JSON evaluation log from eval_results_path.
    # 2. Iterate through the "results" array (e.g., under the 'dense' or 'hybrid' method).
    # 3. If hit == False (meaning the target chunk was NOT in the top K):
    #    a. The query = the original evaluation query.
    #    b. The true_positive_id = the target chunk that *should* have been retrieved.
    #    c. The hard negatives = the actual top 1-3 chunks the model erroneously returned.
    # 4. Construct a TrainingPair and append to curated_pairs.
    # ---------------------------------------------------------
    
    return curated_pairs
