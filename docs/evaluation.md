# RAG Evaluation Framework

This document outlines how the evaluation set for this RAG pipeline is constructed, its target composition, and the workflow for adding and verifying new evaluation questions.

---

## 1. How the Eval Set is Built

We evaluate retrieval quality using a highly curated, offline JSONL dataset containing pairs of human-written questions and their exact "ground-truth" chunk IDs.

Unlike synthetic datasets generated purely by LLMs, this dataset is hand-curated to ensure realism. The target chunks refer directly to the UUIDs stored in the PostgreSQL `chunks` table.

### Ground Truth Logic (OR vs AND)
Our evaluation metrics engine supports complex ground-truth schemas:
*   **Logical OR (Multiple Acceptable Paths):** If an answer can be found in `chunk_A` OR `chunk_B`, finding *either* one counts as a 100% success.
*   **Logical AND (Multi-hop Queries):** If answering a question requires aggregating context from `chunk_C` AND `chunk_D`, the system must retrieve *both* chunks. Our harness uses an `Effective Rank` formula (`Actual Max Rank - N + 1`) to ensure multi-hop queries aren't mathematically penalized in Mean Reciprocal Rank (MRR).

---

## 2. Target Composition

The dataset is designed to contain ~50 questions, strictly balanced to stress-test different retrieval methods (BM25, Dense, Hybrid).

*   **30% Factual / Easy**
    *   *Examples:* "Where is `KMeans` defined?", "What are the parameters for `fit_transform`?"
    *   *Purpose:* Tests keyword matching. BM25 generally wins these if the vocabulary matches exactly.
*   **30% Conceptual / Medium**
    *   *Examples:* "Why does Ridge regression penalize complexity?", "How is the learning rate decayed?"
    *   *Purpose:* Tests semantic overlap. Dense vectors win here by bridging the vocabulary gap between the prompt and the code.
*   **40% Multi-hop / Hard**
    *   *Examples:* "Trace the execution flow when a ValueError is raised during GridSearchCV fitting."
    *   *Purpose:* Stress-tests Hybrid methods and Reciprocal Rank Fusion (RRF). Can the system pull 3 scattered chunks without letting one fall out of the Top-10?

*(Note: As of now, the `eval_set.jsonl` contains placeholder structural data pending the final execution of the ingestion pipeline).*

---

## 3. How to Add New Examples

Adding new questions to the evaluation set is done interactively via the command line.

### Step 1: Run the Interactive Curation Script
Run the helper script to add a question to your draft JSONL:
```bash
python scripts/add_eval_example.py
```
You will be prompted to enter:
1. The question text.
2. The required Chunk IDs (space-separated for multi-hop AND, comma-separated for alternative OR paths).
3. Difficulty (`easy`, `medium`, `hard`).
4. Question Type (`factual`, `conceptual`).

### Step 2: Lint for "Leaky" Questions
We want to avoid questions that are trivially easy (e.g., asking "What does the _validate_data function do?" when the target chunk is named `_validate_data`). 

Run our strict anti-cheat script:
```bash
python scripts/flag_easy_questions.py --eval-file data/eval/eval_set_draft.jsonl
```
This script queries the Postgres database and will flag your question if:
1. The target `function_name` or `file_path` is explicitly written in your question.
2. The question has a **Cosine Distance < 0.2** (meaning it's a near-duplicate of the chunk's code).

If flagged, reword your question to test conceptual understanding rather than exact keyword matching!

### Step 3: Promote to the Main Set
Once your draft questions are clean and properly balanced, move them from `data/eval/eval_set_draft.jsonl` to `data/eval/eval_set.jsonl` to lock them in for the final benchmark runs.

```bash
cat data/eval/eval_set_draft.jsonl >> data/eval/eval_set.jsonl
```
