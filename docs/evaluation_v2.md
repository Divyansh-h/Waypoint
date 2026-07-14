# Evaluation Methodology V2

This document outlines the end-to-end evaluation methodology for the Waypoint RAG system. Evaluation is split into two distinct tiers: **Retrieval Metrics** (how well the vector database surfaces context) and **Generation Metrics** (how well the LLM synthesizes an answer using an automated Judge).

---

## 1. Retrieval Evaluation

Retrieval is evaluated entirely mechanistically based on the known `ground_truth_chunk_ids` associated with every question in `data/eval/eval_set.jsonl`. 

We measure two primary metrics for every query:
1. **Recall@10:** Did *all* required ground truth chunks appear in the top 10 results returned by the database? (Binary: 0.0 or 1.0)
2. **Mean Reciprocal Rank (MRR@10):** How high up the list was the highest-ranking ground truth chunk? Calculated as $\frac{1}{\text{rank}}$.

### The Execution:
Run `scripts/run_eval.py --eval-file data/eval/eval_set.jsonl`. 
The script queries the vector database, extracts the rankings, and computes the aggregate Recall/MRR across `EASY`, `HARD`, and `ADVERSARIAL` splits.

---

## 2. LLM-as-a-Judge Evaluation

Because LLM generation is non-deterministic, we employ an automated "LLM-as-a-Judge" to grade the RAG system's final output.

### Defense Against Sycophancy & Bias
To mathematically defend against model favoritism and verbosity bias, we employ four structural constraints:
1. **Model Isolation:** The generation model (the Agent) and the evaluation model (the Judge) are explicitly configured separately to avoid self-enhancement bias.
2. **Metadata Masking:** Before the Judge sees the retrieved chunks, we actively strip the chunk `id` (which often contains the fully qualified function/class name in synthetic datasets) and replace it with a generic mask (e.g., `Snippet 1`). This prevents the Judge from "cheating" by reading the header instead of verifying the code logic.
3. **The Binary Checklist:** The Judge does not score on a 1-5 scale. It must grade against a strict Boolean checklist:
   - `is_correct`
   - `no_hallucination`
   - `is_complete`
   - `multi_hop_synthesis`
   - `has_citation`
4. **Chain-of-Thought (Devil's Advocate):** The JSON schema forces the Judge to output a `"critique": str` field *before* it outputs the boolean scores. It is instructed to actively hunt for missing constraints or subtle hallucinations, which severely reduces default sycophancy.

### The Calibration Process
To prove the LLM Judge is trustworthy, it must be calibrated against human evaluators.
1. We run a subset of $N=100$ questions through the pipeline.
2. A human evaluator blind-grades the answers.
3. We compute the **Cohen's Kappa ($\kappa$)** statistic between the human and the LLM using `sklearn.metrics.cohen_kappa_score`.
4. **Threshold:** The Judge is only permitted to operate autonomously if $\kappa \ge 0.60$ (Substantial Agreement).

---

## 3. Adding New Evaluation Questions

The dataset is stored as JSONL in `data/eval/eval_set.jsonl`. 

### Defining an Example
To add a manual question, append a JSON object matching the `EvalExample` schema:

```json
{
  "id": "eval_manual_001",
  "question": "If I pass n_jobs=-1 to GridSearchCV, how does it parallelize the cross-validation?",
  "ground_truth_chunk_ids": [
    "556eb6fba465c79b6ba208c3fbb66970d65dce87c62aa2f564cc3b1fd4753e6e",
    "bf1c55c5019441436e2304224b8a4e144d2423d6d1e61ee2085a53b8f662e9cb"
  ],
  "difficulty_tag": "hard",
  "question_type": "multi_hop",
  "metadata": {
    "source": "human_curated"
  }
}
```

### Categorization Guide
* **`question_type = "single_hop"`**: The answer exists entirely within one docstring or method implementation. Requires exactly 1 ground truth chunk ID.
* **`question_type = "multi_hop"`**: The question requires tracing logic across multiple files or classes (e.g., passing a parameter into Class A which affects downstream Function B). Must contain $\ge 2$ ground truth chunk IDs.
* **`question_type = "adversarial"`**: The answer explicitly *does not exist* in the codebase. Used to test if the LLM Generator correctly refuses to answer rather than hallucinating an API.

### Synthetic Generation
If you need bulk data, do not write questions manually. Instead, use the AST-based synthetic data generator, which automatically parses Caller $\rightarrow$ Callee dependencies in the codebase to build hyper-realistic multi-hop constraints:
`python scripts/generate_multihop_synthetic.py`
