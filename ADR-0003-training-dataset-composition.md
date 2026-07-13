# ADR 0003: Training Dataset Composition for MNRL Fine-Tuning

## Status
Accepted

## Context
During Phase 2 (Dataset Generation), we needed to construct a robust dataset to fine-tune our dense retriever model via Multiple Negatives Ranking Loss (MNRL). Relying on a single source of data poses massive risks (e.g., overfitting to developer syntax or catastrophic forgetting of broad natural language semantics). We needed a multi-modal data generation strategy that provides both foundational scale and targeted semantic difficulty. 

## Decision
We decided to construct the MNRL dataset using a meticulously balanced mix of three distinct data sources, applying a Stratified Oversampling target of **70% general coverage** and **30% failure patterns**:

1. **Docstring-Mined (General Coverage - Bulk)**: 
   - *Reasoning:* We need cheap, massive bulk coverage of the entire API surface area. By extracting the AST docstrings, we map the entire codebase with zero API cost, establishing the foundational semantic structure.
   
2. **Synthetic LLM Queries (General Coverage - Intent)**:
   - *Reasoning:* Docstrings are written in formal developer syntax, which causes the model to overfit if used exclusively. We use LLM-generated synthetic queries to bridge this lexical gap, translating formal code chunks into the natural language, intent-based questions that real users actually ask.
   
3. **Failure-Derived Telemetry (Targeted - 30% Weight)**:
   - *Reasoning:* Our Phase 1 evaluation telemetry proved the baseline hybrid model heavily struggles with "Hard/Multi-hop" queries. By actively curating the exact queries the model failed on in Phase 1 (and weighting them at 30% of the dataset), we force the fine-tuned model to explicitly learn and correct the abstract edge cases that naive RRF couldn't solve.

## Consequences
* **Positive:** The model learns both the broad structure of the codebase (from bulk docstrings) and the natural vernacular of users (from synthetic data), while surgically correcting its worst behaviors (from targeted failure telemetry).
* **Negative:** Requires a significantly more complex orchestration pipeline (implementing `tenacity` retries, LLM prompt engineering, `pgvector` hard-negative mining, quality filtration, and stratified oversampling) rather than a simple 1:1 database dump.
