# ADR: Use AST-Based Chunking and pgvector for Code-Search Pipeline

**Date:** 2026-07-10
**Status:** Accepted
**Author(s):** Divyansh

## Context
We are building a Retrieval-Augmented Generation (RAG) pipeline to index and search source code. The target codebase is a subset of `scikit-learn` (`linear_model`, `ensemble`, `utils`, `base.py`), comprising roughly 115k lines of complex, class-heavy, and mathematically dense Python code. A critical constraint is the need to maintain object-oriented structural context—specifically ensuring that deep logic (like `fit` or `predict` methods) is not decoupled from its parent class configuration or associated massive NumPy-style docstrings. Additionally, there is a goal to ensure the solution is robust for local deployment with a clear migration path to Kubernetes.

## Decision
We will use `tree-sitter-python` for AST-based (Abstract Syntax Tree) structural chunking of the source code, and PostgreSQL with the `pgvector` extension as our vector database.

## Alternatives Considered
- **Fixed-Token Window Chunking:** Rejected. Scikit-learn features massive docstrings. A fixed token window (e.g., 512 tokens) frequently captures an entire docstring but chops off the actual implementation logic below it, severing the connection between documentation and code.
- **Semantic/Embedding-Based Splitting:** Rejected. This approach struggles with the rigid OOP structure of `sklearn/base.py`. It relies on similarity thresholds that might arbitrarily split a class in half (e.g., separating data validation from algorithmic execution), breaking the OOP context. It is also computationally expensive during ingestion.
- **Dedicated Vector DBs (Qdrant/Chroma):** Postgres with `pgvector` was chosen over these because Postgres offers a proven, highly robust migration path to Kubernetes and allows us to easily combine relational metadata filtering with vector similarity search.

## Consequences
- **Positive:** 
  - **High Retrieval Quality:** Maintains structural integrity. It isolates specific algorithm implementations (e.g., `RandomForestClassifier.fit`) into logical, self-contained chunks that LLMs excel at reading.
  - **Test Isolation:** Perfectly isolates the heavily structured ~49k lines of `test_*` functions for distinct retrieval.
  - **Infrastructure:** Robust, production-ready migration path to Kubernetes via Postgres.
- **Negative:** 
  - **Increased Complexity:** Higher implementation complexity in the ingestion pipeline, requiring us to manage `tree-sitter` parsers, multi-byte Unicode bugs, and edge cases like decorated methods and nested classes.
  - **God Nodes:** Extremely large "God" classes or massive `fit` methods might still exceed context limits as a single chunk.
  - **Hybrid Search Friction:** Lack of out-of-the-box score fusion for hybrid search in pgvector requires writing custom SQL.
