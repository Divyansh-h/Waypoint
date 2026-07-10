# ADR: Use AST-Based Chunking and pgvector for Code-Search Pipeline

**Date:** [YYYY-MM-DD]
**Status:** [Draft / Proposed / Accepted / Rejected]
**Author(s):** [Your Name]

## Context
[Describe the problem you are solving, the business/technical context, and the constraints (e.g., scoping the pipeline to scikit-learn, the need to maintain object-oriented context like parent classes, local-to-kubernetes deployment goals).]

## Decision
[State the decision clearly. E.g., "We will use tree-sitter for AST-based structural chunking of the source code, and PostgreSQL with the pgvector extension for our vector database."]

## Alternatives Considered
[List the alternative options and briefly note why they were not chosen. For example:
- Fixed-token window chunking
- Semantic/embedding-based chunking
- Qdrant or Chroma for the vector database]

## Consequences
[Describe the impact of this decision. This should include both positive and negative consequences. For example:
- Positive: High retrieval quality due to preserved OOP structures, robust migration path to Kubernetes via Postgres.
- Negative: Increased complexity in the ingestion pipeline (managing tree-sitter parsers), lack of out-of-the-box score fusion for hybrid search in pgvector requiring custom SQL.]
