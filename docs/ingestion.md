# Code Ingestion Pipeline

The ingestion pipeline is responsible for parsing a local repository, breaking it down into semantically meaningful chunks, converting those chunks into dense vector embeddings, and indexing them into a vector database for semantic search.

This pipeline is optimized for complex, class-heavy Python codebases (like `scikit-learn`) using an Abstract Syntax Tree (AST) approach, guaranteeing that object-oriented structures and massive docstrings remain contextually whole.

## Pipeline Architecture

The ingestion process runs sequentially across four distinct stages:

### 1. Crawl (`crawler.py`)
The pipeline begins by walking the target repository directory on disk. 
- It applies strict inclusion and exclusion filters to focus purely on the relevant codebase (e.g., Python scripts and Markdown documentation).
- It ignores standard noise directories (like `.git`, `__pycache__`, or `node_modules`).

### 2. Chunk (`chunker.py` / `doc_chunker.py`)
Instead of naïvely sliding a fixed-token text window over the source code, this pipeline leverages **AST-Based Chunking** using `tree-sitter-python`.
- **Structural Integrity:** Files are parsed into distinct nodes (Classes, Functions, Methods). 
- **Context Preservation:** Nested structures (like `fit` methods inside `BaseEstimator` classes) are extracted, preserving the structural boundaries of the code logic. Decorators are kept strictly attached to their target definitions.
- **Portability:** Extracted chunks store relative file paths (e.g., `sklearn/linear_model/base.py`) rather than absolute machine paths, ensuring database portability.

### 3. Embed (`embed.py`)
Chunks are grouped into batches and sent to the **Jina API** (`jina-embeddings-v3`). 
- **Parallelization:** Requests are sent concurrently using a `ThreadPoolExecutor` to bypass heavy network IO bottlenecks, scaling throughput.
- **Resiliency:** Implements exponential backoff and retry logic to smoothly handle API rate limits and transient network failures.

### 4. Index (`indexer.py`)
The generated `EmbeddedChunk` objects are upserted into a **PostgreSQL** database powered by the `pgvector` extension.
- **Idempotency:** A stable, unique SHA-256 ID is generated for every chunk based on a composite key: `{file_path}:{line_start}`. 
- **Conflict Resolution:** If the pipeline is run twice, the database uses an `ON CONFLICT (id) DO UPDATE` (upsert) clause. It overwrites the existing row with fresh code and embeddings instead of creating duplicate records.
- **JSONB Metadata:** Deep context details (chunk type, function name, structural path) are stored in a native `JSONB` column, enabling lightning-fast metadata filtering during vector search.

---

## Configuration (`configs/ingestion.yaml`)

The entire pipeline is driven by a YAML configuration file. 

The pipeline strictly validates this file's structure at runtime, cleanly exiting if required nested keys are missing.

### Available Options

```yaml
# 1. Repository Settings
repo:
  path: "/path/to/local/repo"        # (Required) Absolute or relative path to the repo to index

# 2. File Filtering
filtering:
  include_extensions:
    - ".py"                          # (Required) File extensions to process
    - ".md"
  exclude_extensions:                # (Optional) Specific extensions to drop
    - ".c"
  exclude_directories:               # (Optional) Directories to ignore during the crawl
    - ".git"
    - "__pycache__"

# 3. Chunking Parameters
chunking:
  max_tokens: 1500                   # (Required) Soft-limit on chunk size (e.g., embedding model context limit)

# 4. Vector Database
database:
  connection_string: "postgresql://user:pass@localhost:5432/db" # (Required) Standard Postgres URI
  collection_name: "sklearn_code"                               # (Optional) Table name (defaults to 'chunks')
```

## Running the Pipeline

To execute the pipeline:
```bash
# Run a dry-run to sample 10 parsed chunks without touching APIs or DBs
python scripts/run_ingestion.py --config configs/ingestion.yaml --dry-run

# Execute full end-to-end ingestion
python scripts/run_ingestion.py --config configs/ingestion.yaml
```
