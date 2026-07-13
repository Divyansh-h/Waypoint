# Training Data Generation Pipeline

## 1. Overview
This document details the architecture and execution of the Dataset Generation Orchestrator (`scripts/generate_training_data.py`), which constructs the training dataset used to fine-tune our retriever model via Multiple Negatives Ranking Loss (MNRL). 

The goal of this pipeline is to create a robust, leak-proof dataset that balances scale (covering the entire codebase) with targeted semantic difficulty.

---

## 2. The 5-Step Pipeline

### Step 1: Mine (Positive Docstrings)
* **Goal:** Establish cheap, foundational bulk coverage of the `scikit-learn` API surface.
* **Mechanism:** The AST chunker crawls the database and extracts function/class docstrings, treating the first sentence as the `anchor` (query) and the stripped implementation code beneath it as the `positive` chunk.

### Step 2: Generate (Synthetic Queries)
* **Goal:** Bridge the lexical gap between formal developer syntax (docstrings) and intent-based natural language (how users actually search).
* **Mechanism:** Code chunks are sent to a generative LLM (e.g., `gemini-1.5-flash-8b`) which generates a synthetic user query and a core concept. 
* **Safeguards:** This step is wrapped in a robust `tenacity` retry loop with exponential backoff to handle rate-limits, and it features a resumable checkpoint system (`raw_questions.jsonl`) so failed runs do not require starting from scratch.

### Step 3: Mine Hard Negatives
* **Goal:** Teach the model to differentiate between highly similar concepts.
* **Mechanism:** For every generated `anchor`, we query `pgvector` using the un-finetuned baseline model's embeddings to retrieve the Top-20 nearest neighbors. 
* **Safeguards:** The SQL query explicitly excludes the true positive chunk ID (`WHERE id != %s`) to prevent self-negatives. 

### Step 4: Quality Filtering & Stratification
* **Filtering:** All pairs are pushed through a quality pipeline that:
  1. Drops questions that are too short (`< 4` words).
  2. Drops near-duplicate questions using TF-IDF cosine similarity (`> 0.85` threshold).
  3. Flags LLM hallucinations by verifying the LLM's `core_concept` actually shares semantic/lexical overlap with the target code chunk.
* **Stratified Oversampling:** The surviving pristine pairs are bucketed by source and oversampled to meet a strict target ratio: **70% general coverage** (mined/synthetic) and **30% failure telemetry** (from Phase 1 eval logs).

### Step 5: Train/Val Split
* **Goal:** Ensure absolute isolation between the training set and validation set to prevent data leakage.
* **Mechanism:** The dataset is partitioned at the **module (file) level**. All chunks derived from a specific file (e.g., `ensemble.py`) will go entirely to `train` or entirely to `val`. 

---

## 3. Dataset Statistics (Current Artifact)
*(Note: These stats represent the scaffolded test run. Production runs will scale these numbers up significantly).*

* **Total Dataset Size:** 700 pairs (padded to meet the oversampler target)
* **Docstring Mined Pairs:** 519 pristine pairs (Oversampled to 700)
* **Synthetic LLM Pairs:** 0 (API endpoint `gemini-2.5-flash` was deprecated mid-run)
* **Failure-derived Pairs:** 0 (Phase 1 telemetry ingestion not yet scaffolded)
* **Hard Negatives per Query:** ~20 
* **Train/Val Split:** 526 pairs (Train) / 174 pairs (Val)

---

## 4. Known Limitations & Risks

1. **False Negative Punishment:** Hard negative mining inherently risks selecting chunks that are equally valid alternative answers to a query (e.g. `RandomForestClassifier` vs `RandomForestRegressor`). This inadvertently penalizes the MNRL model for retrieving semantically viable code.
2. **Docstring Overfitting Bias:** Because the API failed during synthetic generation, the current dataset artifact is 100% docstring-mined data. Training on this immediately would cause the model to overfit heavily to developer syntax and lose natural language semantic flexibility.
3. **TF-IDF Vocabulary Sparsity:** The duplicate filter relies on TF-IDF word frequencies. For extremely short queries (e.g. 4 words), differing by just a single word can artificially inflate the mathematical variance, causing near-duplicates to occasionally slip past the `0.85` similarity threshold.
