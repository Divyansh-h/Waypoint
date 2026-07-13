# ADR 0004: Retrieval Fine-Tuning Architecture (MNRL + LoRA)

**Date**: July 2026  
**Status**: Accepted  
**Context**: Phase 3 Evaluation Pipeline 

## Context & Problem Statement
In Phase 1, our out-of-the-box embedding models heavily bottlenecked our retrieval pipeline. Our evaluation identified two catastrophic failure modes:
1. **Namespace Collisions (Class-Method Mapping):** The model fundamentally failed to distinguish between `KMeans.fit()` and `LinearRegression.fit()`, retrieving incorrect clusters simply because they shared the lexical verb "fit".
2. **Parameter Ignorance:** The model treated critical structural parameters (like `n_jobs` or `random_state`) as generic nouns, failing to retrieve chunks that explicitly defined them.

We need to fine-tune a domain-adapted retrieval model to learn the `scikit-learn` vocabulary without suffering from "Catastrophic Forgetting" (where the model forgets general English semantics in exchange for memorizing Python code). 

## Decision

We will implement a custom PyTorch HuggingFace Training loop using **MultipleNegativesRankingLoss (MNRL)** combined with **Low-Rank Adaptation (LoRA)** on top of the `sentence-transformers/all-MiniLM-L6-v2` architecture.

### 1. Base Model: `all-MiniLM-L6-v2`
* **Reasoning:** It natively outputs L2-normalized 384-dimensional vectors. This perfectly aligns with our PostgreSQL `pgvector` index which executes the `<=>` (Cosine Distance) operator. Furthermore, at only 22 million parameters, the memory footprint allows for incredibly rapid, local fine-tuning iterations without requiring cloud A100 GPU compute. 

### 2. Loss Function: Multiple Negatives Ranking Loss (MNRL)
* **Reasoning:** In Phase 1, the model failed due to false lexical overlap. MNRL is a contrastive loss function designed specifically for this problem. By passing in triplets (`Anchor`, `Positive`, `Hard Negative`), MNRL physically forces the embeddings of the `Anchor` (e.g., a query about KMeans) closer to the `Positive` docstring, while actively pushing it away from the `Hard Negative` (a docstring that looks lexically identical, like LinearRegression, but is semantically wrong). 

### 3. Strategy: Low-Rank Adaptation (LoRA)
* **Reasoning:** Training all 22 million parameters on pure Python docstrings guarantees Catastrophic Forgetting of the English language. By freezing the base model and only attaching tiny trainable LoRA matrices (Rank=32, Alpha=64) strictly to the `query` and `value` Attention heads, we cleanly map the `scikit-learn` vocabulary into the model's latent space while mathematically preserving its foundational linguistic knowledge.

### 4. Hyperparameters: High Batch Size (64)
* **Reasoning:** The effectiveness of MNRL scales directly with Batch Size because every anchor query is compared against every *other* positive in the batch as an "in-batch negative". By doubling our batch size to 64 (yielding 63 in-batch negatives per anchor) and lowering the learning rate to `5e-5` to stabilize the gradients, we achieved significantly cleaner convergence and pushed Recall@10 over our 60.0% goal.

## Consequences
* **Positive:** We mathematically solved the Phase 1 retrieval bottlenecks, boosting Recall@10 from 44.0% to 63.5% on our blind holdout set.
* **Positive:** The pipeline fits into <1.5GB of VRAM and runs locally on consumer hardware in under 3 minutes.
* **Negative:** Because we are dynamically injecting LoRA adapters at runtime via `PEFT`, loading the model into our inference pipeline requires an extra HuggingFace initialization step compared to a raw, statically merged `.safetensors` file.
