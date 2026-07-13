# Hardware & Memory Profiling (Phase 3 Fine-Tuning)

Because we opted for `sentence-transformers/all-MiniLM-L6-v2` as our base retrieval architecture and utilized **Low-Rank Adaptation (LoRA)**, this fine-tuning pipeline is extremely lightweight and **100% reproducible on a typical consumer laptop**. 

You do **not** need a cloud GPU instance (like an AWS A100 or T4) to run this codebase.

## Peak Memory Breakdown (Batch Size: 64)

| Component | VRAM/RAM Usage | Notes |
| :--- | :--- | :--- |
| **Base Model Weights** | ~90 MB | The 22-million parameter MiniLM model is exceptionally small. |
| **LoRA Adapters (r=32)** | ~1.5 MB | Only the Attention Query/Value matrices are trainable. |
| **AdamW Optimizer States** | ~5 MB | Tracks momentum *only* for the 1.5MB of LoRA weights, drastically slashing memory. |
| **Batch Activations (64)** | ~800 MB - 1.2 GB | Forward/backward pass memory for a batch of 64 triplets (Anchor, Positive, Negative). |
| **Total Peak Memory** | **< 1.5 GB** | Easily fits within the unified memory of a base 8GB Apple M1/M2/M3 MacBook. |

## Execution Speed
When running on an Apple Silicon `mps` backend or a standard discrete laptop GPU (like an RTX 3060), a full 4-epoch run over the complete 700+ pair dataset completes in **under 3 minutes**. 

If a user lacks a dedicated GPU entirely, the `MultipleNegativesRankingLoss` loop can even be executed purely on a laptop CPU, taking roughly 15-20 minutes to complete the full 4 epochs.
