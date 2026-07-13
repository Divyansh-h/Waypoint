import argparse
import logging
import json
import torch
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("debug_loss")

def get_batch_loss(model, loss_fct, examples, batch_size=16):
    """Calculates the raw MNRL loss for a given set of examples."""
    dataloader = DataLoader(examples, batch_size=batch_size, shuffle=False)
    # We just run one forward pass to get the starting loss
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            features, labels = model.smart_batching_collate(batch)
            loss_val = loss_fct(features, labels)
            total_loss += loss_val.item()
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser(description="Debug Fine-Tuning Loss and Negatives Quality")
    parser.add_argument("--base-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--train-file", type=str, default="data/training/train/train.jsonl")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    logger.info("🚀 Starting Systematic Training Debugger")
    
    try:
        model = SentenceTransformer(args.base_model)
        loss_fct = losses.MultipleNegativesRankingLoss(model)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Load 1 batch of data
    hard_negative_examples = []
    random_negative_examples = []
    
    logger.info(f"Loading {args.batch_size} samples from {args.train_file}")
    with open(args.train_file, "r", encoding="utf-8") as f:
        lines = f.readlines()[:args.batch_size]
        
    for line in lines:
        obj = json.loads(line)
        anchor = obj.get("anchor")
        positive = obj.get("positive")
        # Assume 'hard_negative' key exists if pgvector mining succeeded
        hard_neg = obj.get("hard_negative") 
        
        if anchor and positive:
            # 1. Random Negatives (MNRL relies purely on other positives in the batch)
            random_negative_examples.append(InputExample(texts=[anchor, positive]))
            
            # 2. Explicit Hard Negatives (MNRL uses the explicit 3rd element as a hard negative)
            if hard_neg:
                hard_negative_examples.append(InputExample(texts=[anchor, positive, hard_neg]))
            else:
                # If no hard negatives exist, we just simulate by using a different positive
                hard_negative_examples.append(InputExample(texts=[anchor, positive, positive]))
                
    logger.info("--- DIAGNOSTIC 1: HARD VS RANDOM NEGATIVES ---")
    random_loss = get_batch_loss(model, loss_fct, random_negative_examples, args.batch_size)
    hard_loss = get_batch_loss(model, loss_fct, hard_negative_examples, args.batch_size)
    
    logger.info(f"Loss with Random (In-Batch) Negatives: {random_loss:.4f}")
    logger.info(f"Loss with Explicit Hard Negatives:     {hard_loss:.4f}")
    
    if hard_loss > random_loss + 0.5:
        logger.info("✅ SUCCESS: Hard negatives are significantly harder than random. The pgvector mining is working perfectly.")
    elif hard_loss < random_loss:
        logger.warning("❌ WARNING: Hard negatives generated lower loss than random! Your pgvector mining might be pulling completely unrelated junk, or 'false negatives'.")
    else:
        logger.info("⚠️ NEUTRAL: Hard negatives are only marginally harder. Consider lowering the pgvector distance threshold.")

    logger.info("\n--- DIAGNOSTIC 2: LEARNING RATE SWEEP ---")
    logger.info("If your gains are flat or negative, DO NOT immediately change the dataset.")
    logger.info("Run three quick 2-epoch smoke tests using W&B to check the loss curve slope:")
    logger.info("  1. LR: 2e-4 (Standard LoRA)")
    logger.info("  2. LR: 2e-5 (Order of magnitude lower - fixes catastrophic 'loss explosions')")
    logger.info("  3. LR: 1e-3 (Order of magnitude higher - fixes 'flatline learning')")

if __name__ == "__main__":
    main()
