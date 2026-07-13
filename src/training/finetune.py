# ruff: noqa: E501
import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from datasets import Dataset  # type: ignore
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

import wandb

# Ensure src is in python path
src_dir = Path(__file__).parent.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from eval_during_training import create_ir_evaluator  # type: ignore[import-not-found] # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def load_jsonl_to_dataset(filepath: str, max_samples: Optional[int] = None) -> Dataset:
    data: Dict[str, List[Any]] = {"anchor": [], "positive": []}
    
    if not os.path.exists(filepath):
        logger.warning(f"Dataset not found: {filepath}")
        return Dataset.from_dict(data)
        
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    if max_samples:
        lines = lines[:max_samples]
        
    for line in lines:
        obj = json.loads(line)
        if "anchor" in obj and "positive" in obj:
            data["anchor"].append(obj["anchor"])
            data["positive"].append(obj["positive"])
            
    return Dataset.from_dict(data)

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a bi-encoder retrieval model using LoRA and MNRL.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training configuration YAML file.")
    parser.add_argument("--base-model", type=str, required=True, help="HuggingFace model ID or path to the base embedding model.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the fine-tuned LoRA checkpoints.")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to a previous checkpoint directory to resume training from.")
    args = parser.parse_args()
    
    logger.info("🚀 Starting Retrieval Fine-Tuning Pipeline")
    logger.info(f"   -> Base Model: {args.base_model}")
    logger.info(f"   -> Config: {args.config}")
    
    # 1. Load Configuration & Init W&B
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
        
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Inject CLI args into config for full logging traceability
    config["base_model"] = args.base_model
        
    wandb.init(
        project=config.get("wandb_project", "waypoint-retriever-finetune"),
        name=f"lora-{os.path.basename(args.base_model)}",
        config=config
    )
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = wandb.run.id if wandb.run else "offline"
    run_output_dir = os.path.join(args.output_dir, f"run_{timestamp}_{run_id}")
    
    # 2. Load Datasets
    logger.info("STEP 1: Loading datasets...")
    train_path = "data/training/train/train.jsonl"
    val_path = "data/training/val/val.jsonl"
    
    train_dataset = load_jsonl_to_dataset(train_path, config.get("max_train_samples"))
    val_dataset = load_jsonl_to_dataset(val_path, config.get("max_val_samples"))
    logger.info(f"Loaded {len(train_dataset)} training pairs and {len(val_dataset)} validation pairs.")
    
    # 3. Load Model and PEFT
    logger.info("STEP 2: Initializing base model and LoRA configuration...")
    model = SentenceTransformer(args.base_model)
    
    lora_config = LoraConfig(
        r=config.get("lora_rank", 16), 
        lora_alpha=config.get("lora_alpha", 32), 
        target_modules=config.get("target_modules", ["query", "value"]),
        bias=config.get("lora_bias", "none")
    )
    # Apply PEFT to the underlying transformer model
    model[0].auto_model = get_peft_model(model[0].auto_model, lora_config)  # type: ignore
    model[0].auto_model.print_trainable_parameters()
    
    # 4. Prepare MNRL Loss
    logger.info("STEP 3: Preparing MultipleNegativesRankingLoss (MNRL)...")
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # 5. Build Evaluator
    evaluator = create_ir_evaluator(val_path, name="val_eval")
    
    # 6. Training Arguments & Trainer
    logger.info("STEP 4: Executing Training Loop...")
    
    training_args = SentenceTransformerTrainingArguments(
        output_dir=run_output_dir,
        num_train_epochs=config.get("epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 64),
        per_device_eval_batch_size=config.get("batch_size", 64),
        learning_rate=float(config.get("learning_rate", 2e-4)),
        warmup_ratio=float(config.get("warmup_ratio", 0.1)),
        eval_strategy=config.get("eval_strategy", "epoch"),
        save_strategy=config.get("save_strategy", "epoch"),
        load_best_model_at_end=config.get("load_best_model_at_end", True),
        metric_for_best_model=config.get("metric_for_best_model", "eval_val_eval_ndcg@10"),
        greater_is_better=config.get("greater_is_better", True),
        report_to="wandb",
        logging_steps=config.get("logging_steps", 10)
    )
    
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=train_loss,
        evaluator=evaluator
    )
    
    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("❌ CRITICAL FAILURE: CUDA Out of Memory (OOM).")
            logger.error("-> FIX: Lower the 'batch_size' in your YAML config, or reduce the 'lora_rank'.")
            sys.exit(1)
        else:
            logger.error(f"❌ CRITICAL FAILURE: Runtime error during training: {e}")
            sys.exit(1)
    except ValueError as e:
        if "nan" in str(e).lower():
            logger.error("❌ CRITICAL FAILURE: Loss diverged to NaN.")
            logger.error("-> FIX: Your learning rate is likely too high. Lower 'learning_rate' in the config.")
            sys.exit(1)
        else:
            logger.error(f"❌ CRITICAL FAILURE: Value error during training: {e}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ CRITICAL FAILURE: Unexpected training crash: {e}")
        sys.exit(1)
    
    # 7. Save Model
    logger.info(f"STEP 5: Saving best LoRA adapter to {run_output_dir}...")
    model.save_pretrained(run_output_dir)
    
    wandb.finish()
    logger.info("✅ Fine-tuning complete.")

if __name__ == "__main__":
    main()
