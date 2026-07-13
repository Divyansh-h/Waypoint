import argparse
import os
import yaml
import wandb
import datetime

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a bi-encoder retrieval model using LoRA and MNRL.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the training configuration YAML file (e.g., config/training_config.yaml)."
    )
    parser.add_argument(
        "--base-model", 
        type=str, 
        required=True, 
        help="HuggingFace model ID or path to the base embedding model (e.g., 'sentence-transformers/all-MiniLM-L6-v2')."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        required=True, 
        help="Directory to save the fine-tuned LoRA checkpoints."
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a previous checkpoint directory to resume training from."
    )
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Retrieval Fine-Tuning Pipeline")
    print(f"   -> Base Model: {args.base_model}")
    print(f"   -> Config: {args.config}")
    print(f"   -> Output Dir: {args.output_dir}\n")
    
    # 1. Load Configuration
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file not found: {args.config}")
        return
        
    print("STEP 1: Loading configuration and initializing W&B...")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Initialize Weights & Biases for automated experiment tracking
    wandb.init(
        project=config.get("wandb_project", "waypoint-retriever-finetune"),
        name=f"lora-{os.path.basename(args.base_model)}",
        config={
            "base_model": args.base_model,
            "learning_rate": config.get("learning_rate", 2e-4),
            "batch_size": config.get("batch_size", 64),
            "epochs": config.get("epochs", 3),
            "lora_rank": config.get("lora_rank", 16),
            "lora_alpha": config.get("lora_alpha", 32)
        }
    )
    
    # Generate a unique output directory using timestamp and W&B run ID to prevent overwriting
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(args.output_dir, f"run_{timestamp}_{wandb.run.id}")
    
    # TODO: Load data/training/train/train.jsonl and data/training/val/val.jsonl 
    # TODO: Format them into InputExample objects for SentenceTransformers
    
    print("STEP 2: Initializing base model and LoRA configuration...")
    # TODO: Load SentenceTransformer(args.base_model)
    # TODO: Define LoraConfig (r=wandb.config.lora_rank, alpha=wandb.config.lora_alpha, target_modules=["query", "value"])
    # TODO: Wrap model with get_peft_model()
    
    print("STEP 3: Preparing MultipleNegativesRankingLoss (MNRL)...")
    # TODO: Initialize losses.MultipleNegativesRankingLoss(model)
    
    print("STEP 4: Executing Training Loop...")
    # TODO: Setup DataLoader with the MNRL batch_size
    # TODO: Setup TrainingArguments with:
    #       - report_to="wandb"
    #       - logging_steps=10
    #       - eval_strategy="epoch"
    #       - save_strategy="epoch"
    #       - load_best_model_at_end=True
    #       - metric_for_best_model="eval_val_eval_recall@10"
    #       - output_dir=run_output_dir
    # TODO: Initialize SentenceTransformerTrainer 
    # TODO: Execute trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    print(f"STEP 5: Saving LoRA adapter to {run_output_dir}...")
    os.makedirs(run_output_dir, exist_ok=True)
    # TODO: Call model.save_pretrained(run_output_dir)
    
    wandb.finish()
    print("\n✅ Fine-tuning complete. (STUBBED)")

if __name__ == "__main__":
    main()
