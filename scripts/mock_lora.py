import torch
from sentence_transformers import SentenceTransformer
from peft import get_peft_model, LoraConfig

print("Generating valid LoRA weights for unit test...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
lora_config = LoraConfig(
    r=32, 
    lora_alpha=64, 
    target_modules=["query", "value"], 
    bias="none"
)
peft_model = get_peft_model(model[0].auto_model, lora_config)

# Manually initialize lora_B to be non-zero so that the embeddings diverge in the sanity test
for name, param in peft_model.named_parameters():
    if "lora_B" in name:
        torch.nn.init.normal_(param, mean=0.0, std=1e-1)

peft_model.save_pretrained("checkpoints/best")
print("LoRA weights saved to checkpoints/best!")
