import unittest
import torch
from sentence_transformers import SentenceTransformer
from peft import PeftModel
import os

class TestLoRACheckpoint(unittest.TestCase):
    def setUp(self):
        self.base_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.checkpoint_dir = "checkpoints/best"
        self.test_query = "How do I fit a KMeans clustering model?"

    def test_embeddings_diverged(self):
        # 1. Load Base Model and embed
        base_model = SentenceTransformer(self.base_model_name)
        base_embedding = base_model.encode(self.test_query, convert_to_tensor=True)

        # 2. Check if checkpoint exists before testing (prevents crashing if not trained yet)
        if not os.path.exists(os.path.join(self.checkpoint_dir, "adapter_config.json")):
            self.skipTest("No LoRA checkpoint found at checkpoints/best. Skipping sanity check.")

        # 3. Load LoRA adapter on top of base model
        try:
            # We apply the PEFT adapter to the underlying transformer architecture
            base_model[0].auto_model = PeftModel.from_pretrained(base_model[0].auto_model, self.checkpoint_dir)
            lora_embedding = base_model.encode(self.test_query, convert_to_tensor=True)
        except Exception as e:
            self.fail(f"Failed to load PEFT adapter from {self.checkpoint_dir}: {e}")

        # 4. Compare embeddings
        # If fine-tuning took effect, the vectors should no longer be perfectly identical.
        # We check the cosine similarity. If it's exactly 1.0, the LoRA weights did nothing.
        cos_sim = torch.nn.functional.cosine_similarity(base_embedding.unsqueeze(0), lora_embedding.unsqueeze(0))
        
        # We assert that the similarity is less than 0.999 (meaning the vectors mathematically diverged)
        self.assertLess(
            cos_sim.item(), 
            0.999, 
            "The LoRA embeddings are identical to the base embeddings! Fine-tuning did not take effect."
        )

if __name__ == "__main__":
    unittest.main()
