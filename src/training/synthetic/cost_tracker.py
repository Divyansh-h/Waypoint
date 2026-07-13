class CostTracker:
    """
    Tracks token usage and estimates API costs for synthetic data generation.
    Prices are per 1M tokens.
    """
    PRICING = {
        "gemini-1.5-flash": {"input": 0.35, "output": 1.05}, 
        "gemini-2.5-flash": {"input": 0.35, "output": 1.05},  # Assume same as 1.5
        "gpt-4o-mini": {"input": 0.150, "output": 0.600},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25}
    }
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.pricing = self.PRICING.get(model_name, {"input": 0.0, "output": 0.0})

    def add_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def get_estimated_cost(self) -> float:
        input_cost = (self.total_input_tokens / 1_000_000) * self.pricing["input"]
        output_cost = (self.total_output_tokens / 1_000_000) * self.pricing["output"]
        return input_cost + output_cost
        
    def summary(self) -> str:
        return (f"Model: {self.model_name} | "
                f"Tokens: {self.total_input_tokens:,} in, {self.total_output_tokens:,} out | "
                f"Est. Cost: ${self.get_estimated_cost():.4f}")
