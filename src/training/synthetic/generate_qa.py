import os
import json
import google.generativeai as genai
from typing import Dict, Any
from training.schema import TrainingPair, PositiveChunk

# Note: Update this type hint to use your actual Chunk dataclass/schema once integrated
Chunk = Dict[str, Any]

def load_prompt_template(template_path: str) -> str:
    """
    Loads the LLM prompt template from disk.
    """
    if not os.path.exists(template_path):
        # Return a fallback for testing if file doesn't exist
        return "Generate a realistic, intent-based question for the following code:\n\n{code}"
    
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()

def generate_question_from_chunk(chunk: Chunk, prompt_template_path: str) -> TrainingPair:
    """
    Calls an LLM API to generate a synthetic user query based on a code chunk,
    guided by a specific prompt template (e.g., few-shot or structured JSON).
    
    Args:
        chunk: The code chunk dictionary parsed from the AST.
        prompt_template_path: Path to the text/JSON file containing the prompt instructions.
        
    Returns:
        A strictly typed TrainingPair object.
    """
    prompt_template = load_prompt_template(prompt_template_path)
    
    chunk_id = chunk.get("chunk_id", "unknown_id")
    content = chunk.get("content", "")
    
    # ---------------------------------------------------------
    # Actual LLM API Call
    # ---------------------------------------------------------
    formatted_prompt = prompt_template.replace("{code}", content)
    
    generated_question = f"How do I use the {chunk.get('name', 'target')} implementation?"
    core_concept = "Stubbed concept due to API failure/missing key."
    input_tokens = int(len(formatted_prompt.split()) / 0.75)
    output_tokens = 25
    
    if os.environ.get("GEMINI_API_KEY"):
        try:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            model = genai.GenerativeModel("gemini-2.5-flash")
            
            # Use JSON mode if supported
            response = model.generate_content(
                formatted_prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            
            # Parse JSON
            result = json.loads(response.text)
            generated_question = result.get("synthetic_question", generated_question)
            core_concept = result.get("core_concept", core_concept)
            
            # Try to get actual token usage from the Gemini API response
            if hasattr(response, 'usage_metadata'):
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
            else:
                input_tokens = int(len(formatted_prompt.split()) / 0.75)
                output_tokens = int(len(generated_question.split()) / 0.75)
                
        except Exception as e:
            print(f"   -> [API Error] Chunk {chunk_id}: {e}")
    else:
        # Fallback approximation for testing without a key
        input_tokens = int(len((prompt_template + content).split()) / 0.75)
        output_tokens = int(len(generated_question.split()) / 0.75)
    
    return TrainingPair(
        anchor=generated_question,
        positive=PositiveChunk(
            chunk_id=chunk_id,
            content=content,
            file_path=chunk.get("file_path")
        ),
        negatives=[],
        source="synthetic",
        metadata={
            "prompt_template": prompt_template_path,
            "llm_model": "gemini-2.5-flash",
            "core_concept": core_concept,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
        }
    )
