import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import google.generativeai as genai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Ensure API key is available
if "GEMINI_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])  # type: ignore[attr-defined]
else:
    logger.warning("GEMINI_API_KEY not found in environment. LLM calls will fail.")

@dataclass
class JudgeScore:
    """
    Represents the LLM-as-Judge binary rubric checklist.
    Each True value corresponds to 1 point. Maximum score is 5.
    """
    is_correct: bool          # Does it resolve the query without contradicting context?
    no_hallucination: bool    # Are all APIs explicitly present in the chunks?
    is_complete: bool         # Did it address all sub-questions/constraints?
    multi_hop_synthesis: bool # Did it successfully synthesize >= 2 chunks (if provided)?
    has_citation: bool        # Did it explicitly name the source class/function?
    
    @property
    def total_score(self) -> int:
        """Returns the aggregate score out of 5 based on the binary checklist."""
        return sum([
            self.is_correct,
            self.no_hallucination,
            self.is_complete,
            self.multi_hop_synthesis,
            self.has_citation
        ])

def score_answer(question: str, answer: str, retrieved_chunks: List[Dict[str, Any]]) -> JudgeScore:
    """
    Evaluates a generated RAG answer against the provided ground-truth chunks 
    using the Gemini API and the structured Binary Checklist Rubric.
    """
    logger.info(f"Evaluating answer for question: '{question[:50]}...'")
    
    # 1. Load the Judge Prompt Template
    prompt_path = Path(__file__).parent.parent.parent / "prompts" / "llm_judge_rubric.txt"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read()
    except FileNotFoundError:
        logger.error(f"Could not find prompt template at {prompt_path}")
        return JudgeScore(False, False, False, False, False)
        
    # Format chunks into a readable string
    chunks_text = ""
    for i, chunk in enumerate(retrieved_chunks):
        content = chunk.get("content", "Unknown content")
        file_path = chunk.get("file_path", "Unknown file")
        chunks_text += f"\n--- Chunk {i+1} ({file_path}) ---\n{content}\n"
        
    # Inject variables into the template safely to avoid JSON bracket collisions
    prompt = template.replace("{question}", question)
    prompt = prompt.replace("{chunks}", chunks_text)
    prompt = prompt.replace("{answer}", answer)
    
    # 2. Call the Gemini API with JSON enforcement
    try:
        model = genai.GenerativeModel(  # type: ignore[attr-defined]
            model_name="gemini-2.5-flash",
            generation_config=genai.GenerationConfig(  # type: ignore[attr-defined]
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        
        response = model.generate_content(prompt)
        result_text = response.text
        
        # 3. Parse JSON into the JudgeScore dataclass
        # Clean markdown formatting in case the model hallucinates code block tags despite mime_type
        clean_text = result_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()
            
        try:
            data = json.loads(clean_text)
        except json.JSONDecodeError as e:
            logger.error(f"Judge returned unparseable JSON: {clean_text}\nParse Error: {e}")
            return JudgeScore(False, False, False, False, False)
        
        return JudgeScore(
            is_correct=bool(data.get("is_correct", False)),
            no_hallucination=bool(data.get("no_hallucination", False)),
            is_complete=bool(data.get("is_complete", False)),
            multi_hop_synthesis=bool(data.get("multi_hop_synthesis", False)),
            has_citation=bool(data.get("has_citation", False))
        )
        
    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        # Return a zero-score fallback on crash
        return JudgeScore(False, False, False, False, False)
