from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator
from typing import List, Literal, Optional


class Chunk(BaseModel):
    """
    Represents a chunk of code or text parsed from a file.
    Designed for embedding and insertion into a vector database.
    """
    content: str = Field(..., min_length=1, description="Raw chunk content.")
    file_path: str = Field(..., description="Path to the source file.")
    chunk_type: str = Field(..., description="Type of the chunk.")
    function_name: Optional[str] = Field(default=None, description="Name of function/class.")
    section_path: Optional[str] = Field(default=None, description="Markdown header path.")
    line_start: int = Field(..., ge=1, description="1-indexed start line.")
    line_end: int = Field(..., ge=1, description="1-indexed end line.")

    @model_validator(mode='after')
    def validate_line_range(self) -> 'Chunk':
        """Ensure that line_end is greater than or equal to line_start."""
        if self.line_end < self.line_start:
            raise ValueError(f"line_end ({self.line_end}) < line_start ({self.line_start})")
        return self

    @model_validator(mode='after')
    def validate_file_path(self) -> 'Chunk':
        """Ensure the file_path is not empty and has an extension (rudimentary check)."""
        if not self.file_path.strip():
            raise ValueError("file_path cannot be empty or just whitespace.")
        # Basic check to ensure it looks like a file and not just a directory name
        if not Path(self.file_path).suffix:
            raise ValueError(f"file_path '{self.file_path}' has no valid extension.")
        return self


class EmbeddedChunk(Chunk):
    """
    Extends the base Chunk model to include the vector embedding.
    """
    vector: list[float] = Field(..., description="The dense vector embedding.")


class EvalExample(BaseModel):
    """
    A single evaluation test case for the RAG retrieval system.
    """
    id: str = Field(..., description="Unique identifier for the eval example (e.g., 'eval_001').")
    
    question: str = Field(
        ..., 
        description="The natural language query a user would ask."
    )
    
    ground_truth_chunk_ids: List[str] = Field(
        ..., 
        description="A list of chunk IDs that must ALL be retrieved to successfully answer the question."
    )
    
    difficulty_tag: Literal["easy", "medium", "hard", "adversarial"] = Field(
        ..., 
        description="Difficulty level of the question."
    )
    
    question_type: Literal["factual", "debugging", "conceptual", "api_usage", "out_of_scope", "multi_hop"] = Field(
        ..., 
        description="Categorization of the user intent to ensure diverse testing."
    )

    metadata: Optional[dict] = Field(
        default=None,
        description="Optional context like the source of the question (e.g., GitHub Issue URL)."
    )

class EvalDataset(BaseModel):
    """
    The full suite of evaluation examples.
    """
    dataset_name: str
    version: str
    examples: List[EvalExample]
