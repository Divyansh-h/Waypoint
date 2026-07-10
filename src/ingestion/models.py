from typing import Optional
from pydantic import BaseModel, Field, model_validator
from pathlib import Path


class Chunk(BaseModel):
    """
    Represents a chunk of code or text parsed from a file.
    Designed for embedding and insertion into a vector database.
    """
    content: str = Field(..., min_length=1, description="The raw string content of the chunk.")
    file_path: str = Field(..., description="The absolute or relative path to the source file.")
    chunk_type: str = Field(..., description="The type of the chunk (e.g., 'class', 'function', 'fallback_text').")
    function_name: Optional[str] = Field(default=None, description="The name of the function or class, if applicable.")
    line_start: int = Field(..., ge=1, description="The 1-indexed starting line number.")
    line_end: int = Field(..., ge=1, description="The 1-indexed ending line number.")

    @model_validator(mode='after')
    def validate_line_range(self) -> 'Chunk':
        """Ensure that line_end is greater than or equal to line_start."""
        if self.line_end < self.line_start:
            raise ValueError(f"line_end ({self.line_end}) cannot be less than line_start ({self.line_start})")
        return self

    @model_validator(mode='after')
    def validate_file_path(self) -> 'Chunk':
        """Ensure the file_path is not empty and has an extension (rudimentary check)."""
        if not self.file_path.strip():
            raise ValueError("file_path cannot be empty or just whitespace.")
        # Basic check to ensure it looks like a file and not just a directory name
        if not Path(self.file_path).suffix:
            raise ValueError(f"file_path '{self.file_path}' does not appear to have a valid file extension.")
        return self


class EmbeddedChunk(Chunk):
    """
    Extends the base Chunk model to include the vector embedding.
    """
    vector: list[float] = Field(..., description="The dense vector embedding representing the chunk.")
