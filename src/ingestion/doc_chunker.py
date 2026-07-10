import re
from pathlib import Path
from typing import List, Tuple

from ingestion.models import Chunk


class DocChunker:
    """
    Parses Markdown documentation by header hierarchy.
    Collects content until the next header and maintains a breadcrumb
    section_path (e.g., 'Installation > Requirements').
    """

    def __init__(self, max_lines_per_chunk: int = 150) -> None:
        self.max_lines_per_chunk = max_lines_per_chunk
        # Regex to match markdown headers (e.g., "## My Header")
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.*)')

    def chunk_file(self, file_path: Path) -> List[Chunk]:
        """
        Reads a Markdown file, chunks it by headers, and returns Chunk models.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            raise IOError(f"Failed to read file {file_path}: {e}")

        chunks: List[Chunk] = []
        # Keeps track of the active header hierarchy: List of (level, title)
        current_path: List[Tuple[int, str]] = [] 
        
        current_content_lines: List[str] = []
        chunk_start_line = 1
        
        def commit_chunk(end_line: int) -> None:
            """Helper to finalize the active block and convert it to a Chunk."""
            if not current_content_lines:
                return
                
            content = "".join(current_content_lines).strip()
            if not content:
                return
                
            # Build breadcrumb string
            path_str = " > ".join(title for _, title in current_path) if current_path else "Root"
            
            # EDGE CASE: God Sections (Massive markdown blocks)
            # If the text between headers is massively long, we sub-split it.
            if len(current_content_lines) > self.max_lines_per_chunk:
                for i in range(0, len(current_content_lines), self.max_lines_per_chunk):
                    sub_lines = current_content_lines[i : i + self.max_lines_per_chunk]
                    sub_content = "".join(sub_lines).strip()
                    if sub_content:
                        part_number = (i // self.max_lines_per_chunk) + 1
                        chunks.append(Chunk(
                            content=sub_content,
                            file_path=str(file_path),
                            chunk_type="markdown_part",
                            section_path=f"{path_str} (Part {part_number})",
                            line_start=chunk_start_line + i,
                            line_end=chunk_start_line + i + len(sub_lines) - 1
                        ))
            else:
                chunks.append(Chunk(
                    content=content,
                    file_path=str(file_path),
                    chunk_type="markdown",
                    section_path=path_str,
                    line_start=chunk_start_line,
                    line_end=end_line
                ))
            
            current_content_lines.clear()

        # Iterate through lines to build chunks
        for idx, line in enumerate(lines):
            line_num = idx + 1
            match = self.header_pattern.match(line)
            
            if match:
                # We found a new header, commit the previous block
                commit_chunk(line_num - 1)
                chunk_start_line = line_num
                
                level = len(match.group(1))
                title = match.group(2).strip()
                
                # Pop off headers from the hierarchy that are at the same or deeper level
                # e.g., If we hit a `##`, we discard the active `###` and `##`.
                while current_path and current_path[-1][0] >= level:
                    current_path.pop()
                    
                current_path.append((level, title))
                
            current_content_lines.append(line)
            
        # Commit the final block at the end of the file
        commit_chunk(len(lines))
        
        return chunks
