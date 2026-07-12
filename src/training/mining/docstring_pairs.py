import ast
from typing import List, Dict, Any
from training.schema import TrainingPair, PositiveChunk

# Note: Update this type hint to use your actual Chunk dataclass/schema once integrated
Chunk = Dict[str, Any]

def is_trivial(docstring: str) -> bool:
    """Returns True if the docstring is a trivial one-liner (less than 5 words)."""
    lines = docstring.strip().splitlines()
    words = docstring.split()
    if len(lines) <= 1 and len(words) < 5:
        return True
    return False

def mine_docstring_pairs(chunks: List[Chunk]) -> List[TrainingPair]:
    """
    Mines (query, relevant-chunk) positive pairs by extracting docstrings as the query
    and using the function/class body as the positive chunk.
    
    Args:
        chunks: A list of code chunks parsed from the AST.
        
    Returns:
        A list of TrainingPair dictionaries ready for negative-mining or filtering.
    """
    pairs: List[TrainingPair] = []
    
    for chunk in chunks:
        content = chunk.get("content", "")
        if not content:
            continue
            
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue
            
        if not tree.body:
            continue
            
        node = tree.body[0]
        # We only extract from classes and functions
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
            
        docstring = ast.get_docstring(node)
        if not docstring or is_trivial(docstring):
            continue
            
        # Extract the first summary sentence/line to use as the 'anchor'
        lines = docstring.strip().splitlines()
        summary = lines[0].strip() if lines else docstring
        
        # 3. Strip the docstring from the raw code to use the remaining implementation as 'positive'.
        # We use the AST node line numbers to safely slice it out while preserving comments.
        positive_content = content
        if node.body and isinstance(node.body[0], ast.Expr):
            expr = node.body[0]
            if hasattr(expr.value, "value") and isinstance(expr.value.value, str):
                start_lineno = expr.lineno - 1 # 0-indexed
                end_lineno = expr.end_lineno if hasattr(expr, "end_lineno") else expr.lineno
                
                content_lines = content.splitlines()
                # Slice out the exact lines containing the docstring
                positive_content = "\\n".join(content_lines[:start_lineno] + content_lines[end_lineno:])
        
        pairs.append(
            TrainingPair(
                anchor=summary,
                positive=PositiveChunk(
                    chunk_id=chunk.get("chunk_id", "unknown"),
                    content=positive_content.strip(),
                    file_path=chunk.get("file_path")
                ),
                negatives=[],
                source="mined",
                metadata={"extraction_method": "docstring_summary"}
            )
        )
        
    return pairs
