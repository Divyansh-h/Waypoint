from pathlib import Path
from typing import List, Optional

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node

from ingestion.models import Chunk


class ASTChunker:
    """
    Parses Python source code using Tree-sitter to extract structurally meaningful
    chunks (Functions, Classes, and Methods), while handling edge cases like
    decorators, God nodes (massive functions), and nested structures.
    """

    def __init__(self, max_lines_per_chunk: int = 100) -> None:
        self.language = Language(tspython.language())
        self.parser = Parser(self.language)
        self.max_lines_per_chunk = max_lines_per_chunk

    def chunk_file(self, file_path: Path) -> List[Chunk]:
        """
        Reads a Python file, parses its AST, and extracts chunks.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
        except Exception as e:
            raise IOError(f"Failed to read file {file_path}: {e}")

        tree = self.parser.parse(bytes(source_code, "utf8"))
        chunks: List[Chunk] = []

        def extract_code(node: Node) -> str:
            return source_code[node.start_byte:node.end_byte]

        def handle_god_node(chunk: Chunk) -> List[Chunk]:
            """
            EDGE CASE 2: Very long functions (God Nodes).
            If a node exceeds the maximum allowed lines, we fallback to a naive
            line-based split within the node to ensure it fits in the embedding window.
            """
            lines = chunk.content.split('\n')
            total_lines = len(lines)
            
            if total_lines <= self.max_lines_per_chunk:
                return [chunk]
            
            sub_chunks = []
            for i in range(0, total_lines, self.max_lines_per_chunk):
                sub_lines = lines[i : i + self.max_lines_per_chunk]
                part_number = (i // self.max_lines_per_chunk) + 1
                
                sub_chunks.append(Chunk(
                    content='\n'.join(sub_lines),
                    file_path=chunk.file_path,
                    chunk_type=f"{chunk.chunk_type}_part",
                    function_name=f"{chunk.function_name} (Part {part_number})",
                    line_start=chunk.line_start + i,
                    line_end=chunk.line_start + i + len(sub_lines) - 1
                ))
            return sub_chunks

        def walk(node: Node, context_prefix: Optional[str] = None) -> None:
            if node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                func_name = extract_code(name_node) if name_node else "unknown"
                full_name = f"{context_prefix}.{func_name}" if context_prefix else func_name
                
                # EDGE CASE 1: Decorated Functions.
                # In Tree-sitter, a decorated function is a child of a `decorated_definition` node.
                # If we just extract `function_definition`, we lose the `@decorator` lines above it.
                # We check if the parent is a `decorated_definition` and use its boundaries instead.
                target_node = node.parent if node.parent and node.parent.type == "decorated_definition" else node
                
                base_chunk = Chunk(
                    content=extract_code(target_node),
                    file_path=str(file_path),
                    chunk_type="method" if context_prefix else "function",
                    function_name=full_name,
                    line_start=target_node.start_point.row + 1,
                    line_end=target_node.end_point.row + 1
                )
                chunks.extend(handle_god_node(base_chunk))
                
                # We do NOT recurse into functions here, so closures are kept intact 
                # inside the parent function (unless split by the God Node handler).
                return

            if node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                cls_name = extract_code(name_node) if name_node else "unknown"
                
                # EDGE CASE 3: Nested Classes.
                # By updating the context_prefix dynamically, a class inside a class
                # or a class inside a function correctly gets prefixed (e.g., OuterClass.InnerClass)
                new_context = f"{context_prefix}.{cls_name}" if context_prefix else cls_name
                
                # Same decorator check for classes
                target_node = node.parent if node.parent and node.parent.type == "decorated_definition" else node
                
                base_chunk = Chunk(
                    content=extract_code(target_node),
                    file_path=str(file_path),
                    chunk_type="class",
                    function_name=new_context,
                    line_start=target_node.start_point.row + 1,
                    line_end=target_node.end_point.row + 1
                )
                chunks.extend(handle_god_node(base_chunk))
                
                # Walk the class body to extract methods and nested classes
                body_node = node.child_by_field_name("body")
                if body_node:
                    for child in body_node.children:
                        walk(child, context_prefix=new_context)
                return

            # Keep walking down for unhandled nodes
            for child in node.children:
                walk(child, context_prefix=context_prefix)

        walk(tree.root_node)
        return chunks
