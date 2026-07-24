# ruff: noqa: E501
import ast
import logging
from typing import Any, Dict

import psycopg2
import yaml
from psycopg2 import sql

from agent.tools.base import BaseTool

logger = logging.getLogger(__name__)


class ASTSearchTool(BaseTool):
    """
    Deterministic structural search tool for finding definitions, callers, 
    and inheritance graphs using Abstract Syntax Trees.
    
    Implements the context manager protocol for safe DB connection cleanup:
        with ASTSearchTool() as tool:
            tool.execute({"query_type": "find_definition", "target_name": "fit"})
    """

    def __init__(self, db_config_path: str = "configs/ingestion.yaml"):
        with open(db_config_path, "r") as f:
            db_config = yaml.safe_load(f)
            self.conn_str = db_config["database"]["connection_string"]
            self.table_name = db_config["database"].get("collection_name", "chunks")
            
        self.conn = psycopg2.connect(self.conn_str)

    def __enter__(self) -> "ASTSearchTool":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Explicitly close the DB connection. Idempotent."""
        if hasattr(self, "conn") and self.conn and not self.conn.closed:
            self.conn.close()
            logger.debug("ASTSearchTool DB connection closed.")

    @property
    def name(self) -> str:
        return "ast_search"

    @property
    def description(self) -> str:
        return (
            "Performs deterministic, structural AST searches across the codebase. "
            "Use this instead of semantic search when you know the exact name of a class or method, "
            "or when you need to navigate strict inheritance hierarchies. "
            "Supported query types: 'find_definition', 'find_callers', 'find_subclasses'."
        )

    def execute(self, args: Dict[str, Any]) -> str:
        query_type = args.get("query_type")
        target_name = args.get("target_name")

        if not query_type or not target_name:
            return "Error: Missing 'query_type' or 'target_name' parameter."

        if query_type == "find_definition":
            return self._stub_find_definition(target_name)
        elif query_type == "find_callers":
            return self._stub_find_callers(target_name)
        elif query_type == "find_subclasses":
            return self._stub_find_subclasses(target_name)
        else:
            return f"Error: Unsupported query_type '{query_type}'."

    def _stub_find_definition(self, name: str) -> str:
        results = []
        with self.conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT file_path, line_start, line_end, content FROM {} WHERE content LIKE %s").format(sql.Identifier(self.table_name)), (f"%{name}%",))
            for file_path, line_start, line_end, content in cur.fetchall():
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name:
                            results.append(f"{file_path} (Lines {line_start}-{line_end})")
                except Exception:
                    continue
        if not results:
            return f"Definition for '{name}' not found in AST."
        return "\n".join(set(results))

    def _stub_find_callers(self, name: str) -> str:
        results = []
        with self.conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT file_path, line_start, line_end, content FROM {} WHERE content LIKE %s").format(sql.Identifier(self.table_name)), (f"%{name}%",))
            for file_path, line_start, line_end, content in cur.fetchall():
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            if (isinstance(node.func, ast.Attribute) and node.func.attr == name) or \
                               (isinstance(node.func, ast.Name) and node.func.id == name):
                                results.append(f"{file_path} (Lines {line_start}-{line_end})")
                except Exception:
                    continue
        if not results:
            return f"No callers found for '{name}'."
        return "\n".join(set(results))

    def _stub_find_subclasses(self, base_class: str) -> str:
        results = []
        with self.conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT file_path, line_start, line_end, content FROM {} WHERE content LIKE %s").format(sql.Identifier(self.table_name)), (f"%{base_class}%",))
            for file_path, line_start, line_end, content in cur.fetchall():
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            for base in node.bases:
                                if isinstance(base, ast.Name) and base.id == base_class:
                                    results.append(f"class {node.name} in {file_path} (Lines {line_start}-{line_end})")
                except Exception:
                    continue
        if not results:
            return f"No subclasses found for '{base_class}'."
        return "\n".join(set(results))

    def get_function_declaration(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query_type": {
                        "type": "STRING",
                        "description": "The specific structural query to run. Must be one of: 'find_definition', 'find_callers', 'find_subclasses'."
                    },
                    "target_name": {
                        "type": "STRING",
                        "description": "The exact name of the class, function, or method to target (e.g., 'RandomForestClassifier', 'fit')."
                    }
                },
                "required": ["query_type", "target_name"]
            }
        }
