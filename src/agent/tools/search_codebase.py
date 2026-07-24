# ruff: noqa: E501
import logging
from typing import Any, Dict

import psycopg2
import yaml
from pgvector.psycopg2 import register_vector
from psycopg2 import sql

from agent.tools.base import BaseTool
from retrieval.pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)


class SearchCodebaseTool(BaseTool):
    """
    Agent tool wrapper for the Phase 2 Retrieval Pipeline.
    Allows the Agent to perform semantic vector searches across the codebase.
    
    Implements the context manager protocol for safe DB connection cleanup:
        with SearchCodebaseTool() as tool:
            tool.execute({"query": "..."})
    """
    
    def __init__(self, config_path: str = "configs/agent.yaml", db_config_path: str = "configs/ingestion.yaml"):
        # Load Retrieval Preset from agent config
        with open(config_path, "r") as f:
            agent_config = yaml.safe_load(f)
            # Default to hybrid if not strictly defined, or use the preset logic
            self.method = "hybrid" if agent_config["agent"].get("retrieval_preset") == "lora_fine_tuned" else "dense"
            
        # Load DB Config
        with open(db_config_path, "r") as f:
            db_config = yaml.safe_load(f)
            self.conn_str = db_config["database"]["connection_string"]
            self.table_name = db_config["database"].get("collection_name", "chunks")
            
        # Establish persistent connection and pipeline
        self.conn = psycopg2.connect(self.conn_str)
        register_vector(self.conn)
        self.pipeline = RetrievalPipeline(self.conn, self.table_name)

    def __enter__(self) -> "SearchCodebaseTool":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Explicitly close the DB connection. Idempotent."""
        if hasattr(self, "conn") and self.conn and not self.conn.closed:
            self.conn.close()
            logger.debug("SearchCodebaseTool DB connection closed.")
        
    @property
    def name(self) -> str:
        return "search_codebase"
        
    @property
    def description(self) -> str:
        return (
            "Performs a semantic vector search across the entire codebase. "
            "Use this to find where specific concepts, classes, or functions are implemented. "
            "Args must be a JSON object with a single 'query' string key."
        )

    def execute(self, args: Dict[str, Any]) -> str:
        query = args.get("query")
        if not query:
            return "ERROR: Missing 'query' argument."
            
        try:
            # 1. Retrieve the top-K chunk IDs
            retrieved_chunk_ids = self.pipeline.retrieve(query, self.method, k=5)
            
            if not retrieved_chunk_ids:
                return "No semantic overlap found in the codebase. Try a different search strategy."
                
            # 2. Fetch the actual raw Python strings for those IDs
            with self.conn.cursor() as cur:
                format_strings = sql.SQL(', ').join([sql.Placeholder()] * len(retrieved_chunk_ids))
                cur.execute(
                    sql.SQL("SELECT id, content FROM {} WHERE id IN ({})").format(
                        sql.Identifier(self.table_name), format_strings
                    ),
                    tuple(retrieved_chunk_ids),
                )
                rows = cur.fetchall()
                
            # 3. Format as a string for the Agent's clipboard
            result_str = ""
            for row in rows:
                chunk_id = row[0]
                content = row[1]
                result_str += f"\n--- SNIPPET ID: {chunk_id} ---\n{content}\n"
                
            return result_str
            
        except Exception as e:
            return f"ERROR during codebase search: {str(e)}"

    def get_function_declaration(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": "The search term or concept to look up in the codebase (e.g., 'RandomForest validate inputs')."
                    }
                },
                "required": ["query"]
            }
        }
