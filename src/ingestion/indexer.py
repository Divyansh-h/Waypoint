import hashlib
import json
import logging
from typing import Any, List

import psycopg2
from pgvector.psycopg2 import register_vector  # type: ignore
from psycopg2.extras import execute_batch

from ingestion.models import EmbeddedChunk

logger = logging.getLogger("ingestion_pipeline")


class PgVectorIndexer:
    """
    Handles indexing EmbeddedChunks into a PostgreSQL database using pgvector.
    This class abstracts the complexity of database connection management and 
    hybrid data storage (storing dense vectors alongside raw JSONB metadata), 
    ensuring that the ingestion pipeline stays decoupled from SQL syntax specifics.
    """

    def __init__(self, connection_string: str, table_name: str = "chunks", vector_dim: int = 1024):
        """
        Args:
            connection_string: Standard Postgres connection URI.
            table_name: The table to insert vectors into.
            vector_dim: Dimension size of the vectors. 
            
        Note (Flag): `vector_dim` defaults to 1024 (Jina v3 specific). If we swap embedding models,
        this hardcoded default could cause silent dimension mismatch errors in Postgres.
        """
        self.conn_str = connection_string
        self.table_name = table_name
        self.vector_dim = vector_dim

    def _generate_id(self, chunk: EmbeddedChunk) -> str:
        """
        Generates a stable SHA-256 ID based on the file path and the starting line number.
        This guarantees that re-running the pipeline will update the exact same chunk
        instead of creating a duplicate row, even if the content inside the chunk changed.
        This is the foundation of our idempotent ingestion strategy.
        """
        unique_string = f"{chunk.file_path}:{chunk.line_start}"
        return hashlib.sha256(unique_string.encode('utf-8')).hexdigest()

    def _setup_database(self, conn: Any) -> None:
        """
        Idempotently creates the vector extension and the chunks table if they do not exist.
        This exists so the pipeline can bootstrap itself on a fresh Postgres instance without
        requiring external manual database migration scripts.
        """
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # Register the vector type with the psycopg2 connection context
            register_vector(conn)
            
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id VARCHAR(64) PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    chunk_type VARCHAR(50) NOT NULL,
                    function_name TEXT,
                    section_path TEXT,
                    line_start INT NOT NULL,
                    line_end INT NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    embedding VECTOR({self.vector_dim})
                );
            """)
            
            # Optional: Add an HNSW index for much faster approximate nearest neighbor (ANN) search
            # Ensure index creation doesn't fail if it already exists
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
                ON {self.table_name} USING hnsw (embedding vector_cosine_ops);
            """)
        conn.commit()

    def index_chunks(self, embedded_chunks: List[EmbeddedChunk], batch_size: int = 100) -> None:
        """
        Executes a bulk upsert of EmbeddedChunks into the PostgreSQL database using `execute_batch`.
        This uses `ON CONFLICT` combined with our stable SHA-256 ID to ensure that re-running
        the pipeline updates existing vectors rather than polluting the DB with duplicate rows.
        It packs structural context into a `metadata` JSONB column, unlocking powerful hybrid 
        filtering (e.g. searching only within `sklearn/linear_model`).
        """
        if not embedded_chunks:
            logger.info("No chunks provided to indexer.")
            return

        try:
            conn = psycopg2.connect(self.conn_str)
            self._setup_database(conn)
            # Re-register just in case, though _setup_database handles it
            register_vector(conn)
        except Exception as e:
            logger.error(f"Failed to connect or setup database: {e}")
            raise

        # The ON CONFLICT clause enables upserting based on the stable ID
        query = f"""
            INSERT INTO {self.table_name} (
                id, file_path, chunk_type, function_name, section_path, 
                line_start, line_end, content, metadata, embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT (id) DO UPDATE SET
                file_path = EXCLUDED.file_path,
                chunk_type = EXCLUDED.chunk_type,
                function_name = EXCLUDED.function_name,
                section_path = EXCLUDED.section_path,
                line_start = EXCLUDED.line_start,
                line_end = EXCLUDED.line_end,
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding;
        """

        data = []
        for chunk in embedded_chunks:
            chunk_id = self._generate_id(chunk)
            
            # Pack key attributes into a flexible JSONB metadata payload
            # This makes filtering easier later if we want to query strictly via JSON operators
            metadata = {
                "file_path": chunk.file_path,
                "chunk_type": chunk.chunk_type,
                "lines": [chunk.line_start, chunk.line_end]
            }
            if chunk.function_name:
                metadata["function_name"] = chunk.function_name
            if chunk.section_path:
                metadata["section_path"] = chunk.section_path

            data.append((
                chunk_id,
                chunk.file_path,
                chunk.chunk_type,
                chunk.function_name,
                chunk.section_path,
                chunk.line_start,
                chunk.line_end,
                chunk.content,
                json.dumps(metadata),
                chunk.vector
            ))

        logger.info(f"Upserting {len(data)} chunks into '{self.table_name}' table...")
        try:
            with conn.cursor() as cur:
                # execute_batch sends data in pages to avoid massive single-query memory overhead
                execute_batch(cur, query, data, page_size=batch_size)
            conn.commit()
            logger.info("✅ Database indexing complete.")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to upsert chunks: {e}")
            raise
        finally:
            conn.close()
