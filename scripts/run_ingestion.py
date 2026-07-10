#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add the 'src' directory to the Python path so we can import our modules
# when running as a standalone script.
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path.resolve()))

from ingestion.logger import setup_ingestion_logger
from ingestion.crawler import RepoCrawler
from ingestion.embed import embed_chunks
from ingestion.models import Chunk


def parse_args():
    parser = argparse.ArgumentParser(description="Run the RAG code ingestion pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ingestion.yaml",
        help="Path to the ingestion configuration YAML file."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_ingestion_logger()
    
    logger.info(f"Starting ingestion pipeline using config: {args.config}")
    
    # ---------------------------------------------------------
    # 1. SETUP / CONFIG LOADING (Stubbed)
    # ---------------------------------------------------------
    logger.info("[STEP 1/5] Loading configuration...")
    # In reality, you'd parse args.config here using pyyaml
    repo_path = "../scikit-learn"
    include_exts = {".py"}
    
    # ---------------------------------------------------------
    # 2. CRAWLING
    # ---------------------------------------------------------
    logger.info(f"[STEP 2/5] Crawling repository at {repo_path}...")
    # crawler = RepoCrawler(repo_path=repo_path, include_extensions=include_exts)
    # file_paths = list(crawler.walk())
    
    # Stubbing the file paths for the dry run shape
    file_paths = [Path(f"{repo_path}/dummy_file_1.py"), Path(f"{repo_path}/dummy_file_2.py")]
    logger.info(f"Crawled {len(file_paths)} files matching criteria.")
    
    # ---------------------------------------------------------
    # 3. CHUNKING (AST Parsing)
    # ---------------------------------------------------------
    logger.info("[STEP 3/5] Parsing files into AST chunks...")
    chunks = []
    for file_path in file_paths:
        # STUB: Here is where you will initialize tree-sitter, parse the file, 
        # extract nodes, and validate them into our Pydantic `Chunk` model.
        logger.info(f"  -> Chunking {file_path.name}")
        
        # Creating a dummy chunk just to feed the next step
        dummy_chunk = Chunk(
            content="def dummy_func():\n    pass",
            file_path=str(file_path),
            chunk_type="function",
            function_name="dummy_func",
            line_start=1,
            line_end=2
        )
        chunks.append(dummy_chunk)
        
    logger.info(f"Generated a total of {len(chunks)} chunks.")

    # ---------------------------------------------------------
    # 4. EMBEDDING
    # ---------------------------------------------------------
    logger.info("[STEP 4/5] Generating vector embeddings...")
    # STUB: Currently returns 1536-dim vectors of zeroes.
    embedded_chunks = embed_chunks(chunks)
    logger.info(f"Successfully embedded {len(embedded_chunks)} chunks.")

    # ---------------------------------------------------------
    # 5. INDEXING (Vector Database)
    # ---------------------------------------------------------
    logger.info("[STEP 5/5] Indexing into pgvector database...")
    # STUB: Here is where you will connect to pgvector via psycopg2/SQLAlchemy/Langchain
    # and batch insert the `embedded_chunks`.
    for idx, echunk in enumerate(embedded_chunks):
        logger.info(f"  -> Inserted vector {idx+1} for {echunk.function_name} ({len(echunk.vector)} dims)")

    logger.info("Pipeline execution complete! 🎉")


if __name__ == "__main__":
    main()
