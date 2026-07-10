#!/usr/bin/env python3
import argparse
import sys
import time
import random
from collections import Counter
from pathlib import Path

import yaml
from tqdm import tqdm

# Add the 'src' directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path.resolve()))

from ingestion.logger import setup_ingestion_logger
from ingestion.crawler import RepoCrawler
from ingestion.chunker import ASTChunker
from ingestion.doc_chunker import DocChunker
from ingestion.embed import embed_chunks
from ingestion.indexer import PgVectorIndexer


def parse_args():
    parser = argparse.ArgumentParser(description="Run the RAG code ingestion pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ingestion.yaml",
        help="Path to the ingestion configuration YAML file."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run crawling and chunking only, print sample chunks, and skip embedding/indexing."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_ingestion_logger()
    
    logger.info(f"Starting ingestion pipeline using config: {args.config}")
    start_time = time.time()
    
    # ---------------------------------------------------------
    # 1. SETUP / CONFIG LOADING
    # ---------------------------------------------------------
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        sys.exit(1)
        
    base_repo_path = Path(config["repo"]["path"]).resolve()
    
    # The scoped subset requested by the user
    target_subpaths = [
        "sklearn/linear_model",
        "sklearn/ensemble",
        "sklearn/utils",
        "sklearn/base.py"
    ]
    
    include_exts = set(config["filtering"]["include_extensions"])
    exclude_exts = set(config["filtering"].get("exclude_extensions", []))
    exclude_dirs = set(config["filtering"].get("exclude_directories", []))
    
    # ---------------------------------------------------------
    # 2. CRAWLING
    # ---------------------------------------------------------
    logger.info("[STEP 1/4] Crawling repository subset...")
    all_files = []
    
    for subpath in target_subpaths:
        target_path = base_repo_path / subpath
        if not target_path.exists():
            logger.warning(f"Target path {target_path} does not exist. Skipping.")
            continue
            
        if target_path.is_file():
            # If it's an explicit file like base.py, just add it directly
            all_files.append(target_path)
        else:
            crawler = RepoCrawler(
                repo_path=target_path,
                include_extensions=include_exts,
                exclude_extensions=exclude_exts,
                exclude_directories=exclude_dirs
            )
            all_files.extend(list(crawler.walk()))
            
    logger.info(f"Crawled {len(all_files)} files matching criteria.")
    
    # ---------------------------------------------------------
    # 3. CHUNKING (AST & Doc Parsing)
    # ---------------------------------------------------------
    logger.info("[STEP 2/4] Parsing files into chunks...")
    ast_chunker = ASTChunker(max_lines_per_chunk=config["chunking"]["max_tokens"])
    doc_chunker = DocChunker()
    
    all_chunks = []
    
    for file_path in tqdm(all_files, desc="Chunking files", unit="file"):
        try:
            if file_path.suffix == ".py":
                all_chunks.extend(ast_chunker.chunk_file(file_path))
            elif file_path.suffix == ".md":
                all_chunks.extend(doc_chunker.chunk_file(file_path))
        except Exception as e:
            logger.error(f"Error chunking {file_path}: {e}")
            
    logger.info(f"Generated a total of {len(all_chunks)} chunks.")

    if args.dry_run:
        logger.info("\n" + "="*50)
        logger.info("DRY RUN ACTIVE: Skipping embedding and indexing.")
        logger.info("Printing sample chunks (10 random):")
        sample_chunks = random.sample(all_chunks, min(10, len(all_chunks))) if all_chunks else []
        for i, chunk in enumerate(sample_chunks):
            print(f"\n--- Sample {i+1} ---")
            print(f"File: {chunk.file_path}")
            print(f"Type: {chunk.chunk_type}")
            name = chunk.function_name or chunk.section_path or 'N/A'
            print(f"Name/Section: {name}")
            print(f"Lines: {chunk.line_start}-{chunk.line_end}")
            print("Content Snippet (first 150 chars):")
            print(f"{chunk.content[:150]}...")
        print("="*50 + "\n")
    else:
        # ---------------------------------------------------------
        # 4. EMBEDDING
        # ---------------------------------------------------------
        logger.info("[STEP 3/4] Generating vector embeddings via Jina API...")
        # The embed_chunks function internally logs its batches
        embedded_chunks = embed_chunks(all_chunks, batch_size=64)
        logger.info(f"Successfully embedded {len(embedded_chunks)} chunks.")

        # ---------------------------------------------------------
        # 5. INDEXING (Vector Database)
        # ---------------------------------------------------------
        logger.info("[STEP 4/4] Indexing into pgvector database...")
        db_config = config["database"]
        indexer = PgVectorIndexer(
            connection_string=db_config["connection_string"],
            table_name=db_config["collection_name"]
        )
        indexer.index_chunks(embedded_chunks)

    end_time = time.time()
    total_time = end_time - start_time

    # ---------------------------------------------------------
    # 6. SUMMARY
    # ---------------------------------------------------------
    chunk_counts = Counter(c.chunk_type for c in all_chunks)
    
    print("\n" + "="*50)
    print(" PIPELINE EXECUTION SUMMARY")
    print("="*50)
    print(f" Total Time Taken : {total_time:.2f} seconds")
    print(f" Files Processed  : {len(all_files)}")
    print(f" Total Chunks     : {len(all_chunks)}")
    print("-" * 50)
    print(" Chunks by Type:")
    for ctype, count in chunk_counts.items():
        print(f"  - {ctype.ljust(15)}: {count}")
    print("="*50 + "\n")
    
    logger.info("Pipeline execution complete! 🎉")


if __name__ == "__main__":
    main()
