#!/usr/bin/env python3
import argparse
import sys
import time
import random
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any

import yaml
from tqdm import tqdm

# Add the 'src' directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path.resolve()))

from utils.logger import setup_logger
from ingestion.crawler import RepoCrawler
from ingestion.chunker import ASTChunker
from ingestion.doc_chunker import DocChunker
from ingestion.embed import embed_chunks
from ingestion.indexer import PgVectorIndexer
from ingestion.models import Chunk


def parse_args():
    parser = argparse.ArgumentParser(description="Run the RAG code ingestion pipeline.")
    parser.add_argument("--config", type=str, default="configs/ingestion.yaml", help="Path to the ingestion configuration YAML file.")
    parser.add_argument("--dry-run", action="store_true", help="Run crawling and chunking only, print sample chunks, and skip embedding/indexing.")
    return parser.parse_args()


def load_config(config_path: str, logger) -> Dict[str, Any]:
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        if not isinstance(config, dict):
            raise ValueError("Config file must contain a YAML dictionary.")
            
        # Validate structurally required keys (excluding optionals)
        required_keys = [
            ("repo", "path"),
            ("filtering", "include_extensions"),
            ("chunking", "max_tokens"),
            ("database", "connection_string"),
        ]
        
        for keys in required_keys:
            current_level = config
            for key in keys:
                if not isinstance(current_level, dict) or key not in current_level:
                    raise ValueError(f"Missing or malformed required config key: '{' -> '.join(keys)}'")
                current_level = current_level[key]
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Malformed YAML in config file: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}")
        sys.exit(1)


def crawl_repository(base_repo_path: Path, target_subpaths: List[str], config: Dict[str, Any], logger) -> List[Path]:
    logger.info("[STEP 1/4] Crawling repository subset...")
    
    include_exts = set(config["filtering"]["include_extensions"])
    exclude_exts = set(config["filtering"].get("exclude_extensions", []))
    exclude_dirs = set(config["filtering"].get("exclude_directories", []))
    
    all_files = []
    for subpath in target_subpaths:
        target_path = base_repo_path / subpath
        if not target_path.exists():
            logger.warning(f"Target path {target_path} does not exist. Skipping.")
            continue
            
        if target_path.is_file():
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
    return all_files


def chunk_files(all_files: List[Path], base_repo_path: Path, max_tokens: int, logger) -> List[Chunk]:
    logger.info("[STEP 2/4] Parsing files into chunks...")
    ast_chunker = ASTChunker(max_lines_per_chunk=max_tokens)
    doc_chunker = DocChunker()
    
    all_chunks = []
    
    for file_path in tqdm(all_files, desc="Chunking files", unit="file"):
        try:
            chunks = []
            if file_path.suffix == ".py":
                chunks = ast_chunker.chunk_file(file_path)
            elif file_path.suffix == ".md":
                chunks = doc_chunker.chunk_file(file_path)
                
            try:
                rel_path = str(file_path.relative_to(base_repo_path))
            except ValueError:
                rel_path = str(file_path)
                
            for chunk in chunks:
                chunk.file_path = rel_path
                
            all_chunks.extend(chunks)
        except Exception as e:
            logger.warning(f"Unexpected error chunking {file_path}, skipping. Details: {e}")
            
    logger.info(f"Generated a total of {len(all_chunks)} chunks.")
    return all_chunks


def run_dry_run(all_chunks: List[Chunk], logger):
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


def run_full_ingestion(all_chunks: List[Chunk], db_config: Dict[str, Any], logger):
    logger.info("[STEP 3/4] Generating vector embeddings via Jina API...")
    embedded_chunks = embed_chunks(all_chunks, batch_size=64)
    logger.info(f"Successfully embedded {len(embedded_chunks)} chunks.")

    logger.info("[STEP 4/4] Indexing into pgvector database...")
    indexer = PgVectorIndexer(
        connection_string=db_config["connection_string"],
        table_name=db_config.get("collection_name", "chunks")
    )
    indexer.index_chunks(embedded_chunks)


def print_summary(total_time: float, num_files: int, all_chunks: List[Chunk]):
    chunk_counts = Counter(c.chunk_type for c in all_chunks)
    print("\n" + "="*50)
    print(" PIPELINE EXECUTION SUMMARY")
    print("="*50)
    print(f" Total Time Taken : {total_time:.2f} seconds")
    print(f" Files Processed  : {num_files}")
    print(f" Total Chunks     : {len(all_chunks)}")
    print("-" * 50)
    print(" Chunks by Type:")
    for ctype, count in chunk_counts.items():
        print(f"  - {ctype.ljust(15)}: {count}")
    print("="*50 + "\n")


def main():
    args = parse_args()
    logger = setup_logger("ingestion_pipeline")
    
    logger.info(f"Starting ingestion pipeline using config: {args.config}")
    start_time = time.time()
    
    config = load_config(args.config, logger)
    base_repo_path = Path(config["repo"]["path"]).resolve()
    
    # The scoped subset requested by the user
    target_subpaths = [
        "sklearn/linear_model",
        "sklearn/ensemble",
        "sklearn/utils",
        "sklearn/base.py"
    ]
    
    all_files = crawl_repository(base_repo_path, target_subpaths, config, logger)
    all_chunks = chunk_files(all_files, base_repo_path, config["chunking"]["max_tokens"], logger)
    
    if args.dry_run:
        run_dry_run(all_chunks, logger)
    else:
        run_full_ingestion(all_chunks, config["database"], logger)
        
    total_time = time.time() - start_time
    print_summary(total_time, len(all_files), all_chunks)
    logger.info("Pipeline execution complete! 🎉")


if __name__ == "__main__":
    main()
