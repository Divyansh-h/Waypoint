import logging
from typing import Any, List, Optional

import yaml
from psycopg2 import sql

from ingestion.embed import get_jina_embeddings
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import Reranker, get_reranker

logger = logging.getLogger(__name__)

class RetrievalPipeline:
    def __init__(
        self,
        conn: Any,
        table_name: str,
        config_path: str = "configs/ingestion.yaml",
        rrf_k_override: Optional[int] = None,
        pool_size_override: Optional[int] = None,
    ):
        self.conn = conn
        self.table_name = table_name
        self.rrf_k_override = rrf_k_override
        self.pool_size_override = pool_size_override
        
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config {config_path}: {e}")
            self.config = {}
            
        retrieval_cfg = self.config.get("retrieval", {})
        
        if self.pool_size_override is not None:
            self.pool_size = self.pool_size_override
        else:
            self.pool_size = retrieval_cfg.get("candidate_pool_size", 30)
            
        if self.rrf_k_override is not None:
            self.rrf_k = self.rrf_k_override
        else:
            self.rrf_k = 60
            
        self.model_name = retrieval_cfg.get("reranker_model", "stub")
        self.cache_dir = retrieval_cfg.get("model_cache_dir", ".models_cache")
        
        self._reranker: Optional[Reranker] = None
        
    def _get_reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = get_reranker(self.model_name, self.cache_dir)
        return self._reranker

    def _retrieve_dense(self, query: str, k: int) -> List[str]:
        embeddings = get_jina_embeddings([query])
        if not embeddings:
            return []
            
        with self.conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT id FROM {} ORDER BY embedding <=> %s::vector LIMIT %s").format(
                    sql.Identifier(self.table_name)
                ),
                (embeddings[0], k),
            )
            return [row[0] for row in cur.fetchall()]

    def _retrieve_bm25(self, query: str, k: int) -> List[str]:
        with self.conn.cursor() as cur:
            cur.execute(
                sql.SQL("""
                SELECT id FROM {}
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                ORDER BY ts_rank(
                    to_tsvector('english', content), 
                    plainto_tsquery('english', %s)
                ) DESC
                LIMIT %s
                """).format(sql.Identifier(self.table_name)),
                (query, query, k),
            )
            return [row[0] for row in cur.fetchall()]

    def _retrieve_hybrid(self, query: str, k: int) -> List[str]:
        embeddings = get_jina_embeddings([query])
        if not embeddings:
            return []
            
        tbl = sql.Identifier(self.table_name)
        rrf_k = sql.Literal(self.rrf_k)
        with self.conn.cursor() as cur:
            cur.execute(
                sql.SQL("""
                WITH semantic_search AS (
                    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) as rank 
                    FROM {tbl} 
                    LIMIT %s
                ), keyword_search AS (
                    SELECT id, ROW_NUMBER() OVER (
                        ORDER BY ts_rank(
                            to_tsvector('english', content), 
                            plainto_tsquery('english', %s)
                        ) DESC
                    ) as rank
                    FROM {tbl} 
                    WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s) 
                    LIMIT %s
                )
                SELECT COALESCE(s.id, k.id) as id, 
                       s.rank as dense_rank, 
                       k.rank as bm25_rank,
                       COALESCE(1.0 / ({rrf_k} + s.rank), 0.0) + 
                       COALESCE(1.0 / ({rrf_k} + k.rank), 0.0) as rrf_score
                FROM semantic_search s FULL OUTER JOIN keyword_search k ON s.id = k.id
                ORDER BY rrf_score DESC 
                LIMIT %s;
                """).format(tbl=tbl, rrf_k=rrf_k),
                (embeddings[0], k, query, query, k, k),
            )
            return [row[0] for row in cur.fetchall()]

    def _retrieve_reranked_hybrid(self, query: str, k: int) -> List[str]:
        reranker = self._get_reranker()
        embeddings = get_jina_embeddings([query])
        if not embeddings:
            return []
            
        with self.conn.cursor() as cur:
            # Dense Retrieval
            cur.execute(
                sql.SQL("SELECT id, content FROM {} ORDER BY embedding <=> %s::vector LIMIT %s").format(
                    sql.Identifier(self.table_name)
                ),
                (embeddings[0], self.pool_size),
            )
            dense_results = [{"id": r[0], "content": r[1]} for r in cur.fetchall()]
            
            # BM25 Retrieval
            cur.execute(
                sql.SQL("""
                SELECT id, content FROM {}
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                ORDER BY ts_rank(
                    to_tsvector('english', content), 
                    plainto_tsquery('english', %s)
                ) DESC
                LIMIT %s
                """).format(sql.Identifier(self.table_name)),
                (query, query, self.pool_size),
            )
            bm25_results = [{"id": r[0], "content": r[1]} for r in cur.fetchall()]
            
        # Fusion
        candidates = reciprocal_rank_fusion(
            dense_results, bm25_results, k=self.rrf_k, top_n=self.pool_size
        )
        
        # Rerank
        ranked_candidates = reranker.rerank(query, candidates)
        return [c["id"] for c in ranked_candidates[:k]]

    def retrieve(self, query: str, method: str = "dense", k: int = 10) -> List[str]:
        """
        Public entrypoint for executing retrieval using the specified method preset.
        """
        method_map = {
            "dense": self._retrieve_dense,
            "bm25": self._retrieve_bm25,
            "hybrid": self._retrieve_hybrid,
            "reranked_hybrid": self._retrieve_reranked_hybrid,
        }
        
        if method not in method_map:
            raise ValueError(f"Unknown retrieval method: {method}")
            
        try:
            return method_map[method](query, k)
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Retrieval pipeline failed for method '{method}': {e}")
            self.conn.rollback()
            return []
