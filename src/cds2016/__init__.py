"""
CDS-2016 retrieval and evaluation module.

This package provides:
- Dense retrieval (FAISS similarity search: inner product/cosine)
- Reranking (Gemma Reranker)
- Standard IR evaluation metrics (nDCG, MAP, Recall, Precision)

Reserved features (TODO, disabled by default):
- BM25 retrieval (Pyserini / Lucene index)

Main components:
- Config: configuration management
- FaissIndex: FAISS vector index management
- Retrieval functions: dense_search (bm25_* reserved)
- Reranking functions: rerank (Gemma)
- Evaluation: evaluate_runs
"""

from .config import Config
from .index import FaissIndex
from .search import dense_search, bm25_search, bm25_prepare
from .rerank import build_gemma_reranker, rerank_gemma
from .evaluate import evaluate_runs
from .embeddings import build_embedder, encode_texts, encode_query
from .data import load_dataset, load_queries, load_qrels, collect_docs

# Public API
__all__ = [
    # configuration
    "Config",
    
    # index management
    "FaissIndex",
    
    # data loading
    "load_dataset",
    "load_queries", 
    "load_qrels",
    "collect_docs",
    
    # embeddings
    "build_embedder",
    "encode_texts",
    "encode_query",
    
    # retrieval
    "dense_search",
    "bm25_search",
    "bm25_prepare",
    
    # reranking
    "build_gemma_reranker",
    "rerank_gemma",
    
    # evaluation
    "evaluate_runs"
]


