"""
Retrieval functions module.

Provides implementations for:
- Dense retrieval via FAISS similarity search (inner product/cosine) using embeddings from `embeddings.py`
- BM25 retrieval via Pyserini (when a Lucene index is available)

Note:
- BM25/Pyserini is a reserved TODO and disabled by default in this demo. The code will try to
  load a Lucene index only if present; otherwise it returns None and skips BM25.

Exposed functions:
- dense_search: Dense similarity search over a FAISS index
- bm25_prepare: Prepare a Pyserini LuceneSearcher (optional)
- bm25_search: Run BM25 search if a searcher is available
"""
import os
import logging
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np

from .config import Config
from .embeddings import encode_query
from .index import FaissIndex


logger = logging.getLogger(__name__)


def dense_search(
    cfg: Config, 
    faiss_index: FaissIndex, 
    embedder: Any, 
    doc_ids: Sequence[str], 
    query: str, 
    topk: int
) -> List[Tuple[str, float]]:
    """Perform dense vector retrieval.
    
    Uses a FAISS index to perform similarity search (inner product/cosine)
    and returns the top-k most similar documents to the query vector.
    
    Args:
        cfg: The config object (unused here but kept for interface consistency)
        faiss_index: The FAISS index object
        embedder: The embedding model
        doc_ids: List of document IDs corresponding to the index order
        query: The query text
        topk: Number of documents to return
        
    Returns:
        List[Tuple[str, float]]: A list of (document ID, similarity score) sorted by score descending
        
    Raises:
        ValueError: When the query is empty or topk is invalid
        Exception: When an error occurs during retrieval
    """
    if not query or not query.strip():
        raise ValueError("Query text cannot be empty")
    
    if topk <= 0:
        raise ValueError(f"topk must be a positive number, but got: {topk}")
    
    try:
        # Encode the query into a vector
        qvec = encode_query(embedder, query)
        
        # Perform FAISS search
        distances, indices = faiss_index.search(qvec, topk)
        
        # Convert the results format
        hits = []
        for rank, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(doc_ids):  # Ensure the index is valid
                doc_id = doc_ids[idx]
                score = float(distances[0][rank])
                hits.append((doc_id, score))
            else:
                logger.warning(f"Invalid document index: {idx}")
        
        logger.debug(f"Dense retrieval completed, returned {len(hits)} results")
        return hits
        
    except Exception as e:
        logger.error(f"An error occurred during dense retrieval: {e}")
        raise


def bm25_prepare(cfg: Config) -> Optional[Any]:  # Optional[LuceneSearcher]
    """Prepare the BM25 retriever.
    
    Tries to load Pyserini's Lucene searcher. If Pyserini is not installed
    or the index does not exist, returns None.
    
    Args:
        cfg: Config object containing Lucene index path and BM25 parameters
        
    Returns:
        Optional[LuceneSearcher]: The configured searcher, or None
    """
    try:
        # Lazy import to avoid hard dependency
        from pyserini.search.lucene import LuceneSearcher
        
        # Check if the index directory exists and is not empty
        if not os.path.isdir(cfg.lucene_dir):
            logger.warning(f"Lucene index directory does not exist: {cfg.lucene_dir}")
            return None
            
        if not os.listdir(cfg.lucene_dir):
            logger.warning(f"Lucene index directory is empty: {cfg.lucene_dir}")
            return None
        
        # Create the searcher
        logger.debug(f"Loading Lucene index: {cfg.lucene_dir}")
        searcher = LuceneSearcher(cfg.lucene_dir)
        
        # Set BM25 parameters
        searcher.set_bm25(cfg.bm25_k1, cfg.bm25_b)
        logger.debug(f"BM25 parameters set: k1={cfg.bm25_k1}, b={cfg.bm25_b}")
        
        return searcher
        
    except ImportError:
        logger.warning("Pyserini is not installed, skipping BM25 retrieval")
        return None
    except Exception as e:
        logger.warning(f"An error occurred while preparing the BM25 retriever: {e}")
        return None


def bm25_search(
    searcher: Optional[Any],  # Optional[LuceneSearcher]
    query: str, 
    topk: int
) -> List[Tuple[str, float]]:
    """Perform BM25 retrieval.
    
    Args:
        searcher: The Lucene searcher, which may be None
        query: The query text
        topk: Number of documents to return
        
    Returns:
        List[Tuple[str, float]]: A sorted list of (document ID, score)
        
    Raises:
        ValueError: When parameters are invalid
    """
    if searcher is None:
        logger.debug("BM25 retriever is not enabled")
        return []
    
    if not query or not query.strip():
        raise ValueError("Query text cannot be empty")
    
    if topk <= 0:
        raise ValueError(f"topk must be a positive number, but got: {topk}")
    
    try:
        # Perform retrieval
        hits = searcher.search(query, k=topk)
        
        # Convert the results format
        results = [(h.docid, float(h.score)) for h in hits]
        
        logger.debug(f"BM25 retrieval completed, returned {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"An error occurred during BM25 retrieval: {e}")
        raise


