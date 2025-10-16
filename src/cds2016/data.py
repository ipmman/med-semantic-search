"""
Data loading module.

Provides functionality to load the TREC CDS 2016 dataset from IR-Datasets.
"""
import logging
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple, Any

import ir_datasets
from tqdm import tqdm

from .config import Config


logger = logging.getLogger(__name__)


def load_dataset(cfg: Config) -> Any:  # ir_datasets.Dataset
    """Load an IR-Datasets dataset.
    
    Args:
        cfg: Config object containing the dataset ID
        
    Returns:
        ir_datasets.Dataset: The dataset object
        
    Raises:
        ValueError: When the dataset ID is invalid
    """
    try:
        logger.debug(f"Loading dataset: {cfg.dataset_id}")
        ds = ir_datasets.load(cfg.dataset_id)
        return ds
    except Exception as e:
        logger.error(f"Failed to load dataset {cfg.dataset_id}: {e}")
        raise ValueError(f"Invalid dataset ID: {cfg.dataset_id}") from e


def load_queries(ds: Any) -> List[Tuple[str, str]]:  # ds: ir_datasets.Dataset
    """Load query data.
    
    Args:
        ds: The dataset object
        
    Returns:
        List[Tuple[str, str]]: A list of (query_id, query_text)
    """
    queries: List[Tuple[str, str]] = []
    
    logger.debug("Loading queries...")
    for q in ds.queries_iter():
        # CDS 2016 queries have different fields, prioritize summary
        qtext = q.summary or q.description or q.note or ""
        # qtext = q.description or q.note or q.summary or ""
        if qtext.strip():
            queries.append((q.query_id, qtext))
        else:
            logger.warning(f"Empty query: {q.query_id}")
    
    logger.debug(f"Loaded {len(queries)} queries")
    return queries


def load_qrels(ds: Any) -> List[Any]:  # List[ir_datasets.Qrel]
    """Load relevance judgment data.
    
    Args:
        ds: The dataset object
        
    Returns:
        List[Qrel]: A list of relevance judgments
    """
    logger.debug("Loading relevance judgments...")
    qrels = list(ds.qrels_iter())
    logger.debug(f"Loaded {len(qrels)} relevance judgments")
    return qrels


def iter_docs(ds: Any, cfg: Config) -> Iterable[Tuple[str, str]]:  # ds: ir_datasets.Dataset
    """Iterate over and load document data.
    
    Args:
        ds: The dataset object
        cfg: Config object with max_docs and use_body settings
        
    Yields:
        Tuple[str, str]: A (doc_id, doc_text) pair
    """
    doc_count = 0
    empty_count = 0
    
    for d in ds.docs_iter():
        if cfg.max_docs and doc_count >= cfg.max_docs:
            logger.debug(f"Reached max docs limit: {cfg.max_docs}")
            break
        
        # Combine document content
        parts = []
        if d.title:
            parts.append(f"**Title:** {d.title.strip()}")
        if d.abstract:
            parts.append(f"**Abstract:** {d.abstract.strip()}")
        if cfg.use_body and hasattr(d, 'body') and d.body:
            parts.append(f"**Body:** {d.body.strip()}")
        
        text = "\n\n".join(parts)
        
        if text.strip():
            yield d.doc_id, text
            doc_count += 1
        else:
            empty_count += 1
            logger.debug(f"Empty document: {d.doc_id}")
    
    if empty_count > 0:
        logger.warning(f"Skipped {empty_count} empty documents")


def collect_docs(ds: Any, cfg: Config) -> Dict[str, str]:  # ds: ir_datasets.Dataset
    """Collect all document data into a dictionary.
    
    Args:
        ds: The dataset object
        cfg: The config object
        
    Returns:
        Dict[str, str]: A mapping from doc_id to document content
    """
    docid_to_text: Dict[str, str] = {}
    
    logger.debug("Collecting document data...")
    for doc_id, text in tqdm(iter_docs(ds, cfg), desc="Collecting documents", total=cfg.max_docs if cfg.max_docs else None):
        docid_to_text[doc_id] = text
    
    logger.debug(f"Collected {len(docid_to_text)} documents in total")
    return docid_to_text


