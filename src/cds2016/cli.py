"""
CLI main program module.

Provides a command-line interface for CDS2016 experiments, integrating all retrieval, 
reranking, and evaluation functions.
"""
import os
import sys
import time
import logging
import argparse

import numpy as np
import pandas as pd

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from .config import Config
from .data import collect_docs, load_dataset, load_queries, load_qrels
from .embeddings import build_embedder, encode_texts, prep_text_for_embed
from .index import FaissIndex
from .search import dense_search, bm25_prepare, bm25_search
from .rerank import (
    build_reranker,
    rerank,
)
from .enhance import enhance_query
from .evaluate import evaluate_runs


logger = logging.getLogger(__name__)


def configure_logging(log_dir: Optional[Path] = None, verbose: bool = False, quiet: bool = False) -> None:
    """Configure the logging system.
    
    Args:
        log_dir: Directory to save log files, outputs to console only if None
        verbose: Whether to enable detailed logging (DEBUG level)
        quiet: Whether to only show warnings and above (WARNING level)
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"cds2016_{timestamp}.log"
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    # Root level is determined by user flags
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    # Ensure that DEBUG logs within the module can be captured by the root logger
    logging.getLogger("cds2016").setLevel(logging.DEBUG)


def log_experiment_info(cfg: Config) -> None:
    """Log experiment configuration information.
    
    Args:
        cfg: Experiment configuration
    """
    logger.info("=" * 80)
    logger.info("CDS2016 Experiment Start")
    logger.info("=" * 80)
    logger.info(f"Config:")
    logger.info(f"Device: {cfg.device}")
    logger.info(f"Dataset: {cfg.dataset_id}")
    logger.info(f"Enhance Query: {cfg.enhance_query}")
    logger.info(f"Embedding Model: {cfg.embed_model}")
    logger.info(f"Rerank Model: {cfg.rerank_model}")
    logger.info(f"Retrieval-TopK: {cfg.retrieval_topk}")
    logger.info(f"Rerank-K: {cfg.rerank_k}")
    logger.info(f"Final-K: {cfg.final_k}")
    logger.info(f"Evaluate at: {cfg.eval_at}")
    logger.info("=" * 80)


def run_all(cfg: Config) -> None:
    """Execute the complete retrieval experiment pipeline.
    
    Includes data loading, index building, retrieval, reranking, and evaluation.
    
    Args:
        cfg: Experiment configuration
        
    Raises:
        Exception: When an error occurs during the experiment
    """
    start_time = time.time()
    
    try:
        # Log experiment information
        log_experiment_info(cfg)
        
        # Load dataset
        logger.info(f"Loading dataset: {cfg.dataset_id}")
        ds = load_dataset(cfg)
        queries = load_queries(ds)
        qrels = load_qrels(ds)
        logger.info(f"Number of queries: {len(queries)}")
        logger.info(f"Number of relevance judgments: {len(qrels)}")

        # Collect documents
        logger.info("Collecting document data...")
        docid_to_text = collect_docs(ds, cfg)
        doc_ids = list(docid_to_text.keys())
        logger.info(f"Number of documents collected: {len(doc_ids)}")

        # Create embedding model and index
        logger.info("Creating embedding model...")
        embedder = build_embedder(cfg)
        faiss_index = FaissIndex(cfg)

        if faiss_index.exists():
            logger.info("Loading existing FAISS index...")
            faiss_index.load()
            doc_ids = faiss_index.doc_ids
            logger.info(f"Index loaded successfully, containing {len(doc_ids)} documents")
        else:
            logger.info("Creating new FAISS index...")
            logger.info("Preparing document texts...")
            texts = [prep_text_for_embed(docid_to_text[i]) for i in tqdm(doc_ids, desc="Processing documents")]
            
            logger.info("Encoding document vectors...")
            doc_vectors = encode_texts(embedder, texts, cfg.embedding_batch)

            logger.info("Building FAISS index...")
            # Let the build method determine GPU usage from config
            index = faiss_index.build(doc_vectors)

            logger.info("Saving index...")
            faiss_index.save(index, doc_ids, doc_vectors)
            logger.info(f"FAISS index saved to: {faiss_index.faiss_path}")
        

        # Prepare BM25 retriever
        logger.info("Preparing BM25 retriever...")
        bm25_searcher = bm25_prepare(cfg)
        if bm25_searcher:
            logger.info("BM25 retriever prepared successfully")
        else:
            logger.info("BM25 retriever not enabled or Pyserini not installed")

        # Prepare query variants: baseline and optional enhanced
        original_queries: List[Tuple[str, str]] = list(queries)
        enhanced_queries: List[Tuple[str, str]] = []
        if cfg.enhance_query:
            logger.info("Enhancing queries via LLM endpoint...")
            for qid, qtext in tqdm(original_queries, desc="Enhance queries"):
                new_q = enhance_query(qtext, cfg)
                enhanced_queries.append((qid, new_q))

        # Dense vector retrieval: baseline
        logger.info("Executing dense vector retrieval (baseline queries)...")
        dense_run_base: Dict[str, List[Tuple[str, float]]] = {}
        for qid, qtext in tqdm(original_queries, desc="Dense Retrieval [baseline]"):
            dense_run_base[qid] = dense_search(cfg, faiss_index, embedder, doc_ids, qtext, cfg.retrieval_topk)
        logger.info(f"Dense retrieval (baseline) completed, processed {len(dense_run_base)} queries")

        # Dense vector retrieval: enhanced (optional)
        dense_run_enh: Dict[str, List[Tuple[str, float]]] = {}
        if enhanced_queries:
            logger.info("Executing dense vector retrieval (enhanced queries)...")
            for qid, qtext in tqdm(enhanced_queries, desc="Dense Retrieval [enhanced]"):
                dense_run_enh[qid] = dense_search(cfg, faiss_index, embedder, doc_ids, qtext, cfg.retrieval_topk)
            logger.info(f"Dense retrieval (enhanced) completed, processed {len(dense_run_enh)} queries")

        # BM25 retrieval (optional, baseline query text)
        bm25_run: Dict[str, List[Tuple[str, float]]] = {}
        if bm25_searcher is not None:
            logger.info("Executing BM25 retrieval (baseline queries)...")
            for qid, qtext in tqdm(original_queries, desc="BM25 Retrieval [baseline]"):
                bm25_run[qid] = bm25_search(bm25_searcher, qtext, cfg.retrieval_topk)
            logger.info(f"BM25 retrieval completed, processed {len(bm25_run)} queries")

        # Rerank dense retrieval results (optional)
        rerank_run_base: Dict[str, List[Tuple[str, float]]] = {}
        rerank_run_enh: Dict[str, List[Tuple[str, float]]] = {}
        method = (cfg.rerank_method or "none").lower()
        if method != "none":
            logger.info(f"Creating reranker model (method={method})...")
            reranker, _ = build_reranker(cfg, method=method)
            logger.info(
                f"Executing reranking (baseline) (method={method}, candidates: {cfg.rerank_k}, final output: {cfg.final_k}, batch={cfg.rerank_batch})..."
            )
            for qid, qtext in tqdm(original_queries, desc=f"Reranking[{method}][baseline]"):
                base = dense_run_base[qid][:cfg.rerank_k]
                rerank_run_base[qid] = rerank(
                    reranker,
                    qtext,
                    base,
                    docid_to_text,
                    method=method,
                    batch_size=cfg.rerank_batch,
                )[:cfg.final_k]

            if enhanced_queries:
                logger.info(
                    f"Executing reranking (enhanced) (method={method}, candidates: {cfg.rerank_k}, final output: {cfg.final_k}, batch={cfg.rerank_batch})..."
                )
                for qid, qtext in tqdm(enhanced_queries, desc=f"Reranking[{method}][enhanced]"):
                    base = dense_run_enh[qid][:cfg.rerank_k]
                    rerank_run_enh[qid] = rerank(
                        reranker,
                        qtext,
                        base,
                        docid_to_text,
                        method=method,
                        batch_size=cfg.rerank_batch,
                    )[:cfg.final_k]

        # Prepare evaluation views to ensure fair comparison: use final_k for Dense
        dense_eval_base = {qid: hits[:cfg.final_k] for qid, hits in dense_run_base.items()}
        dense_eval_enh = None
        if enhanced_queries:
            dense_eval_enh = {qid: hits[:cfg.final_k] for qid, hits in dense_run_enh.items()}

        # Evaluate results
        results = {}
        logger.info("Evaluating retrieval results...")
        logger.info("\n" + "=" * 80)
        results["Dense (baseline)"] = evaluate_runs(dense_eval_base, qrels, eval_at=cfg.eval_at)
        if enhanced_queries:
            results["Dense (enhanced)"] = evaluate_runs(dense_eval_enh, qrels, eval_at=cfg.eval_at)
        if bm25_searcher is not None:
            results["BM25"] = evaluate_runs(bm25_run, qrels, eval_at=cfg.eval_at)
        if rerank_run_base:
            results[f"Rerank of Dense (baseline)"] = evaluate_runs(rerank_run_base, qrels, eval_at=cfg.eval_at)
        if rerank_run_enh:
            results[f"Rerank of Dense (enhanced)"] = evaluate_runs(rerank_run_enh, qrels, eval_at=cfg.eval_at)
        logger.info("=" * 80)

        results_df = pd.DataFrame(results)
        logger.info("\nResults:\n" + results_df.to_string())
        
        # Done
        elapsed_time = time.time() - start_time
        logger.info(f"\nExperiment completed, total time: {elapsed_time:.2f} seconds")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("\nUser interrupted the experiment")
        raise
    except Exception as e:
        logger.exception(f"An error occurred during the experiment: {e}")
        raise


def main() -> None:
    """CLI main entry point.
    
    Sets up the environment, loads configuration, and runs the experiment.
    """
    parser = argparse.ArgumentParser(description="Run CDS2016 retrieval experiments")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed log output (DEBUG level)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Only show warning and error logs (WARNING level)"
    )
    args = parser.parse_args()

    if args.verbose and args.quiet:
        print("Error: --verbose and --quiet cannot be used simultaneously", file=sys.stderr)
        sys.exit(1)
        
    try:
        # Configure logging system
        log_dir = Path("./logs") if args.verbose else None
        configure_logging(log_dir, verbose=args.verbose, quiet=args.quiet)
        
        # Load configuration
        logger.info("Loading experiment configuration...")
        cfg = Config.from_env()
        
        # Ensure necessary directories exist
        os.makedirs(cfg.faiss_dir, exist_ok=True)
        
        # Run experiment
        run_all(cfg)
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nProgram interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
