"""
Configuration management module.

Provides all configuration parameters for CDS2016 experiments, with support for
environment variable overrides and parameter validation.
"""
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import torch


logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration class for CDS2016 experiments.
    
    Manages all experiment parameters, including dataset, models, index paths, etc.
    Supports loading configurations from environment variables for easy experiment tuning.
    
    Attributes:
        dataset_id: IR Datasets dataset ID
        embed_model: Embedding model name (Hugging Face model ID)
        rerank_model: Reranker model name (Hugging Face model ID)
        embed_model_kwargs: Advanced model kwargs for embedding model (e.g., flash_attention_2)
        embed_tokenizer_kwargs: Advanced tokenizer kwargs for embedding model (e.g., padding_side)
        rerank_model_kwargs: Advanced model kwargs for reranking model
        faiss_dir: Directory to store the FAISS index
        retrieval_topk: Number of documents to retrieve in the first stage and feed into the reranker
        rerank_k: Number of candidate documents to actually process during reranking
        final_k: Number of documents to finally output after reranking
        eval_at: List of ranking positions to evaluate at
        max_docs: Maximum number of documents to process (for development/testing)
        use_body: Whether to use the document body content
        device: Computing device (cuda/cpu)
        embedding_batch: Embedding batch processing size
        lucene_dir: Lucene/BM25 index directory
        bm25_k1: BM25 k1 parameter
        bm25_b: BM25 b parameter
    """
    # Dataset configuration
    dataset_id: str = "pmc/v2/trec-cds-2016"
    
    # Model configuration
    embed_model: str = "Qwen/Qwen3-Embedding-8B"
    rerank_model: str = "BAAI/bge-reranker-v2-gemma"  # BAAI/bge-reranker-v2-gemma, ncbi/MedCPT-Cross-Encoder, Qwen/Qwen3-Reranker-8B
    # Reranking options
    rerank_method: str = "gemma"  # one of: "none", "gemma", "seq_cls", "qwen3", "minicpm_layerwise"
    rerank_batch: int = 16

    # Advanced model configuration (optional)
    embed_model_kwargs: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "attn_implementation": "flash_attention_2"
    })
    embed_tokenizer_kwargs: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "padding_side": "left"
    })
    rerank_model_kwargs: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "attn_implementation": "flash_attention_2"
    })
    
    # Index and output configuration
    faiss_dir: str = "./artifacts/cds2016_faiss"
    
    # Retrieval parameters
    retrieval_topk: int = 1000
    rerank_k: int = 1000
    final_k: int = 100
    eval_at: List[int] = field(default_factory=lambda: [10])
    
    # Data processing parameters
    max_docs: Optional[int] = None  # 1000 / None
    use_body: bool = False
    
    # Computing resource configuration
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    embedding_batch: int = 2  # for embedding

    # BM25/Lucene configuration (optional)
    lucene_dir: str = "./artifacts/lucene_cds2016"
    bm25_k1: float = 0.9
    bm25_b: float = 0.4

    # Query enhancement configuration (optional)
    enhance_query: bool = False
    enhance_endpoint: str = "http://localhost:11434/api/chat"
    # enhance_model: str = "gpt-oss:120b"
    enhance_model: str = "medgemma-27b-text-it:latest"


    def __post_init__(self) -> None:
        """Post-initialization validation and processing."""
        self._validate_config()
        self._setup_directories()
        
    def _validate_config(self) -> None:
        """Validate the reasonableness of configuration parameters."""
        # Validate numerical parameters
        if self.retrieval_topk <= 0:
            raise ValueError(f"retrieval_topk must be positive, but got: {self.retrieval_topk}")
        
        if self.rerank_k <= 0 or self.rerank_k > self.retrieval_topk:
            raise ValueError(
                f"rerank_k must be positive and not greater than retrieval_topk, "
                f"but got: rerank_k={self.rerank_k}, retrieval_topk={self.retrieval_topk}"
            )
        
        if self.final_k <= 0 or self.final_k > self.rerank_k:
            raise ValueError(
                f"final_k must be positive and not greater than rerank_k, "
                f"but got: final_k={self.final_k}, rerank_k={self.rerank_k}"
            )
        
        if self.embedding_batch <= 0:
            raise ValueError(f"embedding_batch must be positive, but got: {self.embedding_batch}")
        
        if self.max_docs is not None and self.max_docs <= 0:
            raise ValueError(f"max_docs must be positive, but got: {self.max_docs}")
        
        # Validate BM25 parameters
        if not (0.0 <= self.bm25_k1 <= 3.0):
            logger.warning(f"BM25 k1 parameter is usually between 0-3, but current value is: {self.bm25_k1}")
        
        if not (0.0 <= self.bm25_b <= 1.0):
            logger.warning(f"BM25 b parameter should be between 0-1, but current value is: {self.bm25_b}")
        
        # Validate evaluation positions
        if not self.eval_at:
            raise ValueError("eval_at cannot be an empty list")
        
        for k in self.eval_at:
            if k <= 0:
                raise ValueError(f"Values in eval_at must be positive, but got: {k}")
        
        # Validate that final_k is sufficient for evaluation
        max_eval_at = max(self.eval_at)
        if self.final_k < max_eval_at:
            raise ValueError(
                f"final_k must be at least as large as max(eval_at) to enable proper evaluation, "
                f"but got: final_k={self.final_k}, max(eval_at)={max_eval_at}"
            )

        # Normalize/validate device keyword
        try:
            device_norm = str(self.device).strip().lower()
        except Exception:
            device_norm = "cpu"

        if device_norm in ("cuda", "gpu"):
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                logger.warning("CUDA requested but not available; falling back to CPU")
                self.device = "cpu"
        
        elif device_norm == "cpu":
            self.device = "cpu"
        
        else:
            logger.warning(f"Unknown device '{self.device}', falling back to CPU")
            self.device = "cpu"
    
    def _setup_directories(self) -> None:
        """Create necessary directories."""
        Path(self.faiss_dir).mkdir(parents=True, exist_ok=True)
        # Path(self.lucene_dir).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables.
        
        Returns:
            Config: A config instance with environment variables loaded
            
        Raises:
            ValueError: When an environment variable value cannot be parsed
        """
        def get_env_int(key: str, default: str, allow_none: bool = False) -> Optional[int]:
            """Safely get an integer value from an environment variable.
            
            Args:
                key: Environment variable name
                default: Default value as string
                allow_none: If True, allows None/null/empty values to return None
                
            Returns:
                int or None: The parsed integer value or None if allow_none=True and value indicates None
            """
            value = os.getenv(key, default)
            if allow_none and (value is None or value.lower() in ('none', 'null', '')):
                return None
            try:
                return int(value)
            except ValueError as e:
                error_msg = f"Environment variable {key} must be an integer"
                if allow_none:
                    error_msg += " or None"
                error_msg += f", but got: {value}"
                raise ValueError(error_msg) from e
        
        def get_env_float(key: str, default: str) -> float:
            """Safely get a float value from an environment variable."""
            try:
                return float(os.getenv(key, default))
            except ValueError as e:
                raise ValueError(f"Environment variable {key} must be a number, but got: {os.getenv(key)}") from e
        
        def get_env_bool(key: str, default: str = "0") -> bool:
            """Safely get a boolean value from an environment variable."""
            value = os.getenv(key, default).lower()
            return value in ("1", "true", "yes", "on")
        
        def get_env_int_list(key: str, default: str) -> List[int]:
            """Safely get a list of integers from an environment variable."""
            value = os.getenv(key, default)
            try:
                return [int(x.strip()) for x in value.split(",") if x.strip()]
            except ValueError as e:
                raise ValueError(f"Environment variable {key} must be a comma-separated list of integers, but got: {value}") from e

        
        # Log settings loaded from environment variables
        logger.debug("Loading configuration from environment variables...")
        
        # Create default instance to get default values
        default_config = cls()
        
        config = cls(
            dataset_id=os.getenv("DATASET_ID", default_config.dataset_id),
            embed_model=os.getenv("EMBED_MODEL", default_config.embed_model),
            rerank_model=os.getenv("RERANK_MODEL", default_config.rerank_model),
            embed_model_kwargs=default_config.embed_model_kwargs,
            embed_tokenizer_kwargs=default_config.embed_tokenizer_kwargs,
            rerank_model_kwargs=default_config.rerank_model_kwargs,
            faiss_dir=os.getenv("FAISS_DIR", default_config.faiss_dir),
            retrieval_topk=get_env_int("RETRIEVAL_TOPK", str(default_config.retrieval_topk)),
            rerank_k=get_env_int("RERANK_K", str(default_config.rerank_k)),
            final_k=get_env_int("FINAL_K", str(default_config.final_k)),
            eval_at=get_env_int_list("EVAL_AT", ",".join(map(str, default_config.eval_at))),
            max_docs=get_env_int("MAX_DOCS", str(default_config.max_docs) if default_config.max_docs is not None else "None", allow_none=True),
            use_body=default_config.use_body,
            device=os.getenv("DEVICE", default_config.device),
            embedding_batch=get_env_int("EMBEDDING_BATCH", str(default_config.embedding_batch)),
            lucene_dir=os.getenv("LUCENE_DIR", default_config.lucene_dir),
            bm25_k1=get_env_float("BM25_K1", str(default_config.bm25_k1)),
            bm25_b=get_env_float("BM25_B", str(default_config.bm25_b)),
            enhance_query=get_env_bool("ENHANCE_QUERY", str(default_config.enhance_query).lower()),
            enhance_endpoint=os.getenv("ENHANCE_ENDPOINT", default_config.enhance_endpoint),
            enhance_model=os.getenv("ENHANCE_MODEL", default_config.enhance_model),
            rerank_method=os.getenv("RERANK_METHOD", default_config.rerank_method),
            rerank_batch=get_env_int("RERANK_BATCH", str(default_config.rerank_batch)),
        )
        
        logger.debug("Configuration loaded from environment variables")
        return config
    
    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary format."""
        return {
            "dataset_id": self.dataset_id,
            "embed_model": self.embed_model,
            "rerank_model": self.rerank_model,
            "embed_model_kwargs": self.embed_model_kwargs,
            "embed_tokenizer_kwargs": self.embed_tokenizer_kwargs,
            "rerank_model_kwargs": self.rerank_model_kwargs,
            "faiss_dir": self.faiss_dir,
            "retrieval_topk": self.retrieval_topk,
            "rerank_k": self.rerank_k,
            "final_k": self.final_k,
            "eval_at": self.eval_at,
            "max_docs": self.max_docs,
            "use_body": self.use_body,
            "device": self.device,
            "embedding_batch": self.embedding_batch,
            "lucene_dir": self.lucene_dir,
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
            "enhance_query": self.enhance_query,
            "enhance_endpoint": self.enhance_endpoint,
            "enhance_model": self.enhance_model,
            "rerank_method": self.rerank_method,
            "rerank_batch": self.rerank_batch,
        }


