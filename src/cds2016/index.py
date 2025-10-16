"""
FAISS index management module.

Responsibilities:
- Build a flat inner-product FAISS index over document embeddings
- Optionally accelerate with GPU (falls back to CPU on failure or unavailability)
- Save/load index and associated IDs/vectors
- Provide search() that returns top-k scores and indices (bounded by ntotal)
"""
import json
import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import faiss
import numpy as np

from .config import Config


logger = logging.getLogger(__name__)


class FaissIndex:
    """Manages a FAISS vector index.
    
    Responsible for index creation, storage, loading, and search operations.
    
    Attributes:
        cfg: The config object
        vec_path: Path to store vectors
        ids_path: Path to store document IDs
        faiss_path: Path to store the FAISS index
        index: The FAISS index object
        doc_ids: List of document IDs
    """
    
    def __init__(self, cfg: Config) -> None:
        """Initializes the FAISS index manager.
        
        Args:
            cfg: The config object
        """
        self.cfg = cfg
        self.vec_path = Path(cfg.faiss_dir) / "docvecs.npy"
        self.ids_path = Path(cfg.faiss_dir) / "docids.json"
        self.faiss_path = Path(cfg.faiss_dir) / "faiss.index"
        
        # Ensure the directory exists
        Path(cfg.faiss_dir).mkdir(parents=True, exist_ok=True)
        
        self.index: Optional[faiss.Index] = None
        self.doc_ids: List[str] = []
        self._dimension: Optional[int] = None

    def exists(self) -> bool:
        """Check if the index already exists.
        
        Returns:
            bool: Whether the index files exist
        """
        return self.faiss_path.exists() and self.ids_path.exists()

    def load(self) -> None:
        """Load an existing FAISS index.
        
        Raises:
            FileNotFoundError: When index files do not exist
            Exception: When loading fails
        """
        if not self.exists():
            raise FileNotFoundError(f"FAISS index does not exist: {self.faiss_path}")
        
        try:
            logger.debug(f"Loading FAISS index: {self.faiss_path}")
            self.index = faiss.read_index(str(self.faiss_path))
            
            with open(self.ids_path, "r", encoding="utf-8") as f:
                self.doc_ids = json.load(f)
            
            # Log index information
            self._dimension = self.index.d
            logger.debug(f"Index loaded successfully: {self.index.ntotal} vectors, dimension {self._dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise

    def save(self, index: faiss.Index, doc_ids: List[str], doc_vectors: np.ndarray) -> None:
        """Save the FAISS index and related data.
        
        Args:
            index: The FAISS index object
            doc_ids: List of document IDs
            doc_vectors: Matrix of document vectors
            
        Raises:
            Exception: When saving fails
        """
        try:
            # Ensure the index is on the CPU before saving
            if hasattr(faiss, "GpuIndex") and isinstance(index, faiss.GpuIndex):
                logger.debug("Converting GPU index to CPU for saving")
                index_cpu = faiss.index_gpu_to_cpu(index)
            else:
                index_cpu = index
            
            # Save the FAISS index
            logger.debug(f"Saving FAISS index: {self.faiss_path}")
            faiss.write_index(index_cpu, str(self.faiss_path))
            
            # Save document IDs
            logger.debug(f"Saving document IDs: {self.ids_path}")
            with open(self.ids_path, "w", encoding="utf-8") as f:
                json.dump(doc_ids, f, ensure_ascii=False, indent=4)
            
            # Save vectors (optional, for backup or analysis)
            logger.debug(f"Saving vector data: {self.vec_path}")
            np.save(self.vec_path, doc_vectors)
            
            # Update internal state
            self.index = index
            self.doc_ids = doc_ids
            self._dimension = index.d
            
            logger.debug(f"Index saved successfully: {len(doc_ids)} documents")
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise

    def build(self, doc_vectors: np.ndarray, use_gpu: Optional[bool] = None) -> faiss.Index:
        """Build a new FAISS index.
        
        Args:
            doc_vectors: Matrix of document vectors (n_docs, dim)
            use_gpu: Whether to use GPU acceleration. If None, uses config.device setting
            
        Returns:
            faiss.Index: The created index
            
        Raises:
            ValueError: When the input vectors are invalid
        """
        if doc_vectors.size == 0:
            raise ValueError("Input vectors cannot be empty")
        
        if len(doc_vectors.shape) != 2:
            raise ValueError(f"Vector dimension must be 2, but got: {len(doc_vectors.shape)}")
        
        n_docs, dim = doc_vectors.shape
        logger.debug(f"Building FAISS index: {n_docs} documents, {dim}-dimensional vectors")
        
        # Use a flat index with inner product similarity
        index = faiss.IndexFlatIP(dim)
        
        # Determine GPU usage from config if not specified
        if use_gpu is None:
            use_gpu = self.cfg.device.startswith('cuda')
        
        # GPU acceleration (if available)
        if use_gpu and faiss.get_num_gpus() > 0:
            try:
                logger.debug("Using GPU acceleration for FAISS")
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("Successfully using GPU acceleration for FAISS index")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(
                        f"GPU out of memory error occurred: {e}. "
                        "Automatically falling back to CPU. "
                        "Consider using smaller batches or set DEVICE=cpu to avoid this."
                    )
                else:
                    logger.warning(f"GPU acceleration failed, falling back to CPU: {e}")
            except Exception as e:
                logger.warning(f"GPU acceleration failed, falling back to CPU: {e}")
        
        # Add vectors to the index
        logger.debug("Adding vectors to the index...")
        index.add(doc_vectors)
        
        # Update internal state
        self.index = index
        self._dimension = dim
        
        logger.debug(f"Index built successfully, total vectors: {index.ntotal}")
        return index

    def search(self, query_vector: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for most similar items in the index.
        
        The search uses inner product (equivalent to cosine if vectors are L2-normalized)
        and returns distances (similarity scores) and indices for up to `topk` items.
        
        Args:
            query_vector: The query vector with shape (1, dim)
            topk: Maximum number of results to return (capped by index size)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (distances, indices), each with shape (1, k)
            
        Raises:
            RuntimeError: If the index has not been initialized
            ValueError: If the query shape or topk is invalid
        """
        if self.index is None:
            raise RuntimeError("FAISS index has not been initialized")
        
        if query_vector.shape[0] != 1:
            raise ValueError(f"The first dimension of the query vector must be 1, but got: {query_vector.shape[0]}")
        
        if topk <= 0:
            raise ValueError(f"topk must be a positive number, but got: {topk}")
        
        # Perform the search
        topk = min(topk, self.index.ntotal)  # Ensure not to exceed the total number of vectors
        D, I = self.index.search(query_vector, topk)
        
        return D, I
    
    @property
    def dimension(self) -> Optional[int]:
        """Get the vector dimension of the index."""
        return self._dimension if self._dimension else (self.index.d if self.index else None)
    
    @property
    def total_vectors(self) -> int:
        """Get the total number of vectors in the index."""
        return self.index.ntotal if self.index else 0


