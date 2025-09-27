"""
Embedding vector processing module.

Provides functionality for encoding text into dense vectors, with support for different
handling of documents and queries.
"""
import logging
from typing import List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from .config import Config
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)


def _should_fallback_from_flash_attn(error: Exception) -> bool:
    """Return True if the exception implies Flash Attention 2 is unsupported.

    Looks for known indicator strings in the error message.
    """
    try:
        message = str(error).lower()
    except Exception:
        return False
    keywords = (
        "does not support flash attention 2.0",
        "flash attention 2",
        "flash_attention_2",
        "flash_attn_2_can_dispatch",
    )
    return any(k in message for k in keywords)

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Extract embeddings from the last token position.
    
    Handles both left and right padding by finding the actual last token position.
    
    Args:
        last_hidden_states: Hidden states from the model
        attention_mask: Attention mask tensor
        
    Returns:
        Tensor: Pooled embeddings from last token positions
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format query with task instruction for better embedding performance.
    
    Args:
        task_description: Description of the task
        query: The actual query text
        
    Returns:
        str: Formatted instruction + query text
    """
    return f'Instruct: {task_description}\nQuery:{query}'


def prep_text_for_embed(text: str) -> str:
    """Prepare document text for embedding.
    
    Adds appropriate prefix markers according to model requirements.
    
    Args:
        text: Original document text
        
    Returns:
        str: Text with prefix added
    """
    # Remove extra whitespace and add document prefix
    cleaned_text = " ".join(text.split())
    return f"{cleaned_text}".strip()


def build_embedder(cfg: Config) -> Tuple[AutoModel, AutoTokenizer]:
    """Build and configure the embedding model and tokenizer.
    
    Args:
        cfg: Config object containing model name, device settings, and advanced kwargs
        
    Returns:
        Tuple[AutoModel, AutoTokenizer]: The configured embedding model and tokenizer
        
    Raises:
        Exception: When model loading fails
    """
    logger.debug(f"Loading embedding model: {cfg.embed_model}")
    
    try:
        # Prepare model/tokenizer kwargs with compatibility checks
        model_kwargs = cfg.embed_model_kwargs.copy() if cfg.embed_model_kwargs else {}
        tokenizer_kwargs = cfg.embed_tokenizer_kwargs.copy() if cfg.embed_tokenizer_kwargs else {}

        # Set default kwargs for optimal performance
        model_kwargs.setdefault("attn_implementation", "flash_attention_2")
        # device_map policy: only enable auto-sharding when using CUDA; force CPU otherwise
        device_norm = str(getattr(cfg, 'device', 'cpu')).lower()
        if device_norm in ('cuda', 'gpu') and torch.cuda.is_available():
            model_kwargs.setdefault("device_map", "auto")
        else:
            # Ensure device_map is not set on CPU to avoid accelerate/offload behavior
            model_kwargs.pop("device_map", None)

        # Handle flash_attention_2 compatibility
        if model_kwargs.get("attn_implementation") == "flash_attention_2":
            try:
                import flash_attn
                logger.debug("Flash Attention 2 is available")
            except ImportError:
                logger.warning("Flash Attention 2 not available, removing attn_implementation from kwargs")
                model_kwargs.pop("attn_implementation", None)

        # Handle device_map compatibility
        uses_device_map = "device_map" in model_kwargs
        if uses_device_map:
            try:
                import accelerate
                logger.debug("Accelerate is available for device_map")
            except ImportError:
                logger.warning("Accelerate not available, removing device_map from kwargs")
                model_kwargs.pop("device_map", None)
                uses_device_map = False

        # Build the model and tokenizer with dtype fallback (bf16 -> fp16 -> default)
        dtype_candidates = [torch.bfloat16, torch.float16, None]
        model: Optional[AutoModel] = None
        tokenizer: Optional[AutoTokenizer] = None
        last_error: Optional[Exception] = None

        # Load tokenizer first
        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg.embed_model, **tokenizer_kwargs)
            logger.debug(f"Tokenizer loaded with kwargs={tokenizer_kwargs}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

        # Load model with dtype fallback
        for dtype_candidate in dtype_candidates:
            try_kwargs = model_kwargs.copy()
            # Always specify dtype when possible (unless None for default)
            if dtype_candidate is not None:
                try_kwargs["dtype"] = dtype_candidate
            else:
                try_kwargs.pop("dtype", None)

            try:
                model = AutoModel.from_pretrained(cfg.embed_model, **try_kwargs)
                
                # Move to device if not using device_map
                if not uses_device_map and hasattr(cfg, 'device'):
                    model = model.to(cfg.device)
                
                logger.debug(
                    f"Embedding model loaded with kwargs={try_kwargs} uses_device_map={uses_device_map}"
                )
                break
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to load model with dtype={dtype_candidate}: {e}")
                # Fallback: if FA2 seems to be the cause, drop attn_implementation and retry once for this dtype
                if try_kwargs.get("attn_implementation") == "flash_attention_2" and _should_fallback_from_flash_attn(e):
                    fa2_fallback_kwargs = try_kwargs.copy()
                    fa2_fallback_kwargs.pop("attn_implementation", None)
                    try:
                        model = AutoModel.from_pretrained(cfg.embed_model, **fa2_fallback_kwargs)
                        if not uses_device_map and hasattr(cfg, 'device'):
                            model = model.to(cfg.device)
                        logger.debug(
                            f"Embedding model loaded after FA2 fallback with kwargs={fa2_fallback_kwargs}"
                        )
                        break
                    except Exception as e2:
                        last_error = e2
                        logger.warning(
                            f"Retry without FA2 failed for dtype={dtype_candidate}: {e2}"
                        )

        # If still not loaded and device_map was requested, try again without device_map
        if model is None and uses_device_map:
            logger.warning("Retrying model load without device_map due to previous failures")
            safe_kwargs = model_kwargs.copy()
            safe_kwargs.pop("device_map", None)
            for dtype_candidate in dtype_candidates:
                try_kwargs = safe_kwargs.copy()
                if dtype_candidate is not None:
                    try_kwargs["dtype"] = dtype_candidate
                else:
                    try_kwargs.pop("dtype", None)
                try:
                    model = AutoModel.from_pretrained(cfg.embed_model, **try_kwargs)
                    
                    # Move to device
                    if hasattr(cfg, 'device'):
                        model = model.to(cfg.device)
                    
                    logger.debug(
                        f"Embedding model loaded (no device_map) with kwargs={try_kwargs}"
                    )
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(f"Failed to load model without device_map, dtype={dtype_candidate}: {e}")
                    # Fallback: if FA2 seems to be the cause, drop attn_implementation and retry once for this dtype
                    if try_kwargs.get("attn_implementation") == "flash_attention_2" and _should_fallback_from_flash_attn(e):
                        fa2_fallback_kwargs = try_kwargs.copy()
                        fa2_fallback_kwargs.pop("attn_implementation", None)
                        try:
                            model = AutoModel.from_pretrained(cfg.embed_model, **fa2_fallback_kwargs)
                            if hasattr(cfg, 'device'):
                                model = model.to(cfg.device)
                            logger.debug(
                                f"Embedding model loaded (no device_map) after FA2 fallback with kwargs={fa2_fallback_kwargs}"
                            )
                            break
                        except Exception as e2:
                            last_error = e2
                            logger.warning(
                                f"Retry (no device_map) without FA2 failed for dtype={dtype_candidate}: {e2}"
                            )

        if model is None:
            # Final basic fallback
            logger.warning(
                f"All attempts failed when loading model with kwargs. Last error: {last_error}. "
                "Falling back to basic model loading"
            )
            try:
                model = AutoModel.from_pretrained(cfg.embed_model)
                if hasattr(cfg, 'device'):
                    model = model.to(cfg.device)
            except Exception as e:
                logger.error(f"Basic model loading failed: {e}; previous error: {last_error}")
                raise

        # Set to evaluation mode
        model.eval()
        
        # Get model dimension info
        embedding_dim = model.config.hidden_size
        logger.debug(f"Embedding vector dimension: {embedding_dim}")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise


def _encode_inputs(
    model_and_tokenizer: Tuple[AutoModel, AutoTokenizer],
    inputs: List[str],
    batch_size: int,
    max_length: int,
    is_query: bool,
    task_description: Optional[str] = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Unified encoder for documents and queries.

    Applies optional instruction formatting for queries, tokenizes, runs the
    model forward pass, pools last token embeddings, normalizes, and returns
    float32 numpy arrays.

    Args:
        model_and_tokenizer: Tuple of (model, tokenizer)
        inputs: List of raw input texts
        batch_size: Batch size for processing
        max_length: Maximum sequence length for tokenization
        is_query: Whether inputs are queries (will add instruction)
        task_description: Instruction text when is_query is True
        show_progress: Whether to display a tqdm progress bar

    Returns:
        np.ndarray: Matrix of embeddings with shape (n_inputs, embedding_dim)
    """
    if not inputs:
        raise ValueError("Input text list cannot be empty")

    model, tokenizer = model_and_tokenizer

    # Prepare inputs
    if is_query:
        effective_task = (
            task_description
            if task_description
            else "Given a query, retrieve relevant passages that answer the query"
        )
        prepared_texts = [get_detailed_instruct(effective_task, text) for text in inputs]
    else:
        prepared_texts = [prep_text_for_embed(text) for text in inputs]

    vectors: List[np.ndarray] = []

    # Process in batches
    total_items = len(prepared_texts)
    progress = tqdm(total=total_items, desc="Encoding", unit="text", disable=not show_progress)
    for i in range(0, len(prepared_texts), batch_size):
        batch_texts = prepared_texts[i:i + batch_size]

        # Tokenize
        batch_dict = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Determine whether model is sharded/offloaded via hf_device_map.
        uses_sharded = getattr(model, "hf_device_map", None) is not None
        if not uses_sharded:
            target_device = getattr(model, "device", "cpu")
            batch_dict = batch_dict.to(target_device)

        with torch.inference_mode():
            outputs = model(**batch_dict)
            attn_mask = batch_dict['attention_mask'].to(outputs.last_hidden_state.device)
            embeddings = last_token_pool(outputs.last_hidden_state, attn_mask)
            embeddings = F.normalize(embeddings, p=2, dim=1)  # l2 norm
            vec = embeddings.cpu().to(torch.float32).numpy()

        vectors.append(vec)

        progress.update(len(batch_texts))

        if ((i + batch_size) % (batch_size * 10) == 0):
            logging.getLogger(__name__).debug(
                f"Encoded {min(i + batch_size, len(prepared_texts))}/{len(prepared_texts)} inputs"
            )

        # Release temporaries and reduce fragmentation on CUDA
        # if torch.cuda.is_available():
        #     del outputs, embeddings
        #     torch.cuda.empty_cache()

    progress.close()

    if vectors:
        return np.vstack(vectors)
    else:
        embedding_dim = model.config.hidden_size
        return np.zeros((0, embedding_dim), dtype="float32")


def encode_texts(
    model_and_tokenizer: Tuple[AutoModel, AutoTokenizer], 
    texts: List[str], 
    batch_size: int,
    max_length: int = 8192
) -> np.ndarray:
    """Batch encode multiple texts into embedding vectors.
    
    Args:
        model_and_tokenizer: Tuple of (model, tokenizer)
        texts: List of texts to encode
        batch_size: Batch size
        max_length: Maximum sequence length for tokenization
        
    Returns:
        np.ndarray: A matrix of vectors with shape (n_texts, embedding_dim)
        
    Raises:
        ValueError: When the input list is empty
    """
    if not texts:
        raise ValueError("Input text list cannot be empty")
    
    logger.debug(f"Encoding {len(texts)} texts, batch size: {batch_size}")
    return _encode_inputs(
        model_and_tokenizer=model_and_tokenizer,
        inputs=texts,
        batch_size=batch_size,
        max_length=max_length,
        is_query=False,
        task_description=None,
        show_progress=True,
    )


def encode_query(
    model_and_tokenizer: Tuple[AutoModel, AutoTokenizer], 
    query: str, 
    task_description: str = "Given a query, retrieve relevant passages that answer the query",
    max_length: int = 8192
) -> np.ndarray:
    """Encode a single query into an embedding vector.
    
    Args:
        model_and_tokenizer: Tuple of (model, tokenizer)
        query: The query text
        task_description: Task description for instruction formatting
        max_length: Maximum sequence length for tokenization
        
    Returns:
        np.ndarray: A query vector with shape (1, embedding_dim)
        
    Raises:
        ValueError: When the query is an empty string
    """
    if not query or not query.strip():
        raise ValueError("Query text cannot be empty")
    
    result = _encode_inputs(
        model_and_tokenizer=model_and_tokenizer,
        inputs=[query],
        batch_size=1,
        max_length=max_length,
        is_query=True,
        task_description=task_description,
        show_progress=False,
    )
    return result


