"""
Reranking module.

Provides functionality to rerank retrieval results using:
- BAAI/bge-reranker-v2-gemma (LLM-style causal LM head)
- BAAI/bge-reranker-v2-minicpm-layerwise (Layerwise causal LM)
- Qwen3 reranker (Causal LM with yes/no scoring)
- Cross-Encoder sequence classification models (e.g., ncbi/MedCPT-Cross-Encoder)
"""
import logging
import torch
import numpy as np
from contextlib import nullcontext
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizer
)

from .config import Config


logger = logging.getLogger(__name__)


# =============================
# Helper Functions
# =============================
def _should_fallback_from_flash_attn(error: Exception) -> bool:
    """Detect errors indicating Flash Attention 2 is not supported."""
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


def _is_cuda_target(dev: Any) -> bool:
    """Check if a device is CUDA-based."""
    try:
        if isinstance(dev, int):
            return True
        if isinstance(dev, torch.device):
            return dev.type == "cuda"
        if isinstance(dev, str):
            return dev.startswith("cuda")
        if isinstance(dev, (list, tuple)) and dev:
            return all(_is_cuda_target(d) for d in dev)
    except Exception:
        return False
    return False


def _get_amp_context(model: PreTrainedModel, use_cuda: bool = True):
    """Get appropriate AMP context based on model device configuration."""
    use_amp = use_cuda and torch.cuda.is_available()
    
    # Check device_map compatibility
    if getattr(model, "hf_device_map", None) is not None:
        device_map = getattr(model, "hf_device_map", {})
        try:
            all_cuda = all(_is_cuda_target(dev) for dev in device_map.values())
        except Exception:
            all_cuda = False

        if not all_cuda:
            use_amp = False
    
    param_dtype = next(model.parameters()).dtype  # e.g. torch.float16 or torch.bfloat16
    amp_dtype = torch.bfloat16 if param_dtype == torch.bfloat16 else torch.float16
    
    return (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype) 
        if use_amp 
        else nullcontext()
    )


# =============================
# Base Model Loading
# =============================
def _load_model_with_fallbacks(
    model_name: str,
    model_class: type,
    model_kwargs: Dict[str, Any],
    device: str,
    model_type: str = "model"
) -> PreTrainedModel:
    """Generic model loader with dtype and configuration fallbacks.
    
    Args:
        model_name: Model identifier
        model_class: Class to use for loading (e.g., AutoModelForCausalLM)
        model_kwargs: Additional kwargs for model loading
        device: Target device
        model_type: Description for logging (e.g., "reranker", "Qwen3")
    
    Returns:
        Loaded model
    """
    # Prepare model kwargs
    kwargs = model_kwargs.copy()
    
    # Set defaults
    kwargs.setdefault("attn_implementation", "flash_attention_2")
    # device_map policy: only enable auto-sharding when using CUDA; force CPU otherwise
    device_norm = str(device).lower() if device is not None else "cpu"
    if device_norm in ("cuda", "gpu") and torch.cuda.is_available():
        kwargs.setdefault("device_map", "auto")
    else:
        kwargs.pop("device_map", None)
    
    # Check Flash Attention availability
    if kwargs.get("attn_implementation") == "flash_attention_2":
        try:
            import flash_attn  # type: ignore
            logger.debug(f"Flash Attention 2 is available for {model_type}")
        except ImportError:
            logger.warning(f"Flash Attention 2 not available for {model_type}, removing from kwargs")
            kwargs.pop("attn_implementation", None)
    
    # Check accelerate availability for device_map
    uses_device_map = "device_map" in kwargs
    if uses_device_map:
        try:
            import accelerate  # type: ignore
            logger.debug(f"Accelerate is available for device_map ({model_type})")
        except ImportError:
            logger.warning(f"Accelerate not available, removing device_map from {model_type} kwargs")
            kwargs.pop("device_map", None)
            uses_device_map = False
    
    # Try loading with different dtype configurations
    dtype_candidates = [torch.bfloat16, torch.float16, None]
    model = None
    last_error: Optional[Exception] = None
    
    for dtype_candidate in dtype_candidates:
        try_kwargs = kwargs.copy()
        if dtype_candidate is not None:
            try_kwargs["dtype"] = dtype_candidate
        else:
            try_kwargs.pop("dtype", None)
        
        # First attempt
        try:
            model = model_class.from_pretrained(
                model_name,
                trust_remote_code=True,
                **try_kwargs
            )
            logger.debug(f"{model_type} loaded with kwargs={try_kwargs}")
            break
        except Exception as e:
            last_error = e
            logger.warning(f"Failed to load {model_type} with dtype={dtype_candidate}: {e}")
            
            # Flash Attention fallback
            if (try_kwargs.get("attn_implementation") == "flash_attention_2" and 
                _should_fallback_from_flash_attn(e)):
                fa2_fallback_kwargs = try_kwargs.copy()
                fa2_fallback_kwargs.pop("attn_implementation", None)
                try:
                    model = model_class.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        **fa2_fallback_kwargs
                    )
                    logger.debug(f"{model_type} loaded after FA2 fallback")
                    break
                except Exception as e2:
                    last_error = e2
                    logger.warning(f"FA2 fallback failed for {model_type}: {e2}")
    
    # Retry without device_map if still failing
    if model is None and uses_device_map:
        logger.warning(f"Retrying {model_type} load without device_map")
        safe_kwargs = kwargs.copy()
        safe_kwargs.pop("device_map", None)
        
        for dtype_candidate in dtype_candidates:
            try_kwargs = safe_kwargs.copy()
            if dtype_candidate is not None:
                try_kwargs["dtype"] = dtype_candidate
            else:
                try_kwargs.pop("dtype", None)
            
            try:
                model = model_class.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    **try_kwargs
                )
                logger.debug(f"{model_type} loaded without device_map")
                break
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to load {model_type} without device_map: {e}")
    
    if model is None:
        logger.error(f"All attempts to load {model_type} failed: {last_error}")
        raise last_error if last_error else RuntimeError(f"Failed to load {model_type}")
    
    # Move to device if not using device_map
    has_device_map = getattr(model, "hf_device_map", None) is not None
    if not has_device_map and hasattr(model, "to"):
        try:
            model.to(device)
            logger.debug(f"{model_type} moved to device: {device}")
        except Exception as e:
            logger.warning(f"Failed to move {model_type} to {device}: {e}")
    
    model.eval()
    return model


# =============================
# Base Reranking Class
# =============================
class BaseReranker(ABC):
    """Abstract base class for rerankers."""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    @abstractmethod
    def compute_scores(
        self, 
        pairs: List[Tuple[str, str]], 
        batch_size: int,
        max_length: int,
        **kwargs
    ) -> np.ndarray:
        """Compute relevance scores for query-document pairs."""
        pass
    
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        docid_to_text: Dict[str, str],
        batch_size: int = 32,
        max_length: int = 1024,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """Rerank candidates based on relevance scores."""
        if not candidates:
            logger.debug(f"Candidate list is empty ({self.__class__.__name__})")
            return []
        
        # Build valid pairs
        pairs = []
        valid_indices = []
        for idx, (docid, _) in enumerate(candidates):
            text = docid_to_text.get(docid, "")
            if text and text.strip():
                pairs.append((query, text))
                valid_indices.append(idx)
        
        if not pairs:
            logger.warning(f"No valid query-document pairs to rerank ({self.__class__.__name__})")
            return candidates
        
        # Compute scores
        scores = self.compute_scores(pairs, batch_size, max_length, **kwargs)
        
        # Sort by scores
        order = np.argsort(-scores)
        
        # Build reranked results
        reranked: List[Tuple[str, float]] = []
        for rank in order:
            original_idx = valid_indices[rank]
            doc_id = candidates[original_idx][0]
            reranked.append((doc_id, float(scores[rank])))
        
        # Append invalid candidates at the end
        valid_indices_set = set(valid_indices)
        for idx, (docid, score) in enumerate(candidates):
            if idx not in valid_indices_set:
                reranked.append((docid, score))
        
        logger.debug(f"{self.__class__.__name__} reranking completed, returned {len(reranked)} results")
        return reranked


# =============================
# Gemma Reranker
# =============================
def _gemma_get_inputs(pairs, tokenizer, prompt=None, max_length=1024):
    """Create inputs for BGE-Gemma reranking."""
    if prompt is None:
        prompt = "Given a query A (Electronic Health Record summary) and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
    sep = "\n"
    prompt_inputs = tokenizer(prompt, return_tensors=None, add_special_tokens=False)['input_ids']
    sep_inputs = tokenizer(sep, return_tensors=None, add_special_tokens=False)['input_ids']
    
    inputs = []
    for query, passage in pairs:
        query_inputs = tokenizer(
            f'A: {query}',
            return_tensors=None,
            add_special_tokens=False,
            max_length=max_length * 3 // 4,
            truncation=True
        )
        passage_inputs = tokenizer(
            f'B: {passage}',
            return_tensors=None,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True
        )
        item = tokenizer.prepare_for_model(
            [tokenizer.bos_token_id] + query_inputs['input_ids'],
            sep_inputs + passage_inputs['input_ids'],
            truncation='only_second',
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False
        )
        item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
        item['attention_mask'] = [1] * len(item['input_ids'])
        inputs.append(item)
    
    return tokenizer.pad(
        inputs,
        padding="max_length",
        max_length=max_length + len(sep_inputs) + len(prompt_inputs),
        pad_to_multiple_of=8,
        return_tensors='pt',
    )


class GemmaReranker(BaseReranker):
    """BGE-Gemma reranker using causal LM head."""
    
    def compute_scores(self, pairs, batch_size, max_length, **kwargs):
        yes_id = self.tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
        scores_acc = []
        
        self.model.eval()
        amp_ctx = _get_amp_context(self.model)
        model_device = self.model.get_input_embeddings().weight.device
        
        with torch.inference_mode(), amp_ctx:
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i : i + batch_size]
                inputs = _gemma_get_inputs(batch_pairs, self.tokenizer, max_length=max_length).to(model_device)
                logits = self.model(**inputs, return_dict=True).logits
                batch_scores = logits[:, -1, yes_id].detach().to("cpu").float()
                scores_acc.append(batch_scores)
        
        return torch.cat(scores_acc, dim=0).numpy()


# =============================
# Layerwise Reranker
# =============================
class LayerwiseReranker(BaseReranker):
    """Layerwise BGE reranker using multiple layer outputs."""
    
    def compute_scores(self, pairs, batch_size, max_length, cutoff_layers=None, **kwargs):
        if cutoff_layers is None:
            cutoff_layers = [32]  # Default layer
            
        scores_acc = []
        
        self.model.eval()
        amp_ctx = _get_amp_context(self.model)
        model_device = self.model.get_input_embeddings().weight.device
        
        with torch.inference_mode(), amp_ctx:
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i : i + batch_size]
                inputs = _gemma_get_inputs(batch_pairs, self.tokenizer, max_length=max_length).to(model_device)
                all_scores = self.model(**inputs, return_dict=True, cutoff_layers=cutoff_layers)
                # Extract scores from the specified layer
                layer_scores = all_scores[0][-1][:, -1].view(-1).detach().to("cpu").float()
                scores_acc.append(layer_scores)
        
        return torch.cat(scores_acc, dim=0).numpy()


# =============================
# Qwen3 Reranker
# =============================
def _qwen3_format_instruction(instruction: str, query: str, doc: str) -> str:
    """Format instruction string for Qwen3 reranking."""
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def _qwen3_get_inputs(
    pairs: List[Tuple[str, str]],
    tokenizer,
    instruction: Optional[str] = None,
    max_length: int = 8192,
):
    """Create inputs for Qwen3 reranking."""
    task = instruction or "Given a query (Electronic Health Record summary), retrieve relevant passages that answer the query."
    
    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    
    allowed = max(8, max_length - len(prefix_tokens) - len(suffix_tokens))
    
    formatted_texts = [_qwen3_format_instruction(task, q, d) for (q, d) in pairs]
    inputs = tokenizer(
        formatted_texts,
        return_tensors=None,
        add_special_tokens=False,
        padding=False,
        truncation="longest_first",
        max_length=allowed,
    )
    
    for i, ele in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
    
    return tokenizer.pad(
        inputs,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )


class Qwen3Reranker(BaseReranker):
    """Qwen3 reranker using yes/no next-token scoring."""
    
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        # Ensure left padding for last token scoring
        if getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"
    
    def compute_scores(self, pairs, batch_size, max_length, **kwargs):
        # Get token IDs for yes/no
        try:
            token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
            token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        except Exception:
            token_true_id = self.tokenizer("yes", add_special_tokens=False)["input_ids"][0]
            token_false_id = self.tokenizer("no", add_special_tokens=False)["input_ids"][0]
        
        scores_acc = []
        
        self.model.eval()
        amp_ctx = _get_amp_context(self.model)
        model_device = self.model.get_input_embeddings().weight.device
        
        with torch.inference_mode(), amp_ctx:
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i : i + batch_size]
                inputs = _qwen3_get_inputs(batch_pairs, self.tokenizer, max_length=max_length)
                inputs = inputs.to(model_device)
                logits = self.model(**inputs, return_dict=True).logits  # [B, T, V]
                
                # Extract yes/no probabilities
                last_logits = logits[:, -1, :]
                true_vec = last_logits[:, token_true_id]
                false_vec = last_logits[:, token_false_id]
                stacked = torch.stack([false_vec, true_vec], dim=1)
                log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
                probs_true = log_probs[:, 1].exp().detach().to("cpu").float()
                scores_acc.append(probs_true)
        
        return torch.cat(scores_acc, dim=0).numpy()


# =============================
# Sequence Classification Reranker
# =============================
class SequenceClassificationReranker(BaseReranker):
    """Cross-encoder style sequence classification reranker."""
    
    def compute_scores(self, pairs, batch_size, max_length, **kwargs):
        scores_acc = []
        
        self.model.eval()
        amp_ctx = _get_amp_context(self.model)
        model_device = self.model.get_input_embeddings().weight.device
        
        with torch.inference_mode(), amp_ctx:
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i : i + batch_size]
                encoded = self.tokenizer(
                    batch_pairs,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=max_length,
                )
                encoded = {k: v.to(model_device) for k, v in encoded.items()}
                logits = self.model(**encoded).logits
                
                # Handle different logit shapes
                if logits.dim() == 1:
                    batch_scores = logits.detach().to("cpu").float()
                elif logits.dim() == 2:
                    if logits.size(-1) == 1:
                        # Regression output
                        batch_scores = logits.squeeze(-1).detach().to("cpu").float()
                    elif logits.size(-1) == 2:
                        # Binary classification - take positive class
                        batch_scores = logits[:, 1].detach().to("cpu").float()
                    else:
                        # Multi-class - take last class
                        batch_scores = logits[:, -1].detach().to("cpu").float()
                else:
                    raise ValueError(f"Unexpected logits shape: {logits.shape}")
                
                scores_acc.append(batch_scores)
        
        return torch.cat(scores_acc, dim=0).numpy()


# =============================
# Factory Functions
# =============================
def build_reranker(cfg: Config, method: Optional[str] = None) -> Tuple[Optional[BaseReranker], Optional[str]]:
    """Build reranker based on method.
    
    Args:
        cfg: Configuration object
        method: Reranking method (defaults to cfg.rerank_method)
    
    Returns:
        (reranker_instance, method_name) or (None, "none")
    """
    chosen = (method or getattr(cfg, "rerank_method", "none") or "none").lower()
    
    if chosen == "none":
        return None, "none"
    
    # Load tokenizer (common for all methods)
    tokenizer = AutoTokenizer.from_pretrained(cfg.rerank_model, trust_remote_code=True)
    
    # Build model and reranker based on method
    if chosen in ("gemma", "bge_gemma"):
        model = _load_model_with_fallbacks(
            cfg.rerank_model,
            AutoModelForCausalLM,
            cfg.rerank_model_kwargs or {},
            cfg.device,
            "Gemma reranker"
        )
        return GemmaReranker(model, tokenizer), chosen
    
    elif chosen in ("layerwise", "minicpm_layerwise", "bge_layerwise"):
        model = _load_model_with_fallbacks(
            cfg.rerank_model,
            AutoModelForCausalLM,
            cfg.rerank_model_kwargs or {},
            cfg.device,
            "Layerwise reranker"
        )
        return LayerwiseReranker(model, tokenizer), chosen
    
    elif chosen in ("qwen3", "qwen", "qwen3_reranker"):
        model = _load_model_with_fallbacks(
            cfg.rerank_model,
            AutoModelForCausalLM,
            cfg.rerank_model_kwargs or {},
            cfg.device,
            "Qwen3 reranker"
        )
        return Qwen3Reranker(model, tokenizer), chosen
    
    elif chosen in ("seq_cls", "sequence_classification"):
        model = _load_model_with_fallbacks(
            cfg.rerank_model,
            AutoModelForSequenceClassification,
            cfg.rerank_model_kwargs or {},
            cfg.device,
            "Sequence classification reranker"
        )
        return SequenceClassificationReranker(model, tokenizer), chosen
    
    else:
        raise ValueError(f"Unsupported rerank method: {chosen}")


def rerank(
    reranker: Optional[BaseReranker],
    query: str,
    candidates: List[Tuple[str, float]],
    docid_to_text: Dict[str, str],
    *,
    method: str = "none",
    batch_size: Optional[int] = None,
    max_length: Optional[int] = None,
    **kwargs
) -> List[Tuple[str, float]]:
    """Unified rerank interface.
    
    Args:
        reranker: Reranker instance (can be None)
        query: Query string
        candidates: List of (doc_id, score) tuples
        docid_to_text: Mapping from doc_id to text content
        method: Reranking method name
        batch_size: Batch size for processing
        max_length: Maximum input length
        **kwargs: Additional method-specific arguments
    
    Returns:
        Reranked list of (doc_id, score) tuples
    """
    if reranker is None or method == "none" or not candidates:
        return list(candidates)
    
    # Set defaults based on method
    method_defaults = {
        "gemma": {"batch_size": 32, "max_length": 1024},
        "bge_gemma": {"batch_size": 32, "max_length": 1024},
        "layerwise": {"batch_size": 32, "max_length": 1024},
        "minicpm_layerwise": {"batch_size": 32, "max_length": 1024},
        "bge_layerwise": {"batch_size": 32, "max_length": 1024},
        "qwen3": {"batch_size": 32, "max_length": 8192},
        "qwen": {"batch_size": 32, "max_length": 8192},
        "qwen3_reranker": {"batch_size": 32, "max_length": 8192},
        "seq_cls": {"batch_size": 16, "max_length": 512},
        "sequence_classification": {"batch_size": 16, "max_length": 512},
    }
    
    defaults = method_defaults.get(method.lower(), {"batch_size": 32, "max_length": 1024})
    effective_batch_size = batch_size if batch_size is not None else defaults["batch_size"]
    effective_max_length = max_length if max_length is not None else defaults["max_length"]
    
    return reranker.rerank(
        query,
        candidates,
        docid_to_text,
        batch_size=effective_batch_size,
        max_length=effective_max_length,
        **kwargs
    )


# =============================
# Legacy API Support
# =============================
# These functions maintain backward compatibility with existing code

def build_gemma_reranker(cfg: Config):
    """Legacy: Build BGE-Gemma reranker."""
    reranker, _ = build_reranker(cfg, "gemma")
    if reranker:
        return reranker.model, reranker.tokenizer
    return None, None


def build_qwen3_reranker(cfg: Config):
    """Legacy: Build Qwen3 reranker."""
    reranker, _ = build_reranker(cfg, "qwen3")
    if reranker:
        return reranker.model, reranker.tokenizer
    return None, None


def build_sequence_classification_reranker(cfg: Config):
    """Legacy: Build sequence classification reranker."""
    reranker, _ = build_reranker(cfg, "seq_cls")
    if reranker:
        return reranker.model, reranker.tokenizer
    return None, None


def rerank_gemma(model, tokenizer, query, candidates, docid_to_text, batch_size=32, max_length=1024):
    """Legacy: Rerank using Gemma."""
    reranker = GemmaReranker(model, tokenizer)
    return reranker.rerank(query, candidates, docid_to_text, batch_size, max_length)


def rerank_qwen3(model, tokenizer, query, candidates, docid_to_text, batch_size=32, max_length=8192):
    """Legacy: Rerank using Qwen3."""
    reranker = Qwen3Reranker(model, tokenizer)
    return reranker.rerank(query, candidates, docid_to_text, batch_size, max_length)


def rerank_sequence_classification(model, tokenizer, query, candidates, docid_to_text, batch_size=16, max_length=512):
    """Legacy: Rerank using sequence classification."""
    reranker = SequenceClassificationReranker(model, tokenizer)
    return reranker.rerank(query, candidates, docid_to_text, batch_size, max_length)
