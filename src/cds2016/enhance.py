"""
Query enhancement utility.

Provides a configurable function to rewrite user queries into a form
more suitable for medical (e.g., MRI) report retrieval, using an
LLM endpoint compatible with OpenAI-style chat completions.
"""
from typing import Tuple, Any
import logging
import requests
import textwrap


logger = logging.getLogger(__name__)


def enhance_query_raw(
    original_query: str,
    endpoint: str,
    model: str,
    timeout: Tuple[int, int] = (60, 150),
) -> str:
    """Low-level API: rewrite the query using explicit endpoint/model.

    Returns the original query on failure.
    """
    prompt = textwrap.dedent(f"""\
You are an expert in clinical information retrieval. Please rewrite the user's query(EHR note or case description/summary) into a descriptive statement optimized for **the semantic search**.

**Requirements:**
1. Emphasize the findings section using different phrasing
2. Use English
3. Do not include any extra explanation

Original query: {original_query}

Please return only the rewritten query (no extra explanation):
""")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Reasoning: high"},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "num_ctx": 8192,
            "num_predict": 2048,
            "temperature": 0.3
        }
    }
    try:
        logger.debug("Sending query enhancement request to endpoint: %s", endpoint)
        response = requests.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        enhanced_query = data["message"]["content"].strip()
        logger.info("Query enhancement: '%s' -> '%s'", original_query, enhanced_query)

        return enhanced_query
    
    except requests.exceptions.ConnectTimeout as e:
        logger.warning(f"Cannot connect to LLM endpoint (>60s), using original query: {e}")
    
    except requests.exceptions.ReadTimeout as e:
        logger.warning(f"Server took too long to respond (>150s). Please check server status. Using original query: {e}")
    
    except requests.exceptions.RequestException as e:
        logger.warning(f"Other requests exception, using original query: {e}")
    
    except Exception as e:
        logger.warning(f"Query enhancement failed (non-requests error). Using original query: {e}")
    
    return original_query


def enhance_query(
    original_query: str,
    cfg_or_endpoint: Any,
    model: str | None = None,
    timeout: Tuple[int, int] = (60, 150),
) -> str:
    """High-level API: supports two calling styles.

    1) enhance_query(original_query, cfg): reads endpoint/model from cfg
    2) enhance_query(original_query, endpoint, model=...): explicit endpoint/model
    """
    # Config-style usage
    if not isinstance(cfg_or_endpoint, str):
        cfg = cfg_or_endpoint
        endpoint = getattr(cfg, 'enhance_endpoint', None)
        mdl = getattr(cfg, 'enhance_model', None)
        if not endpoint or not mdl:
            logger.warning("Enhance config missing endpoint/model; returning original query")
            return original_query
        return enhance_query_raw(original_query, endpoint=endpoint, model=mdl, timeout=timeout)

    # Explicit endpoint + model style
    endpoint = cfg_or_endpoint
    if not model:
        logger.warning("Model not provided for explicit enhance_query call; returning original query")
        return original_query
    return enhance_query_raw(original_query, endpoint=endpoint, model=model, timeout=timeout)