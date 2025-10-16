"""
Evaluation module.

Provides standard information retrieval system evaluation metric calculation 
"""
import logging
from typing import Dict, List, Tuple, Any, Optional

import ir_measures as irms


logger = logging.getLogger(__name__)


def evaluate_runs(
    run_dict: Dict[str, List[Tuple[str, float]]], 
    qrels: List[Any],  # List[ir_datasets.Qrel]
    measures: Optional[List[Any]] = None,
    eval_at: Optional[List[int]] = None
) -> Dict[str, float]:
    """Evaluate retrieval results.
    
    Calculates retrieval performance using standard IR evaluation metrics.
    
    Args:
        run_dict: Mapping from query_id to a ranked list of (doc_id, score)
        qrels: Relevance judgments from ir_datasets (qrels_iter())
        measures: List of metric spec strings (e.g., ["nDCG@10", "P@10", "R@100", "MAP"]).
                  If None, defaults to [nDCG@k, P@k for k in eval_at] plus R@100 and MAP.
        eval_at: Cutoff list for position-based metrics; defaults to [10, 100] if None.
        
    Returns:
        Dict[str, float]: Mapping from metric name to aggregate score
        
    Raises:
        ValueError: When the input format is invalid
    """
    if not run_dict:
        logger.warning(f"Run is empty, skipping evaluation")
        return {}
    
    # Default evaluation metrics based on eval_at
    if measures is None:
        if eval_at is None:
            eval_at = [10, 100]  # Default cutoffs
        
        measures = []
        # Add position-based metrics for each cutoff
        for k in eval_at:
            measures.extend([f"nDCG@{k}", f"P@{k}"])
        
        # Add fixed metrics
        measures.extend([
            "R@100",
            "MAP"
        ])
    
    try:
        # Convert to the format required by ir_measures
        run = []
        for qid, items in run_dict.items():
            for rank, (docid, score) in enumerate(items, start=1):
                run.append(
                    irms.ScoredDoc(qid, docid, score)
                )
        
        # Display results
        res = {}
        for measure_str in measures:
            measure = irms.parse_measure(measure_str)
            res[measure_str] = measure.calc_aggregate(qrels, run)

        return res
        
    except Exception as e:
        logger.error(f"Error evaluating: {e}")
        raise

