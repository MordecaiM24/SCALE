"""Simple evaluation module for SCALE content analysis simulation."""

import json
import numpy as np
from typing import List, Dict, Any
from collections import Counter


def accuracy(predictions: List[int], ground_truth: List[int]) -> float:
    """Calculate accuracy."""
    if not predictions:
        return 0.0
    return sum(p == t for p, t in zip(predictions, ground_truth)) / len(predictions)


def hamming_loss(pred_sets: List[set], truth_sets: List[set]) -> float:
    """Calculate Hamming loss for multi-label classification."""
    if not pred_sets:
        return 0.0
    losses = []
    for pred, truth in zip(pred_sets, truth_sets):
        union = pred | truth
        if union:
            losses.append(len(pred ^ truth) / len(union))
    return np.mean(losses) if losses else 0.0


def majority_vote(codes: List[int]) -> int:
    """Get majority vote from a list of codes."""
    return Counter(codes).most_common(1)[0][0]


def calc_stats(values: List[float]) -> Dict[str, float]:
    """Calculate descriptive statistics."""
    if not values:
        return {}
    return {
        "mean": round(np.mean(values), 4),
        "median": round(np.median(values), 4),
        "std": round(np.std(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def evaluate_phase(results: Dict[str, List[Any]], ground_truth: Dict[str, int]) -> Dict[str, Any]:
    """
    Evaluate a coding phase against ground truth.
    
    Args:
        results: Dict mapping text_id to list of agent responses (with .code or ['code'])
        ground_truth: Dict mapping text_id to ground truth label
    
    Returns:
        Evaluation metrics dict
    """
    consensus_preds, truths = [], []
    per_agent = {}
    
    for text_id, responses in results.items():
        if text_id not in ground_truth:
            continue
        
        truth = ground_truth[text_id]
        codes = [r.code if hasattr(r, 'code') else r['code'] for r in responses]
        
        # Track per-agent predictions
        for i, code in enumerate(codes):
            per_agent.setdefault(i, {"preds": [], "truths": []})
            per_agent[i]["preds"].append(code)
            per_agent[i]["truths"].append(truth)
        
        consensus_preds.append(majority_vote(codes))
        truths.append(truth)
    
    return {
        "total": len(truths),
        "correct": sum(p == t for p, t in zip(consensus_preds, truths)),
        "accuracy": round(accuracy(consensus_preds, truths), 4),
        "per_agent_accuracy": {
            i: round(accuracy(d["preds"], d["truths"]), 4) 
            for i, d in per_agent.items()
        }
    }


class Evaluator:
    """Tracks evaluation metrics across simulation runs."""
    
    def __init__(self, ground_truth: Dict[str, int]):
        self.ground_truth = ground_truth
        self.runs: List[Dict[str, Any]] = []
    
    def evaluate_run(self, coding_results: Dict, discussion_results: Dict = None, 
                     log_fn=None) -> Dict[str, Any]:
        """Evaluate a complete simulation run."""
        coding_eval = evaluate_phase(coding_results, self.ground_truth)
        
        if log_fn:
            log_fn(f"\nCoding Phase: {coding_eval['accuracy']:.2%} accuracy ({coding_eval['correct']}/{coding_eval['total']})")
            for i, acc in coding_eval['per_agent_accuracy'].items():
                log_fn(f"   Agent {i+1}: {acc:.2%}")
        
        discussion_eval = None
        if discussion_results:
            discussion_eval = evaluate_phase(discussion_results, self.ground_truth)
            improvement = discussion_eval['accuracy'] - coding_eval['accuracy']
            discussion_eval['improvement'] = round(improvement, 4)
            
            if log_fn:
                log_fn(f"\nPost-Discussion: {discussion_eval['accuracy']:.2%} accuracy")
                log_fn(f"   Improvement: {improvement:+.2%}")
        
        run_result = {
            "coding": coding_eval,
            "discussion": discussion_eval
        }
        self.runs.append(run_result)
        return run_result
    
    def aggregate_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics across all runs."""
        if not self.runs:
            return {}
        
        coding_accs = [r["coding"]["accuracy"] for r in self.runs]
        disc_accs = [r["discussion"]["accuracy"] for r in self.runs if r["discussion"]]
        improvements = [r["discussion"]["improvement"] for r in self.runs if r["discussion"]]
        
        return {
            "num_runs": len(self.runs),
            "coding_accuracy": calc_stats(coding_accs),
            "discussion_accuracy": calc_stats(disc_accs) if disc_accs else None,
            "improvement": calc_stats(improvements) if improvements else None,
        }


def load_ground_truth(df) -> Dict[str, int]:
    """Create ground truth mapping from DataFrame."""
    return {f"Text-{i+1}": int(label) for i, label in enumerate(df['Label'])}


def evaluate_results_file(results_path: str, ground_truth: Dict[str, int]) -> Dict[str, Any]:
    """Evaluate an existing results JSON file."""
    with open(results_path) as f:
        data = json.load(f)
    
    chunks = data if isinstance(data, list) else [data]
    
    all_coding = {}
    all_discussion = {}
    for chunk in chunks:
        all_coding.update(chunk.get("coding_phase", {}).get("results", {}))
        all_discussion.update(chunk.get("discussion_phase", {}).get("results", {}))
    
    evaluator = Evaluator(ground_truth)
    return evaluator.evaluate_run(all_coding, all_discussion or None, print)
