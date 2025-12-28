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


def calculate_agreement_rate(agreements: Dict[str, bool]) -> float:
    """Calculate agreement rate from a dictionary of agreement booleans."""
    if not agreements:
        return 0.0
    return sum(agreements.values()) / len(agreements)


def evaluate_phase(results: Dict[str, List[Any]], ground_truth: Dict[str, int], 
                   agreements: Dict[str, bool] = None) -> Dict[str, Any]:
    """
    Evaluate a coding phase against ground truth.
    
    Args:
        results: Dict mapping text_id to list of agent responses (with .code or ['code'])
        ground_truth: Dict mapping text_id to ground truth label
        agreements: Optional dict mapping text_id to agreement boolean
    
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
        
        for i, code in enumerate(codes):
            per_agent.setdefault(i, {"preds": [], "truths": []})
            per_agent[i]["preds"].append(code)
            per_agent[i]["truths"].append(truth)
        
        consensus_preds.append(majority_vote(codes))
        truths.append(truth)
    
    eval_dict = {
        "total": len(truths),
        "correct": sum(p == t for p, t in zip(consensus_preds, truths)),
        "accuracy": round(accuracy(consensus_preds, truths), 4),
        "per_agent_accuracy": {
            i: round(accuracy(d["preds"], d["truths"]), 4) 
            for i, d in per_agent.items()
        }
    }
    
    if agreements is not None:
        agreement_rate = calculate_agreement_rate(agreements)
        eval_dict["agreement_rate"] = round(agreement_rate, 4)
        eval_dict["agreements"] = sum(agreements.values())
        eval_dict["disagreements"] = len(agreements) - sum(agreements.values())
    
    return eval_dict


class Evaluator:
    """Tracks evaluation metrics across simulation runs."""
    
    def __init__(self, ground_truth: Dict[str, int]):
        self.ground_truth = ground_truth
        self.runs: List[Dict[str, Any]] = []
    
    def evaluate_run(self, coding_results: Dict, discussion_results: Dict = None, 
                     coding_agreements: Dict[str, bool] = None,
                     discussion_agreements: Dict[str, bool] = None,
                     log_fn=None) -> Dict[str, Any]:
        """
        Evaluate a complete simulation run.
        
        Args:
            coding_results: Dict mapping text_id to list of agent responses from coding phase
            discussion_results: Optional dict mapping text_id to list of agent responses from discussion phase
            coding_agreements: Optional dict mapping text_id to agreement boolean from coding phase
            discussion_agreements: Optional dict mapping text_id to agreement boolean from discussion phase
            log_fn: Optional logging function
        """
        coding_eval = evaluate_phase(coding_results, self.ground_truth, coding_agreements)
        
        if log_fn:
            log_fn(f"\nCoding Phase: {coding_eval['accuracy']:.2%} accuracy ({coding_eval['correct']}/{coding_eval['total']})")
            if 'agreement_rate' in coding_eval:
                total_items = coding_eval['agreements'] + coding_eval['disagreements']
                log_fn(f"   Agreement Rate: {coding_eval['agreement_rate']:.2%} ({coding_eval['agreements']}/{total_items})")
            for i, acc in coding_eval['per_agent_accuracy'].items():
                log_fn(f"   Agent {i+1}: {acc:.2%}")
        
        discussion_eval = None
        if discussion_results:
            discussion_eval = evaluate_phase(discussion_results, self.ground_truth, discussion_agreements)
            improvement = discussion_eval['accuracy'] - coding_eval['accuracy']
            discussion_eval['improvement'] = round(improvement, 4)
            
            if log_fn:
                log_fn(f"\nPost-Discussion: {discussion_eval['accuracy']:.2%} accuracy")
                if 'agreement_rate' in discussion_eval:
                    total_items = discussion_eval['agreements'] + discussion_eval['disagreements']
                    log_fn(f"   Agreement Rate: {discussion_eval['agreement_rate']:.2%} ({discussion_eval['agreements']}/{total_items})")
                log_fn(f"   Improvement: {improvement:+.2%}")
                if 'agreement_rate' in coding_eval and 'agreement_rate' in discussion_eval:
                    agreement_improvement = discussion_eval['agreement_rate'] - coding_eval['agreement_rate']
                    log_fn(f"   Agreement Improvement: {agreement_improvement:+.2%}")
        
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
        
        coding_agreement_rates = []
        discussion_agreement_rates = []
        agreement_improvements = []
        
        for r in self.runs:
            if "agreement_rate" in r["coding"]:
                coding_agreement_rates.append(r["coding"]["agreement_rate"])
            if r["discussion"] and "agreement_rate" in r["discussion"]:
                discussion_agreement_rates.append(r["discussion"]["agreement_rate"])
                if "agreement_rate" in r["coding"]:
                    agreement_improvements.append(
                        r["discussion"]["agreement_rate"] - r["coding"]["agreement_rate"]
                    )
        
        result = {
            "num_runs": len(self.runs),
            "coding_accuracy": calc_stats(coding_accs),
            "discussion_accuracy": calc_stats(disc_accs) if disc_accs else None,
            "improvement": calc_stats(improvements) if improvements else None,
        }
        
        if coding_agreement_rates:
            result["coding_agreement_rate"] = calc_stats(coding_agreement_rates)
        if discussion_agreement_rates:
            result["discussion_agreement_rate"] = calc_stats(discussion_agreement_rates)
        if agreement_improvements:
            result["agreement_improvement"] = calc_stats(agreement_improvements)
        
        return result


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
    all_coding_agreements = {}
    all_discussion_agreements = {}
    
    for chunk in chunks:
        coding_phase = chunk.get("coding_phase", {})
        discussion_phase = chunk.get("discussion_phase", {})
        
        all_coding.update(coding_phase.get("results", {}))
        all_discussion.update(discussion_phase.get("results", {}))
        
        if "agreements" in coding_phase:
            all_coding_agreements.update(coding_phase["agreements"])
        if "agreements" in discussion_phase:
            all_discussion_agreements.update(discussion_phase["agreements"])
    
    evaluator = Evaluator(ground_truth)
    return evaluator.evaluate_run(
        all_coding, 
        all_discussion or None,
        coding_agreements=all_coding_agreements if all_coding_agreements else None,
        discussion_agreements=all_discussion_agreements if all_discussion_agreements else None,
        log_fn=print
    )
