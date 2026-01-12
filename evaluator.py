import json
import ast
import numpy as np
from typing import List, Dict, Any, Optional, Set, Union
from collections import Counter
from itertools import combinations


# Distance Functions for Krippendorff's Alpha

def nominal_distance(a: Any, b: Any) -> float:
    """Nominal distance: 0 if same, 1 if different."""
    return 0.0 if a == b else 1.0


def jaccard_distance(a: Union[Set, List, str], b: Union[Set, List, str]) -> float:
    """
    Jaccard distance for sets/multi-label: 1 - |A ∩ B| / |A ∪ B|
    Returns 0 if both empty, 1 if no overlap.
    """
    set_a = _to_set(a)
    set_b = _to_set(b)
    
    if not set_a and not set_b:
        return 0.0
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return 1.0 - (intersection / union) if union > 0 else 0.0


def hamming_distance(a: Union[Set, List, str], b: Union[Set, List, str], 
                     all_labels: Optional[Set] = None,
                     return_details: bool = False) -> Union[float, Dict[str, Any]]:
    """
    Normalized Hamming distance for multi-label: |A Δ B| / |all_labels|
    If all_labels not provided, uses union of a and b.
    If return_details=True, returns dict with distance, fp_count, fn_count.
    """
    set_a = _to_set(a)
    set_b = _to_set(b)
    
    if all_labels is None:
        all_labels = set_a | set_b
    
    if not all_labels:
        return {"distance": 0.0, "fp_count": 0, "fn_count": 0} if return_details else 0.0
    
    fp = set_a - set_b  # in pred but not truth
    fn = set_b - set_a  # in truth but not pred
    distance = (len(fp) + len(fn)) / len(all_labels)
    
    if return_details:
        return {"distance": distance, "fp_count": len(fp), "fn_count": len(fn), 
                "fp": fp, "fn": fn}
    return distance


def _to_set(value: Union[Set, List, str, Any]) -> Set:
    """Convert a value to a set for multi-label comparison."""
    if isinstance(value, set):
        return value
    if isinstance(value, (list, tuple)):
        return set(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return set(parsed)
            except (ValueError, SyntaxError):
                pass
        # Single value as string
        return {value}
    # Single value (int, etc.)
    return {value}



def krippendorff_alpha(reliability_data: List[List[Any]], 
                       distance_fn: str = "nominal",
                       all_labels: Optional[Set] = None) -> float:
    """
    Calculate Krippendorff's alpha for inter-coder reliability.
    
    Args:
        reliability_data: List of lists where each inner list contains
                         codes from different coders for the same unit.
                         e.g., [[coder1_unit1, coder2_unit1], [coder1_unit2, coder2_unit2], ...]
        distance_fn: Distance function to use: "nominal", "jaccard", or "hamming"
        all_labels: For hamming distance, the complete set of possible labels
    
    Returns:
        Krippendorff's alpha coefficient (-1 to 1, where 1 is perfect agreement)
    """
    if not reliability_data:
        return 0.0
    
    # Select distance function
    if distance_fn == "nominal":
        dist_func = nominal_distance
    elif distance_fn == "jaccard":
        dist_func = jaccard_distance
    elif distance_fn == "hamming":
        dist_func = lambda a, b: hamming_distance(a, b, all_labels)
    else:
        raise ValueError(f"Unknown distance function: {distance_fn}")
    
    # Filter out units with fewer than 2 coders
    valid_data = [unit for unit in reliability_data if len([v for v in unit if v is not None]) >= 2]
    
    if not valid_data:
        return 0.0
    
    # Calculate observed disagreement (Do)
    observed_disagreement = 0.0
    n_pairs_observed = 0
    
    for unit in valid_data:
        valid_codes = [v for v in unit if v is not None]
        if len(valid_codes) < 2:
            continue
        
        # All pairs within this unit
        for c1, c2 in combinations(valid_codes, 2):
            observed_disagreement += dist_func(c1, c2)
            n_pairs_observed += 1
    
    if n_pairs_observed == 0:
        return 1.0  # Perfect agreement if no pairs to compare
    
    observed_disagreement /= n_pairs_observed
    
    # Calculate expected disagreement (De) - disagreement by chance
    # Collect all values across all units
    all_values = []
    for unit in valid_data:
        all_values.extend([v for v in unit if v is not None])
    
    if len(all_values) < 2:
        return 1.0
    
    # Expected disagreement: average distance between all possible pairs
    expected_disagreement = 0.0
    n_pairs_expected = 0
    
    for v1, v2 in combinations(all_values, 2):
        expected_disagreement += dist_func(v1, v2)
        n_pairs_expected += 1
    
    if n_pairs_expected == 0:
        return 1.0
    
    expected_disagreement /= n_pairs_expected
    
    # Krippendorff's alpha = 1 - (Do / De)
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else 0.0
    
    alpha = 1.0 - (observed_disagreement / expected_disagreement)
    return alpha


# Classification Metrics (Precision, Recall, F1, Confusion Matrix)

def confusion_matrix(predictions: List[Any], ground_truth: List[Any], 
                     labels: Optional[List[Any]] = None) -> Dict[str, Any]:
    """
    Build a confusion matrix for multi-class classification.
    
    Returns:
        Dict with 'matrix' (2D list), 'labels' (row/column order), and per-class counts
    """
    if labels is None:
        labels = sorted(set(predictions) | set(ground_truth))
    
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n_labels = len(labels)
    
    matrix = [[0] * n_labels for _ in range(n_labels)]
    
    for pred, truth in zip(predictions, ground_truth):
        if pred in label_to_idx and truth in label_to_idx:
            matrix[label_to_idx[truth]][label_to_idx[pred]] += 1
    
    return {
        "matrix": matrix,
        "labels": labels
    }


def precision_recall_f1(predictions: List[Any], ground_truth: List[Any],
                        labels: Optional[List[Any]] = None) -> Dict[str, Any]:
    """
    Calculate precision, recall, and F1 score for multi-class classification.
    
    Returns:
        Dict with macro averages and per-class metrics
    """
    if labels is None:
        labels = sorted(set(predictions) | set(ground_truth))
    
    per_class = {}
    
    for label in labels:
        tp = sum(1 for p, t in zip(predictions, ground_truth) if p == label and t == label)
        fp = sum(1 for p, t in zip(predictions, ground_truth) if p == label and t != label)
        fn = sum(1 for p, t in zip(predictions, ground_truth) if p != label and t == label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn  # number of true instances for this label
        }
    
    # Macro averages (unweighted mean across classes)
    macro_precision = np.mean([m["precision"] for m in per_class.values()])
    macro_recall = np.mean([m["recall"] for m in per_class.values()])
    macro_f1 = np.mean([m["f1"] for m in per_class.values()])
    
    # Weighted averages (weighted by support)
    total_support = sum(m["support"] for m in per_class.values())
    if total_support > 0:
        weighted_precision = sum(m["precision"] * m["support"] for m in per_class.values()) / total_support
        weighted_recall = sum(m["recall"] * m["support"] for m in per_class.values()) / total_support
        weighted_f1 = sum(m["f1"] * m["support"] for m in per_class.values()) / total_support
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0
    
    return {
        "macro": {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1": round(macro_f1, 4)
        },
        "weighted": {
            "precision": round(weighted_precision, 4),
            "recall": round(weighted_recall, 4),
            "f1": round(weighted_f1, 4)
        },
        "per_class": per_class
    }


# Basic Metrics

def accuracy(predictions: List[int], ground_truth: List[int]) -> float:
    """Calculate accuracy."""
    if not predictions:
        return 0.0
    return sum(p == t for p, t in zip(predictions, ground_truth)) / len(predictions)


def hamming_loss(pred_sets: List[set], truth_sets: List[set], 
                 return_details: bool = False) -> Union[float, Dict[str, Any]]:
    """Calculate Hamming loss for multi-label classification.
    If return_details=True, returns dict with loss, total_fp, total_fn."""
    if not pred_sets:
        return {"loss": 0.0, "total_fp": 0, "total_fn": 0} if return_details else 0.0
    
    losses, total_fp, total_fn = [], 0, 0
    for pred, truth in zip(pred_sets, truth_sets):
        union = pred | truth
        if union:
            fp, fn = len(pred - truth), len(truth - pred)
            losses.append((fp + fn) / len(union))
            total_fp += fp
            total_fn += fn
    
    loss = np.mean(losses) if losses else 0.0
    if return_details:
        return {"loss": loss, "total_fp": total_fp, "total_fn": total_fn,
                "fp_rate": total_fp / (total_fp + total_fn) if (total_fp + total_fn) else 0.0}
    return loss


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


def evaluate_phase(results: Dict[str, List[Any]], ground_truth: Optional[Dict[str, int]] = None, 
                   agreements: Optional[Dict[str, bool]] = None,
                   task_type: str = "class") -> Dict[str, Any]:
    """
    Evaluate a coding phase against ground truth or using inter-coder agreement.
    
    Args:
        results: Dict mapping text_id to list of agent responses (with .code or ['code'])
        ground_truth: Optional dict mapping text_id to ground truth label. If None or empty,
                      evaluation falls back to inter-coder agreement only.
        agreements: Optional dict mapping text_id to agreement boolean
        task_type: "class" for multi-class, "label" for multi-label (affects distance metric)
    
    Returns:
        Evaluation metrics dict
    """
    has_ground_truth = ground_truth is not None and len(ground_truth) > 0
    
    # Build reliability data for Krippendorff's alpha (always computed)
    reliability_data = []
    all_codes_flat = []
    
    for text_id, responses in results.items():
        codes = [r.code if hasattr(r, 'code') else r['code'] for r in responses]
        reliability_data.append(codes)
        all_codes_flat.extend(codes)
    
    # Determine all possible labels (for hamming distance normalization)
    all_labels = set()
    for code in all_codes_flat:
        if task_type == "label":
            all_labels.update(_to_set(code))
        else:
            all_labels.add(code)
    
    if has_ground_truth:
        # Ground truth evaluation mode
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
            },
            "has_ground_truth": True
        }
        
        # Add precision, recall, F1, and confusion matrix
        prf1 = precision_recall_f1(consensus_preds, truths)
        eval_dict["precision"] = prf1["macro"]["precision"]
        eval_dict["recall"] = prf1["macro"]["recall"]
        eval_dict["f1"] = prf1["macro"]["f1"]
        eval_dict["metrics_detail"] = prf1
        
        cm = confusion_matrix(consensus_preds, truths)
        eval_dict["confusion_matrix"] = cm
        
    else:
        # Inter-coder agreement mode (no ground truth)
        eval_dict = {
            "total": len(results),
            "has_ground_truth": False
        }
    
    # Add agreement metrics (available in both modes)
    if agreements is not None:
        agreement_rate = calculate_agreement_rate(agreements)
        eval_dict["agreement_rate"] = round(agreement_rate, 4)
        eval_dict["agreements"] = sum(agreements.values())
        eval_dict["disagreements"] = len(agreements) - sum(agreements.values())
    
    # Compute Krippendorff's alpha (always, since we always have inter-coder data)
    if reliability_data:
        # Nominal distance (always computed)
        alpha_nominal = krippendorff_alpha(reliability_data, distance_fn="nominal")
        eval_dict["krippendorff_alpha_nominal"] = round(alpha_nominal, 4)
        
        # For multi-label tasks, also compute Jaccard and Hamming variants
        if task_type == "label":
            alpha_jaccard = krippendorff_alpha(reliability_data, distance_fn="jaccard")
            alpha_hamming = krippendorff_alpha(reliability_data, distance_fn="hamming", 
                                               all_labels=all_labels)
            eval_dict["krippendorff_alpha_jaccard"] = round(alpha_jaccard, 4)
            eval_dict["krippendorff_alpha_hamming"] = round(alpha_hamming, 4)
    
    return eval_dict


class Evaluator:
    """Tracks evaluation metrics across simulation runs."""
    
    def __init__(self, ground_truth: Optional[Dict[str, int]] = None, task_type: str = "class"):
        self.ground_truth = ground_truth if ground_truth else {}
        self.has_ground_truth = bool(self.ground_truth)
        self.task_type = task_type
        self.runs: List[Dict[str, Any]] = []
    
    def evaluate_run(self, coding_results: Dict, discussion_results: Dict = None, 
                     coding_agreements: Dict[str, bool] = None,
                     discussion_agreements: Dict[str, bool] = None,
                     log_fn=None) -> Dict[str, Any]:
        """
        Evaluate a complete simulation run.
        
        If ground truth is available, computes accuracy metrics.
        If no ground truth, falls back to inter-coder agreement metrics only.
        
        Args:
            coding_results: Dict mapping text_id to list of agent responses from coding phase
            discussion_results: Optional dict mapping text_id to list of agent responses from discussion phase
            coding_agreements: Optional dict mapping text_id to agreement boolean from coding phase
            discussion_agreements: Optional dict mapping text_id to agreement boolean from discussion phase
            log_fn: Optional logging function
        """
        coding_eval = evaluate_phase(coding_results, self.ground_truth, coding_agreements, 
                                     task_type=self.task_type)
        
        if log_fn:
            if coding_eval.get('has_ground_truth'):
                log_fn(f"\nCoding Phase: {coding_eval['accuracy']:.2%} accuracy ({coding_eval['correct']}/{coding_eval['total']})")
                log_fn(f"   Precision: {coding_eval['precision']:.2%}  Recall: {coding_eval['recall']:.2%}  F1: {coding_eval['f1']:.2%}")
                for i, acc in coding_eval['per_agent_accuracy'].items():
                    log_fn(f"   Agent {i+1}: {acc:.2%}")
            else:
                log_fn(f"\nCoding Phase: {coding_eval['total']} texts coded (no ground truth available)")
            
            if 'agreement_rate' in coding_eval:
                total_items = coding_eval['agreements'] + coding_eval['disagreements']
                log_fn(f"   Agreement Rate: {coding_eval['agreement_rate']:.2%} ({coding_eval['agreements']}/{total_items})")
            
            # Log Krippendorff's alpha
            if 'krippendorff_alpha_nominal' in coding_eval:
                log_fn(f"   Krippendorff's α (nominal): {coding_eval['krippendorff_alpha_nominal']:.4f}")
            if 'krippendorff_alpha_jaccard' in coding_eval:
                log_fn(f"   Krippendorff's α (jaccard): {coding_eval['krippendorff_alpha_jaccard']:.4f}")
            if 'krippendorff_alpha_hamming' in coding_eval:
                log_fn(f"   Krippendorff's α (hamming): {coding_eval['krippendorff_alpha_hamming']:.4f}")
        
        discussion_eval = None
        if discussion_results:
            discussion_eval = evaluate_phase(discussion_results, self.ground_truth, discussion_agreements,
                                            task_type=self.task_type)
            
            if log_fn:
                if discussion_eval.get('has_ground_truth'):
                    improvement = discussion_eval['accuracy'] - coding_eval['accuracy']
                    discussion_eval['improvement'] = round(improvement, 4)
                    log_fn(f"\nPost-Discussion: {discussion_eval['accuracy']:.2%} accuracy")
                    log_fn(f"   Precision: {discussion_eval['precision']:.2%}  Recall: {discussion_eval['recall']:.2%}  F1: {discussion_eval['f1']:.2%}")
                    log_fn(f"   Improvement: {improvement:+.2%}")
                else:
                    log_fn(f"\nPost-Discussion: {discussion_eval['total']} texts (no ground truth available)")
                
                if 'agreement_rate' in discussion_eval:
                    total_items = discussion_eval['agreements'] + discussion_eval['disagreements']
                    log_fn(f"   Agreement Rate: {discussion_eval['agreement_rate']:.2%} ({discussion_eval['agreements']}/{total_items})")
                    if 'agreement_rate' in coding_eval:
                        agreement_improvement = discussion_eval['agreement_rate'] - coding_eval['agreement_rate']
                        log_fn(f"   Agreement Improvement: {agreement_improvement:+.2%}")
                
                # Log Krippendorff's alpha
                if 'krippendorff_alpha_nominal' in discussion_eval:
                    log_fn(f"   Krippendorff's α (nominal): {discussion_eval['krippendorff_alpha_nominal']:.4f}")
                if 'krippendorff_alpha_jaccard' in discussion_eval:
                    log_fn(f"   Krippendorff's α (jaccard): {discussion_eval['krippendorff_alpha_jaccard']:.4f}")
                if 'krippendorff_alpha_hamming' in discussion_eval:
                    log_fn(f"   Krippendorff's α (hamming): {discussion_eval['krippendorff_alpha_hamming']:.4f}")
        
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
        
        result = {
            "num_runs": len(self.runs),
            "has_ground_truth": self.has_ground_truth,
            "task_type": self.task_type
        }
        
        # Only compute accuracy stats if ground truth is available
        if self.has_ground_truth:
            coding_accs = [r["coding"]["accuracy"] for r in self.runs if r["coding"].get("has_ground_truth")]
            coding_f1s = [r["coding"]["f1"] for r in self.runs if r["coding"].get("has_ground_truth")]
            coding_precisions = [r["coding"]["precision"] for r in self.runs if r["coding"].get("has_ground_truth")]
            coding_recalls = [r["coding"]["recall"] for r in self.runs if r["coding"].get("has_ground_truth")]
            
            disc_accs = [r["discussion"]["accuracy"] for r in self.runs 
                        if r["discussion"] and r["discussion"].get("has_ground_truth")]
            disc_f1s = [r["discussion"]["f1"] for r in self.runs 
                       if r["discussion"] and r["discussion"].get("has_ground_truth")]
            improvements = [r["discussion"]["improvement"] for r in self.runs 
                          if r["discussion"] and "improvement" in r["discussion"]]
            
            if coding_accs:
                result["coding_accuracy"] = calc_stats(coding_accs)
            if coding_f1s:
                result["coding_f1"] = calc_stats(coding_f1s)
            if coding_precisions:
                result["coding_precision"] = calc_stats(coding_precisions)
            if coding_recalls:
                result["coding_recall"] = calc_stats(coding_recalls)
            if disc_accs:
                result["discussion_accuracy"] = calc_stats(disc_accs)
            if disc_f1s:
                result["discussion_f1"] = calc_stats(disc_f1s)
            if improvements:
                result["improvement"] = calc_stats(improvements)
        
        # Agreement stats (available regardless of ground truth)
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
        
        if coding_agreement_rates:
            result["coding_agreement_rate"] = calc_stats(coding_agreement_rates)
        if discussion_agreement_rates:
            result["discussion_agreement_rate"] = calc_stats(discussion_agreement_rates)
        if agreement_improvements:
            result["agreement_improvement"] = calc_stats(agreement_improvements)
        
        # Krippendorff's alpha stats
        coding_alpha_nominal = [r["coding"]["krippendorff_alpha_nominal"] for r in self.runs 
                               if "krippendorff_alpha_nominal" in r["coding"]]
        coding_alpha_jaccard = [r["coding"]["krippendorff_alpha_jaccard"] for r in self.runs 
                               if "krippendorff_alpha_jaccard" in r["coding"]]
        coding_alpha_hamming = [r["coding"]["krippendorff_alpha_hamming"] for r in self.runs 
                               if "krippendorff_alpha_hamming" in r["coding"]]
        
        if coding_alpha_nominal:
            result["coding_krippendorff_alpha_nominal"] = calc_stats(coding_alpha_nominal)
        if coding_alpha_jaccard:
            result["coding_krippendorff_alpha_jaccard"] = calc_stats(coding_alpha_jaccard)
        if coding_alpha_hamming:
            result["coding_krippendorff_alpha_hamming"] = calc_stats(coding_alpha_hamming)
        
        # Discussion Krippendorff's alpha
        disc_alpha_nominal = [r["discussion"]["krippendorff_alpha_nominal"] for r in self.runs 
                             if r["discussion"] and "krippendorff_alpha_nominal" in r["discussion"]]
        disc_alpha_jaccard = [r["discussion"]["krippendorff_alpha_jaccard"] for r in self.runs 
                             if r["discussion"] and "krippendorff_alpha_jaccard" in r["discussion"]]
        
        if disc_alpha_nominal:
            result["discussion_krippendorff_alpha_nominal"] = calc_stats(disc_alpha_nominal)
        if disc_alpha_jaccard:
            result["discussion_krippendorff_alpha_jaccard"] = calc_stats(disc_alpha_jaccard)
        
        return result


def load_ground_truth(df, ground_truth_column: Optional[str] = None) -> Optional[Dict[str, int]]:
    """
    Create ground truth mapping from DataFrame.
    
    Args:
        df: The DataFrame containing the data
        ground_truth_column: The column name containing ground truth labels.
                            If None, tries 'Label' as default. Returns None if column doesn't exist.
    
    Returns:
        Dict mapping text_id to ground truth label, or None if no ground truth column exists.
    """
    # Determine which column to use
    col = ground_truth_column
    if col is None:
        col = 'Label' if 'Label' in df.columns else None
    
    if col is None or col not in df.columns:
        return None
    
    # Build ground truth dict
    ground_truth = {}
    for i, label in enumerate(df[col]):
        try:
            # Handle numeric labels
            ground_truth[f"Text-{i+1}"] = int(label)
        except (ValueError, TypeError):
            # For non-numeric labels (strings), store as-is
            ground_truth[f"Text-{i+1}"] = label
    
    return ground_truth


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
