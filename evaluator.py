import json
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter


def _extract_code(response: Any) -> int:
    """Extract an integer code from dict-like or object responses."""
    return response.code if hasattr(response, 'code') else response['code']


def _extract_confidence(response: Any) -> Tuple[float, bool]:
    """Extract confidence if explicitly present; otherwise default to 1.0."""
    if hasattr(response, 'confidence'):
        return float(response.confidence), True
    if isinstance(response, dict) and 'confidence' in response:
        return float(response['confidence']), True
    return 1.0, False


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


def cohens_kappa_pairwise(ratings_matrix: List[List[int]]) -> Dict[str, float]:
    """Calculate Cohen's kappa for each pair of raters."""
    if not ratings_matrix:
        return {}

    num_raters = max((len(row) for row in ratings_matrix), default=0)
    if num_raters < 2:
        return {}

    kappas: Dict[str, float] = {}
    for i in range(num_raters):
        for j in range(i + 1, num_raters):
            paired = [(row[i], row[j]) for row in ratings_matrix if len(row) > j]
            if not paired:
                kappas[f"{i}-{j}"] = 0.0
                continue

            obs = np.array(paired)
            po = float(np.mean(obs[:, 0] == obs[:, 1]))
            values = np.unique(obs)

            p1 = {v: np.mean(obs[:, 0] == v) for v in values}
            p2 = {v: np.mean(obs[:, 1] == v) for v in values}
            pe = float(sum(p1[v] * p2[v] for v in values))

            if np.isclose(1.0 - pe, 0.0):
                kappa = 1.0 if np.isclose(po, 1.0) else 0.0
            else:
                kappa = (po - pe) / (1.0 - pe)
            kappas[f"{i}-{j}"] = round(float(kappa), 4)

    return kappas


def fleiss_kappa(ratings_matrix: List[List[int]], num_categories: int) -> float:
    """Calculate Fleiss' kappa for multi-rater agreement."""
    if not ratings_matrix:
        return 0.0

    row_lengths = [len(row) for row in ratings_matrix if row]
    if not row_lengths:
        return 0.0

    n = max(set(row_lengths), key=row_lengths.count)
    if n < 2:
        return 0.0

    complete_rows = [row for row in ratings_matrix if len(row) == n]
    if not complete_rows:
        return 0.0

    observed_categories = sorted({code for row in complete_rows for code in row})
    if not observed_categories:
        return 0.0

    if num_categories <= 0:
        num_categories = len(observed_categories)
    num_categories = max(num_categories, len(observed_categories))

    category_to_index = {cat: idx for idx, cat in enumerate(observed_categories)}
    mat = np.zeros((len(complete_rows), num_categories), dtype=float)
    for i, row in enumerate(complete_rows):
        counts = Counter(row)
        for cat, cnt in counts.items():
            mat[i, category_to_index[cat]] = cnt

    P_i = (np.sum(mat ** 2, axis=1) - n) / (n * (n - 1))
    P_bar = np.mean(P_i)
    p_j = np.sum(mat, axis=0) / (len(complete_rows) * n)
    P_e = np.sum(p_j ** 2)

    if np.isclose(1.0 - P_e, 0.0):
        return 1.0 if np.isclose(P_bar, 1.0) else 0.0
    return round(float((P_bar - P_e) / (1.0 - P_e)), 4)


def krippendorffs_alpha(ratings_matrix: List[List[int]], level: str = "nominal") -> float:
    """Calculate Krippendorff's alpha (nominal or ordinal)."""
    if not ratings_matrix:
        return 0.0

    valid_rows = [row for row in ratings_matrix if len(row) >= 2]
    if not valid_rows:
        return 0.0

    categories = sorted({code for row in ratings_matrix for code in row})
    if len(categories) <= 1:
        return 1.0

    category_to_rank = {cat: idx for idx, cat in enumerate(categories)}

    def distance(a: int, b: int) -> float:
        if level == "nominal":
            return 0.0 if a == b else 1.0
        if level == "ordinal":
            max_rank = max(len(categories) - 1, 1)
            return ((category_to_rank[a] - category_to_rank[b]) / max_rank) ** 2
        raise ValueError("level must be 'nominal' or 'ordinal'")

    observed_distances = []
    for row in valid_rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                observed_distances.append(distance(row[i], row[j]))

    if not observed_distances:
        return 0.0
    D_o = float(np.mean(observed_distances))

    pooled = [code for row in ratings_matrix for code in row]
    total = len(pooled)
    if total < 2:
        return 1.0

    counts = Counter(pooled)
    total_pairs = total * (total - 1) / 2
    D_e_num = 0.0
    for i, cat_a in enumerate(categories):
        for cat_b in categories[i + 1:]:
            D_e_num += counts[cat_a] * counts[cat_b] * distance(cat_a, cat_b)
    D_e = D_e_num / total_pairs if total_pairs else 0.0

    if np.isclose(D_e, 0.0):
        return 1.0 if np.isclose(D_o, 0.0) else 0.0
    return round(float(1.0 - (D_o / D_e)), 4)


def per_category_accuracy(predictions: List[int], ground_truth: List[int]) -> Dict[int, float]:
    """Accuracy broken down by ground truth category."""
    if not predictions or not ground_truth:
        return {}

    by_category: Dict[int, Dict[str, int]] = {}
    for pred, truth in zip(predictions, ground_truth):
        by_category.setdefault(truth, {"correct": 0, "total": 0})
        by_category[truth]["total"] += 1
        if pred == truth:
            by_category[truth]["correct"] += 1

    return {
        cat: round(vals["correct"] / vals["total"], 4)
        for cat, vals in sorted(by_category.items())
        if vals["total"]
    }


def confusion_matrix(predictions: List[int], ground_truth: List[int]) -> Dict[str, Any]:
    """Return confusion matrix and most confused pairs."""
    if not predictions or not ground_truth:
        return {"labels": [], "matrix": {}, "most_confused_pairs": []}

    labels = sorted(set(ground_truth) | set(predictions))
    matrix = {
        truth: {pred: 0 for pred in labels}
        for truth in labels
    }

    for pred, truth in zip(predictions, ground_truth):
        matrix[truth][pred] += 1

    confused_pairs = []
    for truth in labels:
        for pred in labels:
            if pred == truth:
                continue
            count = matrix[truth][pred]
            if count > 0:
                confused_pairs.append({
                    "ground_truth": truth,
                    "predicted": pred,
                    "count": count
                })

    confused_pairs.sort(key=lambda x: x["count"], reverse=True)
    return {
        "labels": labels,
        "matrix": matrix,
        "most_confused_pairs": confused_pairs[:5],
    }


def error_rate_by_chunk(chunk_results: List[Dict], ground_truth: Dict[str, int]) -> List[float]:
    """Calculate error rate per chunk to detect temporal trends."""
    error_rates = []
    for chunk in chunk_results:
        if "results" in chunk and isinstance(chunk["results"], dict):
            chunk_map = chunk["results"]
        else:
            chunk_map = chunk

        preds, truths = [], []
        for text_id, responses in chunk_map.items():
            if text_id not in ground_truth:
                continue
            if not responses:
                continue
            codes = [_extract_code(r) for r in responses]
            preds.append(majority_vote(codes))
            truths.append(ground_truth[text_id])

        if not truths:
            error_rates.append(0.0)
            continue
        error_rates.append(round(1.0 - accuracy(preds, truths), 4))

    return error_rates


def confidence_calibration(predictions: List[int], ground_truth: List[int],
                           confidences: List[float]) -> Dict[str, Any]:
    """Bin predictions by confidence and compute bin accuracy."""
    if not predictions or not confidences:
        return {"bins": []}

    conf = np.array(confidences, dtype=float)
    correct = np.array([int(p == t) for p, t in zip(predictions, ground_truth)], dtype=float)
    edges = np.linspace(0.0, 1.0, 6)
    bins = []

    for i in range(len(edges) - 1):
        lower, upper = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            idx = (conf >= lower) & (conf <= upper)
        else:
            idx = (conf >= lower) & (conf < upper)
        count = int(np.sum(idx))
        if count == 0:
            continue
        bins.append({
            "range": [round(float(lower), 2), round(float(upper), 2)],
            "count": count,
            "avg_confidence": round(float(np.mean(conf[idx])), 4),
            "accuracy": round(float(np.mean(correct[idx])), 4),
        })

    return {"bins": bins}


def accuracy_confidence_correlation(predictions: List[int], ground_truth: List[int],
                                    confidences: List[float]) -> float:
    """Pearson correlation between confidence and correctness."""
    if not predictions or not confidences:
        return 0.0

    correct = np.array([int(p == t) for p, t in zip(predictions, ground_truth)], dtype=float)
    conf = np.array(confidences, dtype=float)
    if np.std(conf) == 0 or np.std(correct) == 0:
        return 0.0
    return round(float(np.corrcoef(conf, correct)[0, 1]), 4)


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
    consensus_confidences = []
    per_agent = {}
    ratings_matrix = []
    
    for text_id, responses in results.items():
        if text_id not in ground_truth:
            continue
        
        truth = ground_truth[text_id]
        codes = [_extract_code(r) for r in responses]
        ratings_matrix.append(codes)

        confidences = []
        confidence_present = False
        for response in responses:
            conf, present = _extract_confidence(response)
            confidences.append(conf)
            confidence_present = confidence_present or present
        
        for i, code in enumerate(codes):
            per_agent.setdefault(i, {"preds": [], "truths": []})
            per_agent[i]["preds"].append(code)
            per_agent[i]["truths"].append(truth)
        
        consensus_preds.append(majority_vote(codes))
        truths.append(truth)
        if confidence_present:
            consensus_confidences.append(float(np.mean(confidences)))
    
    eval_dict = {
        "total": len(truths),
        "correct": sum(p == t for p, t in zip(consensus_preds, truths)),
        "accuracy": round(accuracy(consensus_preds, truths), 4),
        "per_agent_accuracy": {
            i: round(accuracy(d["preds"], d["truths"]), 4) 
            for i, d in per_agent.items()
        }
    }

    num_categories = len(set(truths) | set(consensus_preds))
    eval_dict["irr"] = {
        "cohens_kappa": cohens_kappa_pairwise(ratings_matrix),
        "fleiss_kappa": fleiss_kappa(ratings_matrix, num_categories),
        "krippendorffs_alpha": krippendorffs_alpha(ratings_matrix),
    }
    eval_dict["per_category_accuracy"] = per_category_accuracy(consensus_preds, truths)
    eval_dict["confusion_matrix"] = confusion_matrix(consensus_preds, truths)

    if consensus_confidences and any(not np.isclose(c, 1.0) for c in consensus_confidences):
        eval_dict["confidence_analysis"] = {
            "calibration": confidence_calibration(consensus_preds, truths, consensus_confidences),
            "accuracy_confidence_correlation": accuracy_confidence_correlation(consensus_preds, truths, consensus_confidences),
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
                     chunk_boundaries: List[Tuple[int, int]] = None,
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

        temporal_analysis = {}
        if chunk_boundaries:
            coding_chunks = []
            discussion_chunks = []
            for start, end in chunk_boundaries:
                text_ids = [f"Text-{idx+1}" for idx in range(start, end + 1)]
                coding_chunks.append({
                    tid: coding_results[tid]
                    for tid in text_ids if tid in coding_results
                })
                if discussion_results:
                    discussion_chunks.append({
                        tid: discussion_results[tid]
                        for tid in text_ids if tid in discussion_results
                    })

            temporal_analysis["coding_error_rate_by_chunk"] = error_rate_by_chunk(coding_chunks, self.ground_truth)
            if discussion_chunks:
                temporal_analysis["discussion_error_rate_by_chunk"] = error_rate_by_chunk(discussion_chunks, self.ground_truth)

            if log_fn:
                log_fn(f"   Coding Error Rate by Chunk: {temporal_analysis['coding_error_rate_by_chunk']}")
                if "discussion_error_rate_by_chunk" in temporal_analysis:
                    log_fn(f"   Discussion Error Rate by Chunk: {temporal_analysis['discussion_error_rate_by_chunk']}")

        confidence_analysis = {
            "coding": coding_eval.get("confidence_analysis"),
            "discussion": discussion_eval.get("confidence_analysis") if discussion_eval else None,
        }

        if log_fn:
            log_fn(f"   Coding IRR: {coding_eval['irr']}")
            if discussion_eval:
                log_fn(f"   Discussion IRR: {discussion_eval['irr']}")
            if confidence_analysis["coding"] or confidence_analysis["discussion"]:
                log_fn(f"   Confidence Analysis: {confidence_analysis}")
        
        run_result = {
            "coding": coding_eval,
            "discussion": discussion_eval,
            "temporal_analysis": temporal_analysis if temporal_analysis else None,
            "confidence_analysis": confidence_analysis if (confidence_analysis["coding"] or confidence_analysis["discussion"]) else None,
            "irr_summary": {
                "coding": coding_eval["irr"],
                "discussion": discussion_eval["irr"] if discussion_eval else None,
            }
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
