from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, List

from utils.types import AgreementResult, CodebookUpdate, CodingResponse, IRRMetrics


class JudgeAgent:
    """A judge that evaluates coder agreement and tracks reliability metrics."""

    def __init__(self):
        self._all_ratings: List[List[int]] = []

    def check_agreement(
        self,
        agent_responses: List[CodingResponse],
        confidence_weighted: bool = False,
        confidence_threshold: float = 0.7,
    ) -> bool:
        """Return a binary agreement verdict."""
        return self.check_agreement_detailed(
            agent_responses,
            confidence_weighted=confidence_weighted,
            confidence_threshold=confidence_threshold,
        ).agreed

    def check_agreement_detailed(
        self,
        agent_responses: List[CodingResponse],
        confidence_weighted: bool = False,
        confidence_threshold: float = 0.7,
    ) -> AgreementResult:
        """Return detailed agreement information."""
        if len(agent_responses) == 0:
            return AgreementResult(
                agreed=False,
                codes=[],
                confidence_weighted=confidence_weighted,
                disagreement_categories=None,
            )

        codes = [response.code for response in agent_responses]

        if len(agent_responses) == 1 or len(set(codes)) == 1:
            return AgreementResult(
                agreed=True,
                codes=codes,
                confidence_weighted=confidence_weighted,
                disagreement_categories=None,
            )

        # Baseline behavior remains pure code equality when weighted agreement is off.
        if not confidence_weighted:
            return AgreementResult(
                agreed=False,
                codes=codes,
                confidence_weighted=False,
                disagreement_categories=sorted(set(codes)),
            )

        high_conf_codes = [
            response.code
            for response in agent_responses
            if response.confidence >= confidence_threshold
        ]

        # Require at least two high-confidence coders to establish consensus.
        high_conf_consensus = len(high_conf_codes) >= 2 and len(set(high_conf_codes)) == 1

        return AgreementResult(
            agreed=high_conf_consensus,
            codes=codes,
            confidence_weighted=True,
            disagreement_categories=None if high_conf_consensus else sorted(set(codes)),
        )

    def record_ratings(self, codes: List[int]):
        """Record one item's ratings for later IRR calculation."""
        if codes:
            self._all_ratings.append(codes)

    def calculate_irr(self) -> IRRMetrics:
        """Calculate Cohen's kappa (pairwise), Krippendorff's alpha, and Fleiss' kappa."""
        if not self._all_ratings:
            return IRRMetrics()

        cohens = self._calculate_cohens_kappa_pairwise()
        alpha = self._calculate_krippendorffs_alpha_nominal()
        fleiss = self._calculate_fleiss_kappa()

        return IRRMetrics(
            cohens_kappa=cohens if cohens else None,
            krippendorffs_alpha=alpha,
            fleiss_kappa=fleiss,
        )

    def get_disagreement_patterns(self) -> Dict[str, Any]:
        """Return confusion-style disagreement frequencies."""
        confusion: Dict[tuple[int, int], int] = defaultdict(int)

        for codes in self._all_ratings:
            for idx_a, idx_b in combinations(range(len(codes)), 2):
                code_a, code_b = codes[idx_a], codes[idx_b]
                if code_a != code_b:
                    pair = tuple(sorted((code_a, code_b)))
                    confusion[pair] += 1

        most_common = sorted(
            ((i, j, count) for (i, j), count in confusion.items()),
            key=lambda item: item[2],
            reverse=True,
        )

        return {"confusion": dict(confusion), "most_common": most_common}

    def check_codebook_agreement(self, agent_responses: List[CodebookUpdate]) -> bool:
        """Checks if all agents agree with the mediated codebook."""
        if len(agent_responses) == 0:
            return False

        return all(not response.need_update for response in agent_responses)

    def _calculate_cohens_kappa_pairwise(self) -> Dict[str, float]:
        valid_rows = [codes for codes in self._all_ratings if len(codes) >= 2]
        if not valid_rows:
            return {}

        max_raters = max(len(codes) for codes in valid_rows)
        kappas: Dict[str, float] = {}

        for i, j in combinations(range(max_raters), 2):
            paired = [codes for codes in valid_rows if len(codes) > j]
            if not paired:
                continue

            r1 = [codes[i] for codes in paired]
            r2 = [codes[j] for codes in paired]
            total = len(r1)
            if total == 0:
                continue

            observed = sum(1 for a, b in zip(r1, r2) if a == b) / total
            dist1 = Counter(r1)
            dist2 = Counter(r2)
            categories = set(dist1) | set(dist2)
            expected = sum((dist1[c] / total) * (dist2[c] / total) for c in categories)

            denominator = 1 - expected
            if denominator == 0:
                kappa = 1.0 if observed == 1 else 0.0
            else:
                kappa = (observed - expected) / denominator

            kappas[f"rater_{i + 1}_vs_rater_{j + 1}"] = kappa

        return kappas

    def _calculate_fleiss_kappa(self) -> float | None:
        if not self._all_ratings:
            return None

        # Fleiss assumes constant number of raters per item.
        target_n = Counter(len(codes) for codes in self._all_ratings if len(codes) >= 2)
        if not target_n:
            return None
        n = target_n.most_common(1)[0][0]
        rows = [codes for codes in self._all_ratings if len(codes) == n]
        if len(rows) < 2:
            return None

        categories = sorted({code for row in rows for code in row})
        if len(categories) <= 1:
            return 1.0

        N = len(rows)
        p_j_counts = Counter()
        P_i_values: List[float] = []

        for row in rows:
            counts = Counter(row)
            for c in categories:
                p_j_counts[c] += counts[c]

            numerator = sum(counts[c] * (counts[c] - 1) for c in categories)
            P_i_values.append(numerator / (n * (n - 1)))

        P_bar = sum(P_i_values) / N
        p_j = {c: p_j_counts[c] / (N * n) for c in categories}
        P_e_bar = sum(v * v for v in p_j.values())

        denominator = 1 - P_e_bar
        if denominator == 0:
            return 1.0 if P_bar == 1 else 0.0

        return (P_bar - P_e_bar) / denominator

    def _calculate_krippendorffs_alpha_nominal(self) -> float | None:
        if not self._all_ratings:
            return None

        rows = [codes for codes in self._all_ratings if len(codes) >= 2]
        if not rows:
            return None

        total_disagreement = 0.0
        total_pairs = 0
        pooled = Counter()

        for row in rows:
            counts = Counter(row)
            n_i = len(row)
            pooled.update(row)

            same_pairs = sum(count * (count - 1) for count in counts.values())
            all_pairs = n_i * (n_i - 1)
            disagree_pairs = all_pairs - same_pairs

            total_disagreement += disagree_pairs
            total_pairs += all_pairs

        if total_pairs == 0:
            return None

        D_o = total_disagreement / total_pairs

        n_total = sum(pooled.values())
        if n_total <= 1:
            return None

        expected_same_pairs = sum(count * (count - 1) for count in pooled.values())
        expected_all_pairs = n_total * (n_total - 1)
        D_e = (expected_all_pairs - expected_same_pairs) / expected_all_pairs

        if D_e == 0:
            return 1.0 if D_o == 0 else 0.0

        return 1 - (D_o / D_e)
