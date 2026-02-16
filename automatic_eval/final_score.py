from typing import List, Dict, Any
from reference_based import Ev2RReferenceBasedScorer
from proxy_based import Ev2RProxyScorer


class ClimateCheckEv2RScorer:
    """
    Full Ev2R scorer adapted for the ClimateCheck shared task.

    Original development of the scorer taken from:
    Akhtar et al. (2025), "Ev2R: Evaluating Evidence Retrieval in Automated Fact-Checking".

    This implementation:
    - does not assume a claim-level gold verdict (rather, the gold verdict is evidence-dependent)
    - treats gold abstracts as alternative evidence paths
    - computes Ev2R per retrieved abstract
    - aggregates scores across retrieved abstracts
    """

    def __init__(
        self,
        reference_scorer: Ev2RReferenceBasedScorer,
        proxy_scorer: Ev2RProxyScorer,
        gold_labels: List[str],
        lambda_proxy: float = 0.5,
    ):
        """
        Parameters
        ----------
        reference_scorer : Ev2RReferenceBasedScorer
            Reference-based Ev2R scorer.

        proxy_scorer : Ev2RProxyScorer
            Evidence-relative proxy-based Ev2R scorer.

        gold_labels : List[str]
            Labels corresponding to gold abstracts (same order).

        lambda_proxy : float, default=0.5
            Weight assigned to the proxy component.
        """
        self.reference_scorer = reference_scorer
        self.proxy_scorer = proxy_scorer
        self.gold_labels = gold_labels
        self.lambda_proxy = lambda_proxy

    def score(
        self,
        claim: str,
        retrieved_abstracts: List[str],
        gold_abstracts: List[str],
        aggregation: str = "max",
    ) -> Dict[str, Any]:
        """
        Computes the final ClimateCheck-adapted Ev2R score for a claim.

        Parameters
        ----------
        claim : str
            The claim being evaluated.

        retrieved_abstracts : List[str]
            All abstracts retrieved for the claim.

        gold_abstracts : List[str]
            Gold abstracts associated with the claim.

        aggregation : str, default="max"
            Aggregation method over retrieved abstracts.
            Supported values:
                - "max"
                - "mean"

        Returns
        -------
        Dict[str, Any]
            {
              "Ev2R": float,
              "per_abstract": List[Dict[str, Any]]
            }
        """
        per_abstract_scores = []

        for r in retrieved_abstracts:
            # 1. Reference-based scoring
            ref = self.reference_scorer.score(
                claim=claim,
                retrieved_abstract=r,
                gold_abstracts=gold_abstracts,
            )

            best_gold_idx = ref["best_gold_index"]
            gold_label = self.gold_labels[best_gold_idx]

            # 2. Proxy-based scoring (evidence-relative)
            s_proxy = self.proxy_scorer.score(
                claim=claim,
                retrieved_abstract=r,
                gold_label=gold_label,
            )

            # 3. Combine scores
            s_ev2r = (
                (1 - self.lambda_proxy) * ref["S_F1"]
                + self.lambda_proxy * s_proxy
            )

            per_abstract_scores.append(
                {
                    "S_ref": ref["S_F1"],
                    "S_proxy": s_proxy,
                    "Ev2R": s_ev2r,
                    "gold_label": gold_label,
                }
            )

        if not per_abstract_scores:
            final_score = 0.0
        elif aggregation == "max":
            final_score = max(x["Ev2R"] for x in per_abstract_scores)
        elif aggregation == "mean":
            final_score = sum(x["Ev2R"] for x in per_abstract_scores) / len(per_abstract_scores)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        return {
            "Ev2R": final_score,
            "per_abstract": per_abstract_scores,
        }
