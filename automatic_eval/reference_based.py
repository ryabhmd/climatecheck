from pathlib import Path
from typing import List, Dict, Any
import json


class Ev2RReferenceBasedScorer:
    """
    Reference-based Ev2R scorer as defined in:

    Akhtar et al. (2025), "Ev2R: Evaluating Evidence Retrieval in Automated Fact-Checking".

    This implementation is designed for the ClimateCheck shared task
    and is only used when a retrieved claim–abstract pair is not part
    of the gold annotated evidence set.

    The scorer:
    - decomposes retrieved and gold abstracts into atomic facts
    - evaluates factual support using Gemini-Pro
    - computes S_precision, S_recall, and S_F1 exactly as in the paper
    - treats multiple gold abstracts as alternative (disjunctive) references and returns the max score over gold abstracts
    """

    def __init__(
        self,
        gemini_client: Any,
        prompt_path: Path,
    ):
        """
        Parameters
        ----------
        gemini_client : Any
            Gemini-Pro client or wrapper with a method:
                generate(prompt: str) -> str

        prompt_path : Path
            Path to a .txt file containing the prompt that instructs Gemini to:
            - extract atomic facts from predicted and reference evidence
            - check factual support in both directions
            - output a JSON object in the expected schema
        """
        self.gemini_client = gemini_client
        self.prompt_template = prompt_path.read_text()

    def score(
        self,
        claim: str,
        retrieved_abstract: str,
        gold_abstracts: List[str],
    ) -> Dict[str, Any]:
        """
        Computes the reference-based Ev2R score for a retrieved abstract
        against multiple gold abstracts.

        For each gold abstract:
            - Gemini-Pro produces atomic facts and support counts
            - S_precision, S_recall, and S_F1 are computed

        The final score is the MAX S_F1 over all gold abstracts.

        Parameters
        ----------
        claim : str
            The claim being evaluated.

        retrieved_abstract : str
            The retrieved abstract (predicted evidence).

        gold_abstracts : List[str]
            All gold reference abstracts associated with the claim.

        Returns
        -------
        Dict[str, Any]
            {
              "S_precision": float,
              "S_recall": float,
              "S_F1": float,
              "best_gold_index": int,
              "per_gold_scores": List[Dict[str, float]]
            }
        """
        per_gold_scores = []

        for idx, gold_abstract in enumerate(gold_abstracts):
            gemini_output = self._run_gemini(
                claim, retrieved_abstract, gold_abstract
            )

            s_prec = self._compute_precision(gemini_output)
            s_rec = self._compute_recall(gemini_output)
            s_f1 = self._compute_f1(s_prec, s_rec)

            per_gold_scores.append(
                {
                    "gold_index": idx,
                    "S_precision": s_prec,
                    "S_recall": s_rec,
                    "S_F1": s_f1,
                }
            )

        best = max(per_gold_scores, key=lambda x: x["S_F1"])

        return {
            "S_precision": best["S_precision"],
            "S_recall": best["S_recall"],
            "S_F1": best["S_F1"],
            "best_gold_index": best["gold_index"],
            "per_gold_scores": per_gold_scores,
        }

    def _run_gemini(
        self,
        claim: str,
        retrieved_abstract: str,
        gold_abstract: str,
    ) -> Dict[str, Any]:
        """
        Runs Gemini-Pro with the atomic-fact alignment prompt and parses JSON.

        Expected Gemini output fields (REQUIRED):
        - facts count predicted evidence
        - support predicted evidence
        - facts count reference evidence
        - support reference evidence

        Returns
        -------
        Dict[str, Any]
            Parsed Gemini JSON output.
        """
        prompt = self.prompt_template.format(
            claim=claim,
            predicted_evidence=retrieved_abstract,
            reference_evidence=gold_abstract,
        )

        raw_output = self.gemini_client.generate(prompt)

        try:
            return json.loads(raw_output)
        except json.JSONDecodeError as e:
            raise ValueError(
                "Gemini output is not valid JSON."
            ) from e

    def _compute_precision(self, output: Dict[str, Any]) -> float:
        """
        Computes S_precision as defined in Ev2R:

            S_precision =
                support_predicted_evidence / facts_count_predicted_evidence

        Returns
        -------
        float
            Precision score in [0, 1].
        """
        total = output.get("facts count predicted evidence", 0)
        supported = output.get("support predicted evidence", 0)

        if total == 0:
            return 0.0

        return supported / total

    def _compute_recall(self, output: Dict[str, Any]) -> float:
        """
        Computes S_recall as defined in Ev2R:

            S_recall =
                support_reference_evidence / facts_count_reference_evidence

        Returns
        -------
        float
            Recall score in [0, 1].
        """
        total = output.get("facts count reference evidence", 0)
        supported = output.get("support reference evidence", 0)

        if total == 0:
            return 0.0

        return supported / total

    def _compute_f1(self, precision: float, recall: float) -> float:
        """
        Computes S_F1, the harmonic mean of S_precision and S_recall.

        Returns
        -------
        float
            F1 score in [0, 1].
        """
        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)
