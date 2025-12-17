from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Ev2RProxyScorer:
    """
    Proxy-based Ev2R scorer as defined in:

    Akhtar et al. (2025), "Ev2R: Evaluating Evidence Retrieval in Automated Fact-Checking".

    This scorer evaluates retrieved evidence by measuring how strongly
    a trained verdict model predicts the reference (gold) label for a claim
    conditioned on the retrieved evidence.

    The score corresponds to:
        S_proxy = p_theta(y* | claim, retrieved_evidence)
    """

    def __init__(
        self,
        model_name_or_path: str,
        label2id: Dict[str, int],
        device: str | None = None,
    ):
        """
        Parameters
        ----------
        model_name_or_path : str
            HuggingFace model name or local path for a fine-tuned DeBERTa
            sequence classification model.

        label2id : Dict[str, int]
            Mapping from label strings (e.g. "SUPPORTS", "REFUTES", "NEI")
            to model class indices.

        device : str, optional
            Torch device ("cuda", "cpu"). If None, auto-detected.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path
        )

        self.label2id = label2id
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(
        self,
        claim: str,
        retrieved_abstracts: List[str],
        gold_label: str,
        aggregation: str = "max",
    ) -> Dict[str, Any]:
        """
        Computes the proxy-based Ev2R score for a set of retrieved abstracts.

        Parameters
        ----------
        claim : str
            The claim being evaluated.

        retrieved_abstracts : List[str]
            Retrieved evidence abstracts.

        gold_label : str
            Gold verdict label associated with the best-aligned gold abstract.

        aggregation : str, default="max"
            How to aggregate scores over multiple abstracts.
            Supported values:
                - "max"  : max probability
                - "mean" : mean probability

        Returns
        -------
        Dict[str, Any]
            {
              "S_proxy": float,
              "per_abstract_scores": List[float]
            }
        """
        gold_label_id = self.label2id[gold_label]

        per_abstract_scores = []

        for abstract in retrieved_abstracts:
            prob = self._score_single(claim, abstract, gold_label_id)
            per_abstract_scores.append(prob)

        if not per_abstract_scores:
            s_proxy = 0.0
        elif aggregation == "max":
            s_proxy = max(per_abstract_scores)
        elif aggregation == "mean":
            s_proxy = sum(per_abstract_scores) / len(per_abstract_scores)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        return {
            "S_proxy": s_proxy,
            "per_abstract_scores": per_abstract_scores,
        }

    def _score_single(
        self,
        claim: str,
        abstract: str,
        gold_label_id: int,
    ) -> float:
        """
        Computes p_theta(y* | claim, abstract) for a single abstract.

        Returns
        -------
        float
            Probability assigned to the gold label.
        """
        inputs = self.tokenizer(
            claim,
            abstract,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

        return probs[0, gold_label_id].item()
