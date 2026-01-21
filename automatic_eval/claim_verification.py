from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ClaimVerificationScorer:
    """
    Automatic claim–abstract verification scorer inspired by the Ev2R proxy component.

    This scorer evaluates the quality of a predicted label for a retrieved abstract
    using two signals:
      1) Model confidence in the predicted label.
      2) Consistency with the label of the best-aligned gold abstract.

    The final score is a weighted combination of both.
    """

    def __init__(
        self,
        model_name_or_path: str,
        label2id: Dict[str, int],
        alpha: float = 0.5,
        device: str | None = None,
    ):
        """
        Parameters
        ----------
        model_name_or_path : str
            HuggingFace model name or local path for a fine-tuned
            DeBERTa claim–abstract classification model.

        label2id : Dict[str, int]
            Mapping from label strings (e.g., "SUPPORTS", "REFUTES", "NEI")
            to model output indices.

        alpha : float, default=0.5
            Weight of the gold-consistency component.
            alpha=0.0 → confidence-only
            alpha=1.0 → strict agreement with gold evidence

        device : str, optional
            Torch device ("cuda" or "cpu"). Auto-detected if None.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path
        )

        self.label2id = label2id
        self.alpha = alpha
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(
        self,
        claim: str,
        abstract: str,
        predicted_label: str,
        gold_label: str | None = None,
    ) -> Dict[str, float]:
        """
        Computes the automatic claim verification score for a single
        claim–abstract pair.

        Parameters
        ----------
        claim : str
            The claim text.

        abstract : str
            The retrieved abstract text.

        predicted_label : str
            Label predicted by the system for this claim–abstract pair.

        gold_label : str, optional
            Label of the best-aligned gold abstract.
            If None, gold-consistency is skipped.

        Returns
        -------
        Dict[str, float]
            {
              "confidence": S_conf,
              "consistency": S_cons (or None),
              "score": S_auto
            }
        """
        pred_label_id = self.label2id[predicted_label]

        inputs = self.tokenizer(
            claim,
            abstract,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

        # (a) Confidence in predicted label
        s_conf = probs[0, pred_label_id].item()

        # (b) Consistency with gold evidence
        if gold_label is not None:
            s_cons = float(predicted_label == gold_label)
            s_auto = (1 - self.alpha) * s_conf + self.alpha * s_cons
        else:
            s_cons = None
            s_auto = s_conf

        return {
            "confidence": s_conf,
            "consistency": s_cons,
            "score": s_auto,
        }
