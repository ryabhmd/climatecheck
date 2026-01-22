from typing import Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Ev2RProxyScorer:
    """
    Evidence-relative proxy-based Ev2R scorer adapted for ClimateCheck. 

    Original development of the scorer taken from:
    Akhtar et al. (2025), "Ev2R: Evaluating Evidence Retrieval in Automated Fact-Checking".

    This scorer computes:
        S_proxy(r) = softmax(z(c, r))_{y_g}

    where:
    - r is a retrieved abstract
    - c is the claim
    - y_g is the label of the gold abstract that r aligns with best
      (determined externally via the reference-based component)

    IMPORTANT:
    - This class does not select the gold label (it's based on the reference_based component).
    - It assumes the correct label for r is provided by the orchestrator.
    """

    def __init__(
        self,
        label2id: Dict[str, int],
        device: str | None = None,
        model_name_or_path: str = "rausch/deberta-climatecheck-2463191-step26000",
    ):
        """
        Parameters
        ----------
        model_name_or_path : str
            HuggingFace model name or local path for a fine-tuned
            DeBERTa sequence classification model.

        label2id : Dict[str, int]
            Mapping from label strings (e.g. "SUPPORTS", "REFUTES", "NEI")
            to model output indices.

        device : str, optional
            Torch device ("cuda" or "cpu"). Auto-detected if None.
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
        retrieved_abstract: str,
        gold_label: str,
    ) -> float:
        """
        Computes the proxy score for a single retrieved abstract.

        Parameters
        ----------
        claim : str
            The claim being evaluated.

        retrieved_abstract : str
            A single retrieved abstract.

        gold_label : str
            Label of the gold abstract that this retrieved abstract
            aligns with best.

        Returns
        -------
        float
            S_proxy score in [0, 1].
        """
        gold_label_id = self.label2id[gold_label]

        inputs = self.tokenizer(
            claim,
            retrieved_abstract,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

        return probs[0, gold_label_id].item()
