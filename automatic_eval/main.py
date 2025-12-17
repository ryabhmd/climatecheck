import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

from reference_based import Ev2RReferenceBasedScorer
from proxy_based import Ev2RProxyScorer
from final_score import ClimateCheckEv2RScorer

def read_csv(path: Path) -> List[Dict[str, str]]:
    """
    Reads a CSV file into a list of dicts.
    """
    with path.open() as f:
        reader = csv.DictReader(f)
        return list(reader)


def group_by_claim(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """
    Groups rows by claim_id.
    """
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["claim_id"]].append(row)
    return grouped

def main(
    gold_csv: Path,
    pred_csv: Path,
    reference_scorer: Ev2RReferenceBasedScorer,
    proxy_scorer: Ev2RProxyScorer,
    lambda_proxy: float = 0.5,
):
    """
    Computes Ev2R scores for unannotated claim–abstract pairs.
    """

    gold_rows = read_csv(gold_csv)
    pred_rows = read_csv(pred_csv)

    gold_by_claim = group_by_claim(gold_rows)
    pred_by_claim = group_by_claim(pred_rows)

    all_claim_scores = []

    for claim_id, pred_claim_rows in pred_by_claim.items():
        if claim_id not in gold_by_claim:
            # No gold data → cannot evaluate
            continue

        gold_claim_rows = gold_by_claim[claim_id]

        claim_text = gold_claim_rows[0]["claim"]

        gold_abstracts = [r["abstract"] for r in gold_claim_rows]
        gold_labels = [r["label"] for r in gold_claim_rows]
        gold_pairs = {
            (r["claim_id"], r["abstract_id"]) for r in gold_claim_rows
        }

        # Split predicted abstracts
        unannotated_abstracts = [
            r["abstract"]
            for r in pred_claim_rows
            if (r["claim_id"], r["abstract_id"]) not in gold_pairs
        ]

        if not unannotated_abstracts:
            continue

        orchestrator = ClimateCheckEv2RScorer(
            reference_scorer=reference_scorer,
            proxy_scorer=proxy_scorer,
            gold_labels=gold_labels,
            lambda_proxy=lambda_proxy,
        )

        score = orchestrator.score(
            claim=claim_text,
            retrieved_abstracts=unannotated_abstracts,
            gold_abstracts=gold_abstracts,
        )

        all_claim_scores.append(
            {
                "claim_id": claim_id,
                "Ev2R": score["Ev2R"],
                "num_unannotated": len(unannotated_abstracts),
            }
        )

        print(
            f"[Claim {claim_id}] "
            f"Ev2R={score['Ev2R']:.4f} "
            f"(unannotated={len(unannotated_abstracts)})"
        )

    if not all_claim_scores:
        print("No unannotated claim–abstract pairs found.")
        return

    mean_ev2r = sum(x["Ev2R"] for x in all_claim_scores) / len(all_claim_scores)

    print(f"Evaluated claims: {len(all_claim_scores)}")
    print(f"Mean Ev2R score: {mean_ev2r:.4f}")


if __name__ == "__main__":
    # Paths
    GOLD_CSV = Path("gold.csv")
    PRED_CSV = Path("predictions.csv")

    # Instantiate scorers (placeholders)
    reference_scorer = Ev2RReferenceBasedScorer(
        gemini_client=...,                 # add Gemini client here
        prompt_path=Path("prompt.txt"),
    )

    proxy_scorer = Ev2RProxyScorer(
        model_name_or_path="", # add fine-tuned DeBERTa here
        label2id={
            "SUPPORTS": 0,
            "REFUTES": 1,
            "NEI": 2,
        },
    )

    main(
        gold_csv=GOLD_CSV,
        pred_csv=PRED_CSV,
        reference_scorer=reference_scorer,
        proxy_scorer=proxy_scorer,
        lambda_proxy=0.5,
    )