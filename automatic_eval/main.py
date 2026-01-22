import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any
import argparse
from string import Template


from reference_based import Ev2RReferenceBasedScorer
from proxy_based import Ev2RProxyScorer
from final_score import ClimateCheckEv2RScorer
from claim_verification import ClaimVerificationScorer

import os
from google import genai

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

def remove_only_nei_claims(grouped: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[Dict[str, str]]]:
    """
    Removes claims that only have NEI annotations --> used only to filter out claims in gold data that can't be evaluated properly.
    """
    filtered_grouped = {}
    for claim_id, rows in grouped.items():
        if not all(row["annotation"] == "Not Enough Information" for row in rows):
            filtered_grouped[claim_id] = rows
    return filtered_grouped

def get_abstract_text(unannotated_rows):
    """
    Adds abstract text for unannotated rows by loading the publications corpus and retrieving it based on abstract_id.
    """
    from datasets import load_dataset
    corpus = load_dataset('rabuahmad/climatecheck_publications_corpus')
    corpus_train = corpus['train']

    for row in unannotated_rows:
        abstract_id = row.get("abstract_id")
        abstract = corpus_train[int(abstract_id)]['abstract']
        row['abstract'] = abstract

    return unannotated_rows


def main(
    gold_csv: Path,
    pred_csv: Path,
    reference_scorer: Ev2RReferenceBasedScorer,
    proxy_scorer: Ev2RProxyScorer,
    verification_scorer: ClaimVerificationScorer,
    lambda_proxy: float = 0.5,
):
    """
    Computes Ev2R scores for unannotated claim–abstract pairs.
    """

    gold_rows = read_csv(gold_csv)
    pred_rows = read_csv(pred_csv)

    gold_by_claim = group_by_claim(gold_rows)
    pred_by_claim = group_by_claim(pred_rows)

    gold_by_claim = remove_only_nei_claims(gold_by_claim)

    all_claim_ev2r = []
    all_claim_verification = []

    for claim_id, pred_claim_rows in pred_by_claim.items():
        if claim_id not in gold_by_claim:
            # No gold data → cannot evaluate
            continue

        gold_claim_rows = gold_by_claim[claim_id]

        claim_text = gold_claim_rows[0]["claim"]

        gold_abstracts = [r["abstract"] for r in gold_claim_rows]
        gold_labels = [r["annotation"] for r in gold_claim_rows]
        gold_pairs = {
            (r["claim_id"], r["abstract_original_index"]) for r in gold_claim_rows
        }

        # Keep full rows for verification (need predicted labels)
        unannotated_rows = [
            r for r in pred_claim_rows
            if (r["claim_id"], r["abstract_id"]) not in gold_pairs
        ]

        unannotated_rows = get_abstract_text(unannotated_rows)

        # Check if there are any rows to evaluate automatically
        if not unannotated_rows:
            continue

        orchestrator = ClimateCheckEv2RScorer(
            reference_scorer=reference_scorer,
            proxy_scorer=proxy_scorer,
            gold_labels=gold_labels,
            lambda_proxy=lambda_proxy,
        )

        ev2r_result = orchestrator.score(
            claim=claim_text,
            retrieved_abstracts=[r["abstract"] for r in unannotated_rows],
            gold_abstracts=gold_abstracts,
        )

        all_claim_ev2r.append(ev2r_result["Ev2R"])

        # Automatic claim verification scoring

        per_claim_verification_scores = []

        for row, per_abs in zip(unannotated_rows, ev2r_result["per_abstract"]):
            predicted_label = row.get("label")
            gold_label = per_abs["gold_label"]

            if predicted_label is None:
                continue

            verification = verification_scorer.score(
                claim=claim_text,
                abstract=row["abstract"],
                predicted_label=predicted_label,
                gold_label=gold_label,
            )

            per_claim_verification_scores.append(verification["score"])

        if per_claim_verification_scores:
            claim_verif_score = sum(per_claim_verification_scores) / len(
                per_claim_verification_scores
            )
            all_claim_verification.append(claim_verif_score)

        print(
            f"[Claim {claim_id}] "
            f"Ev2R={ev2r_result['Ev2R']:.4f} | "
            f"AutoVerif={claim_verif_score:.4f} "
            f"(unannotated={len(unannotated_rows)})"
        )

    if not all_claim_ev2r:
        print("No unannotated claim–abstract pairs found.")
        return
    
    print("\n==============================")
    print(f"Evaluated claims: {len(all_claim_ev2r)}")
    print(f"Mean Ev2R score: {sum(all_claim_ev2r) / len(all_claim_ev2r):.4f}")

    if all_claim_verification:
        print(
            f"Mean automatic verification score: "
            f"{sum(all_claim_verification) / len(all_claim_verification):.4f}"
        )
    print("==============================\n")

if __name__ == "__main__":
    # Paths
    GOLD_CSV = Path("gold.csv")
    PRED_CSV = Path("predictions.csv")

    parser = argparse.ArgumentParser()
    parser.add_argument("--gemini_key", required=True, help="API key to access Gemini when running the refenrence-based scorer.")
    parser.add_argument("--gemini_model", required=True, help="Model name to use for prompting in the refenrence-based scorer.")
    args = parser.parse_args()

    gemini_client = genai.Client(api_key=args.gemini_key)

    # Instantiate scorers (placeholders)
    reference_scorer = Ev2RReferenceBasedScorer(
        gemini_client= gemini_client,
        prompt_path=Path("reference_based_prompt.txt"),
        gemini_model=args.gemini_model,
    )

    proxy_scorer = Ev2RProxyScorer(
        model_name_or_path="", # add fine-tuned DeBERTa here
        label2id={
            "SUPPORTS": 0,
            "REFUTES": 1,
            "NEI": 2,
        },
    )

    verification_scorer = ClaimVerificationScorer(
        model_name_or_path="",  # add fine-tuned DeBERTa here
        label2id={
            "SUPPORTS": 0,
            "REFUTES": 1,
            "NEI": 2,
        },
        alpha=0.5,
    )


    main(
        gold_csv=GOLD_CSV,
        pred_csv=PRED_CSV,
        reference_scorer=reference_scorer,
        proxy_scorer=proxy_scorer,
        verification_scorer=verification_scorer,
        lambda_proxy=0.5,
    )