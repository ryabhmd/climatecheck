import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any
import argparse
from string import Template
from tqdm import tqdm
from reference_based import Ev2RReferenceBasedScorer
from proxy_based import Ev2RProxyScorer
from final_score import ClimateCheckEv2RScorer
from claim_verification import ClaimVerificationScorer
from cache import Ev2RCache, make_pair_key

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
    submission_name: str,
    reference_scorer: Ev2RReferenceBasedScorer,
    proxy_scorer: Ev2RProxyScorer,
    verification_scorer: ClaimVerificationScorer,
    lambda_proxy: float = 0.5,
    output_dir: Path = Path("results"),

):
    """
    Computes Ev2R scores for unannotated claim–abstract pairs.
    """

    gold_rows = read_csv(gold_csv)
    pred_rows = read_csv(pred_csv)

    gold_by_claim = group_by_claim(gold_rows)
    pred_by_claim = group_by_claim(pred_rows)

    gold_by_claim = remove_only_nei_claims(gold_by_claim)

    # create submission results dir
    submission_dir = output_dir / f"submission_{submission_name}"
    submission_dir.mkdir(parents=True, exist_ok=True)

    # prep for saving results incrementally
    claim_file = submission_dir / "claim_scores.csv"
    abstract_file = submission_dir / "abstract_scores.csv"
    summary_file = submission_dir / "summary.csv"

    # Initialize cache
    cache = Ev2RCache("results/cache/ev2r_cache.db")

    all_claim_ev2r = []
    all_claim_verification = []

    claim_results = []
    abstract_results = []

    for claim_id, pred_claim_rows in tqdm(
            pred_by_claim.items(),
            total=len(pred_by_claim),
            desc="Scoring claims",
            ):
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
            cache=cache,
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

        # Automatic claim verification scoring (per abstract loop)

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

            # save claim-abstract pair scores
            abstract_results.append({
                "claim_id": claim_id,
                "abstract_id": row.get("abstract_id"),
                "gold_label": gold_label,
                "proxy_score": per_abs.get("proxy_score"),
                "reference_score": per_abs.get("reference_score"),
                "Ev2R_component": per_abs.get("Ev2R_component"),
                "predicted_label": predicted_label,
                "verification_score": verification["score"],
            })

            # write claim-abstract scores
            if abstract_results:
                with open(abstract_file, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=abstract_results[0].keys())
                    writer.writeheader()
                    writer.writerows(abstract_results)

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

        claim_results.append({
            "claim_id": claim_id,
            "Ev2R": ev2r_result["Ev2R"],
            "AutoVerification": claim_verif_score,
            "num_unannotated": len(unannotated_rows),
            })
        
        tqdm.write(
        f"[Claim {claim_id}] "
        f"Ev2R={ev2r_result['Ev2R']:.4f} | "
        f"AutoVerif={claim_verif_score if claim_verif_score is not None else 'N/A'} "
        f"(unannotated={len(unannotated_rows)})"
    )

        # Save per-claim results
        with open(claim_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=claim_results[0].keys())
            writer.writeheader()
            writer.writerows(claim_results)

    # Save summary for all claims (per full submission)
    summary = {
            "num_claims": len(all_claim_ev2r),
            "mean_Ev2R": sum(all_claim_ev2r) / len(all_claim_ev2r),
            "mean_auto_verification": (
                sum(all_claim_verification) / len(all_claim_verification)
                if all_claim_verification else None
                ),
            }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_path", required=True,
                        help="Path to gold data.")
    parser.add_argument("--pred_path", required=True,
                        help="Path to predictions data.")
    parser.add_argument("--gemini_key", required=True, 
                        help="API key to access Gemini when running the refenrence-based scorer.")
    parser.add_argument("--gemini_model", default="gemini-2.5-pro", 
                        help="Model name to use for prompting in the refenrence-based scorer.")
    parser.add_argument("--proxy_scorer_model", default="rausch/deberta-climatecheck-2463191-step26000", 
                        help="Model name (BERT-based) to use for the proxy scorer and the claim verification socrer.")
    parser.add_argument("--output_dir", required=True,
                        help="Output dir to save results.")
    parser.add_argument("--submission_name", required=True,
                        help="Name of submission team/user to save results to.")

    args = parser.parse_args() 

    GOLD_CSV = Path(args.gold_path)
    PRED_CSV = Path(args.pred_path)

    gemini_client = genai.Client(api_key=args.gemini_key)

    # Instantiate scorers (placeholders)
    reference_scorer = Ev2RReferenceBasedScorer(
        gemini_client= gemini_client,
        prompt_path=Path("reference_based_prompt.txt"),
        gemini_model=args.gemini_model,
    )

    proxy_scorer = Ev2RProxyScorer(
        model_name_or_path=args.proxy_scorer_model,
        label2id={
            "Supports": 0,
            "Refutes": 1,
            "Not Enough Information": 2,
        },
    )

    verification_scorer = ClaimVerificationScorer(
        model_name_or_path=args.proxy_scorer_model,
        label2id={
            "Supports": 0,
            "Refutes": 1,
            "Not Enough Information": 2,
        },
        alpha=0.5,
    )

    main(
        gold_csv=GOLD_CSV,
        pred_csv=PRED_CSV,
        submission_name=args.submission_name,
        reference_scorer=reference_scorer,
        proxy_scorer=proxy_scorer,
        verification_scorer=verification_scorer,
        lambda_proxy=0.5,
        output_dir = Path(args.output_dir),
    )
