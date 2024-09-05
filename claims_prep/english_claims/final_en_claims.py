import pandas as pd
import argparse

"""
Script that create the final file of English claims from
1. ClimaConvo
2. MultiFC (rephrased)
3. ClimateFeedback (rephrased)
4. ClimateFEVER (rephrased)
4. DEBAGREEMENT

Final dataset contains the columns: claimID, claim, source, generation_method, original_claim
"""

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--climaconvo_path", type=str, help="Path for ClimaConvo data")
    parser.add_argument("--multifc_feedback_path", type=str, help="Path for MultiFC + ClimateFeedback data")
    parser.add_argument("--climatefever_path", type=str, help="Path for ClimateFEVER data")
    parser.add_argument("--debagreement_path", type=str, help="Path for DEBAGREEMENT data")

    args = parser.parse_args()

    # Process ClimaConvo
    climaconvo = pd.read_csv(args.climaconvo_path)
    climaconvo = climaconvo[(climaconvo['manual_label'] == 1) | (climaconvo['manual_label'] == 2)]
    climaconvo = climaconvo.drop_duplicates(subset=['Tweet'])

    climaconvo_claims = list(climaconvo['Tweet'].values)
    climaconvo_ids = list(climaconvo['Tweet'].values) # For now the text is supposed to be the ID since it's unique
    climaconvo_sources = ['ClimaConvo'] * len(climaconvo_claims)
    climaconvo_gen_method = ['original'] * len(climaconvo_claims)
    climaconvo_original_claims = [None] * len(climaconvo_claims)

    # Process DEBAGREEMENT
    debagreement = pd.read_csv(args.debagreement_path)
    debagreement_ids = list(debagreement['id'].values)
    debagreement_claims = list(debagreement['body'].values)
    debagreement_sources = ['DEBAGREEMENT'] * len(debagreement_claims)
    debagreement_method = ['original'] * len(debagreement_claims)
    debagreement_original_claims = [None] * len(debagreement_claims)

    # Read MultiFC+ClimateFeedback and ClimateFEVER (after running style_transfer_eval)
    multifc_feedback = pd.read_csv(args.multifc_feedback_path)
    climate_fever = pd.read_csv(args.climatefever_path)

    multifc_feedback_claims = list(multifc_feedback_claims['final_gemini_claim'].values)
    multifc_feedback_ids = list(multifc_feedback['claimID'].values)
    multifc_feedback_sources = list(multifc_feedback['source'].values)
    multifc_feedback_method = ['synthetic'] * len(multifc_feedback_claims)
    multifc_feedback_original_claims = list(multifc_feedback_claims['claim'].values)

    climate_fever_claims = list(climate_fever_rephrasings['final_gemini_claim'].values)
    climate_fever_claim_ids = list(climate_fever_rephrasings['claim_id'].values)
    climate_fever_sources = ['ClimateFEVER'] * len(climate_fever_rephrasings)
    climate_fever_method = ['synthetic'] * len(climate_fever_claims)
    climate_fever_original_claims = list(climate_fever_claims['claim'].values)

    final_english_claims = pd.DataFrame(
        {
            'claimID': climaconvo_ids+debagreement_ids+multifc_feedback_ids+climate_fever_claim_ids,
            'claim': climaconvo_claims+debagreement_claims+multifc_feedback_claims+climate_fever_claims,
            'source': climaconvo_sources+debagreement_sources+multifc_feedback_sources+climate_fever_sources,
            'generation_method': climaconvo_gen_method+debagreement_method+multifc_feedback_method+climate_fever_method,
            'original_claim': climaconvo_original_claims+debagreement_original_claims+multifc_feedback_original_claims+climate_fever_original_claims
            }
        )

    final_english_claims.to_pickle('final_english_claims.pkl')

if __name__ == "__main__":
    main()