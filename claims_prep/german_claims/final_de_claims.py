import pandas as pd
import argparse

"""
Script that create the final file of English claims from
1. GerCCT
2. r/Klimawandel
3. Klimafakten (rephrased)
4. Correctiv Faktencheck (rephrased)

Final dataset contains the columns: claimID, claim, source, generation_method, original_claim
"""

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gercct_path", type=str, help="Path for GerCCT data")
    parser.add_argument("--klimawandel_path", type=str, help="Path for Klimawandel data")
    parser.add_argument("--klimafakten_path", type=str, help="Path for Klimafakten data")
    parser.add_argument("--correctiv_path", type=str, help="Path for Correctiv data")

    args = parser.parse_args()

    # Process GerCCT
    gercct = pd.read_csv(args.gercct_path)
    gercct_claimIDs = gercct['claimID'].values.tolist()
    gercct_claim_text = gercct['german_tweets'].values.tolist()
    gercct_sources = ['GerCCT'] * len(gercct)
    gercct_origin_claims = ['original'] * len(gercct)
    gercct_original_claims = [None] * len(gercct)

    gercct_final_claims = pd.DataFrame(
        {
        'claimID': gercct_claimIDs,
        'claim': gercct_claim_text,
        'source': gercct_sources,
        'generation_method': gercct_origin_claims,
        'original_claim': gercct_original_claims
        })

    # Process Klimawandel
    klimawandel = pd.read_pickle(args.klimawandel_path)
    klimawandel_claims = klimawandel.drop(columns=['type', 'author', 'date', 'claim_detection'])
    # keep only rows where the value in 'text' <= 700
    klimawandel_claims = klimawandel_claims[klimawandel_claims['text'].str.len() <= 700]
    klimawandel_claims = klimawandel_claims.reset_index(drop=True)
    klimawandel_claimIDs = klimawandel_claims['id'].values.tolist()
    klimawandel_claim_texts = klimawandel_claims['text'].values.tolist()
    klimawandel_sources = ['Klimawandel-Subreddit'] * len(klimawandel_claims)
    klimawandel_origin_claims = ['original'] * len(klimawandel_claims)
    klimawandel_original_claims = [None] * len(klimawandel_claims)

    klimawandel_final_claims = pd.DataFrame(
    {
        'claimID': klimawandel_claimIDs,
        'claim': klimawandel_claim_texts,
        'source': klimawandel_sources,
        'generation_method': klimawandel_origin_claims,
        'original_claim': klimawandel_original_claims
        })

    # Process Klimafakten
    klimafakten_restructured = pd.read_pickle(args.klimafakten_path)
    klimafakten_claims = klimafakten_restructured.drop(columns=['rephrasings', 'rephrasings_ppl', 'rephrasings_bertscore', 'rephrasings_class', 'gemini_scores'])
    # create IDs: link + positive/negative
    klimafakten_claimIDs = []
    for idx, row in klimafakten_claims.iterrows():
        if row['type'] == 'positive':
            klimafakten_claimIDs.append('klimafakten_positive_' + row['link'])
        else:
            klimafakten_claimIDs.append('klimafakten_negative_' + row['link'])
    
    klimafakten_claims['claimID'] = klimafakten_claimIDs

    klimafakten_claim_text = klimafakten_claims['final_gemini_claim'].values.tolist()
    klimafakten_sources = ['Klimafakten'] * len(klimafakten_claims)
    klimafakten_origin_claims = ['synthetic'] * len(klimafakten_claims)
    klimafakten_original_claims = klimafakten_claims['fact'].values.tolist()

    klimafakten_final_claims = pd.DataFrame(
    {
        'claimID': klimafakten_claimIDs,
        'claim': klimafakten_claim_text,
        'source': klimafakten_sources,
        'generation_method': klimafakten_origin_claims,
        'original_claim': klimafakten_original_claims
        })

    # Process Correctiv
    correctiv = pd.read_pickle(args.correctiv_path)
    correctiv_claims = correctiv.drop(columns=['Bewertung', 'Bewertung Content', 'Resources', 'gemini_rephrasings', 'rephrasings_json', 'rephrasings_bertscore', 'rephrasings_class', 'rephrasings_ppl', 'gemini_scores'])
    correctiv_claimIDs = correctiv_claims['URL'].values.tolist()
    correctiv_claim_texts = correctiv_claims['final_gemini_claim'].values.tolist()
    correctiv_sources = ['Correctiv'] * len(correctiv_claims)
    correctiv_origin_claims = ['synthetic'] * len(correctiv_claims)
    correctiv_original_claims = correctiv_claims['Behauptung'].values.tolist()

    correctiv_final_claims = pd.DataFrame(
    {
        'claimID': correctiv_claimIDs,
        'claim': correctiv_claim_texts,
        'source': correctiv_sources,
        'generation_method': correctiv_origin_claims,
        'original_claim': correctiv_original_claims
        })

    # concat gercct_final_claims, klimafakten_final_claims, correctiv_final_claims, and klimawandel_final_claims
    final_german_claims = pd.concat([gercct_final_claims, klimafakten_final_claims, correctiv_final_claims, klimawandel_final_claims])
    # make new index
    final_german_claims = final_german_claims.reset_index(drop=True)

    final_german_claims.to_pickle('final_german_claims.pkl')

if __name__ == "__main__":
    main()