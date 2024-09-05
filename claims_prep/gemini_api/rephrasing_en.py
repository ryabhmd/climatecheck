import pandas as pd
import time
import argparse
from datasets import load_dataset
import google.generativeai as genai



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--multifc_path", type=str, help="Path for MultiFC dataset")
    parser.add_argument("--climatefeedback_path", type=str, help="Path for ClimateFeedback dataset")
    parser.add_argument("--google_api_key", type=str, help="Google API to use Gemini")

    args = parser.parse_args()
    climate_fever = load_dataset("tdiggelm/climate_fever")

    multi_fc = pd.read_csv(args.multifc_path)
    multi_fc = multi_fc[multi_fc['ScientificClaim'] == 1]
    multi_fc_claims = list(multi_fc['Claim'].values)
    multi_fc_claims_ids = list(multi_fc['claimID'].values)

    climate_feedback = pd.read_csv(args.climatefeedback_path)
    climate_feedback = climate_feedback[climate_feedback['claim'].notna()]
    climate_feedback = climate_feedback.drop_duplicates(subset=['claim'])
    climate_feedback_claims = list(climate_feedback['claim'].values)
    # treating links as IDs since there are no direct IDs in the data
    climate_feedback_ids = list(climate_feedback['link'].values)

    # Merge MultiFC and ClimateFeedback
    multiFC_feedback_claims = pd.DataFrame(columns=['claim', 'claimID', 'source'])
    multiFC_feedback_claims['claim'] = multi_fc_claims + climate_feedback_claims
    multiFC_feedback_claims['claimID'] = multi_fc_claims_ids + climate_feedback_ids
    multiFC_feedback_claims['source'] = ['multiFC'] * len(multi_fc_claims) + ['ClimateFeedback'] * len(climate_feedback_claims)
    # dedup according to claim
    multiFC_feedback_claims = multiFC_feedback_claims.drop_duplicates(subset=['claim'])


    GOOGLE_API_KEY = args.google_api_key

    genai.configure(api_key=GOOGLE_API_KEY)

    model = genai.GenerativeModel('gemini-1.5-flash')

    for split, dataset in climate_fever.items():
        print("Rephrasing ClimateFEVER...")
        print(f"Split: {split}")
        for example in dataset.select(range(0, len(dataset))):
            claim_id = example['claim_id']
            claim = example['claim']
            claim_ids.append(claim_id)
            claims.append(claim)
            
        try:
            response = model.generate_content(f"I will give you a claim extracted from a news article and I want you to rephrase it as if you were a layperson tweeting about it. Take into account stylistic features of social media text such as use of acronyms and abbreviations, informal language and internet slang, use of hashtags, mentions and emojis, and use of non-complex and frequent words. Also pay attention to any verb tenses and use of named entities. Give me three tweet options in JSON format. The claim: {claim}")
            claim_rephrasings.append(response.text)

      except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        claim_rephrasings.append(f"Gemini Error: {error_type}, {error_message}")

      # Free tier limit for Gemini 1.5 Flash is 15 requests per minute and 1.5K requests per day
      if len(claims) % 15 == 0:
        time.sleep(60 * 5) # Sleep for 10 minutes every 15 items

    data = {
        'claim_id': claim_ids,
        'claim': claims,
        'rephrasings': claim_rephrasings,
        }

    climatefever_rephrased = pd.DataFrame(data)
    climatefever_rephrased.to_csv('climatefever_rephrased.csv')
    print('Finished rephrasing ClimateFEVER and saved at climatefever_rephrased.csv')


    gemini_rephrasings = []
    for idx, row in multiFC_feedback_claims.iterrows():
        print('Rephrasing MultiFC and ClimateFeedback...')
        claim = row['claim']
        
        try:
            response = model.generate_content(f"I will give you a claim extracted from a news article and I want you to rephrase it as if you were a layperson tweeting about it. Take into account stylistic features of social media text such as use of acronyms and abbreviations, informal language and internet slang, use of hashtags, mentions and emojis, and use of non-complex and frequent words. Use the present tense more frequently and do not use many named entities. Give me three tweet options in JSON format.” The claim: {claim}")
            gemini_rephrasings.append(response.text)
        
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            gemini_rephrasings.append(f"Gemini Error: {error_type}, {error_message}")
    
    multiFC_feedback_claims['rephrasings'] = gemini_rephrasings
    multiFC_feedback_claims.to_csv('multifc_feedback_rephrased.csv')
    print('Finished rephrasing MultiFC and saved at multifc_rephrased.csv')





if __name__ == "__main__":
    main()
