import pandas as pd
import time
import argparse
from datasets import load_dataset
import google.generativeai as genai

"""
Use Gemini API to rephrase the ClimateFEVER, MultiFC, and ClimateFeedback datasets from 
formal text to tweets. 
"""

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
            response = model.generate_content("""
            Task: Given a claim extracted from a news article, produce a rephrasing as if you are a layperson tweeting about it. 
            Constraints: 
            1. Take into account stylistic features of social media text such as use of slang and informal language. **However, keep in mind that this is a serious topic and not every tweet needs to have emojis and slang.**
            2. Do not overdo your text generations. Keep them plausible enough to believe a human wrote them. 
            3. Introduce variance in rhetoric and syntactic structures of your tweets. **Not every tweet needs to contain a question.** 
            4. **Generate tweets in a neutral tone. Do not add irony or satire.** 
            5. **Keep the scientific claim that is present in the original claim** 
            6. Give three output options in a JSON format {'tweets' : [tweet1, tweet2, tweet3]}. 
            7. Before giving your answer, rewrite the prompt, expand the task at hand, and only then respond. 
            8. If you have follow-up questions, generate them and then answer them before giving the final output.
            Examples of tweets about a similar topic: 
            1. Fossil fuel projects destroy fragile ecosystems, harm social harmony in local communities, and contribute to millions of deaths by polluting the air. Fossil finance is an issue of justice. #FridaysForFuture #ClimateStrike 
            2. Fruit trees produce fresh oxygen, giving you &amp; your family cleaner air to breathe, as well as encouraging wildlife to flourish #COP27 #COP15 #ReFi #Web3 #ImplementPromisesNow #LossAndDamage #PeopleNotProfit #ClimateAction #ClimateJustice #climateStrike #Climate 
            3. your car tires are the largest source of #microplastics pollution on earth ! #gocarfree #banprivatecars #bikedontdrive #eatonlyplants and stop flying !!! #Istayontheground #flygskam #stopecocide  
            4. A study found that at least 95% of wood pallets are recycled within the manufacturing and industrial packaging industry. Recycling, waste reduction, and material utilization are all important parts of making Conner as #sustainable as possible. 
            5. #RegenerativeAgriculture: Increased #soil #carbon sequestration slows/may reverse #globalwarming. The amt of carbon stored in soil is more than  what we release each year from all sectors. #Learn more in #RecipeForSurvival! #FridaysForFuture  13 days! #Sustainability. 
            Claim: """ + claim + """
            Output:""")
            claim_rephrasings.append(response.text)

      except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        claim_rephrasings.append(f"Gemini Error: {error_type}, {error_message}")

      # Free tier limit for Gemini 1.5 Flash is 15 requests per minute and 1.5K requests per day
      if (idx+1) % 15 == 0:
        time.sleep(60)

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
            response = model.generate_content("""
            Task: Given a claim extracted from a news article, produce a rephrasing as if you are a layperson tweeting about it. 
            Constraints: 
            1. Take into account stylistic features of social media text such as use of slang and informal language. **However, keep in mind that this is a serious topic and not every tweet needs to have emojis and slang.**
            2. Do not overdo your text generations. Keep them plausible enough to believe a human wrote them. 
            3. Introduce variance in rhetoric and syntactic structures of your tweets. **Not every tweet needs to contain a question.** 
            4. **Generate tweets in a neutral tone. Do not add irony or satire.** 
            5. **Keep the scientific claim that is present in the original claim** 
            6. Give three output options in a JSON format {'tweets' : [tweet1, tweet2, tweet3]}. 
            7. Before giving your answer, rewrite the prompt, expand the task at hand, and only then respond. 
            8. If you have follow-up questions, generate them and then answer them before giving the final output.
            Examples of tweets about a similar topic: 
            1. Fossil fuel projects destroy fragile ecosystems, harm social harmony in local communities, and contribute to millions of deaths by polluting the air. Fossil finance is an issue of justice. #FridaysForFuture #ClimateStrike 
            2. Fruit trees produce fresh oxygen, giving you &amp; your family cleaner air to breathe, as well as encouraging wildlife to flourish #COP27 #COP15 #ReFi #Web3 #ImplementPromisesNow #LossAndDamage #PeopleNotProfit #ClimateAction #ClimateJustice #climateStrike #Climate 
            3. your car tires are the largest source of #microplastics pollution on earth ! #gocarfree #banprivatecars #bikedontdrive #eatonlyplants and stop flying !!! #Istayontheground #flygskam #stopecocide  
            4. A study found that at least 95% of wood pallets are recycled within the manufacturing and industrial packaging industry. Recycling, waste reduction, and material utilization are all important parts of making Conner as #sustainable as possible. 
            5. #RegenerativeAgriculture: Increased #soil #carbon sequestration slows/may reverse #globalwarming. The amt of carbon stored in soil is more than  what we release each year from all sectors. #Learn more in #RecipeForSurvival! #FridaysForFuture  13 days! #Sustainability. 
            Claim: """ + claim + """
            Output:""")
            gemini_rephrasings.append(response.text)
        
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            gemini_rephrasings.append(f"Gemini Error: {error_type}, {error_message}")

        if (idx+1) % 15 == 0:
            time.sleep(60)
    
    multiFC_feedback_claims['rephrasings'] = gemini_rephrasings
    multiFC_feedback_claims.to_csv('multifc_feedback_rephrased.csv')
    print('Finished rephrasing MultiFC and saved at multifc_feedback_rephrased.csv')


if __name__ == "__main__":
    main()
