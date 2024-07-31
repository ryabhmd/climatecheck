import pandas as pd
import requests
import csv
import json
import time
import os

from datasets import load_dataset
import google.generativeai as genai



def main():

    # load ClimateFEVER
    climate_fever = load_dataset("tdiggelm/climate_fever")

    # Insert API key here
    GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)

    model = genai.GenerativeModel('gemini-1.5-flash')

    claim_ids = []
    claims = []
    claim_rephrasings = []

    for split, dataset in climate_fever.items():
        for example in dataset:
            claim_id = example['claim_id']
            claim = example['claim']
            claim_ids.append(claim_id)
            claims.append(claim)
            
            response = model.generate_content(f"I will give you a claim extracted from a news article and I want you to rephrase it as if you were a layperson tweeting about it. Take into account stylistic features of social media text such as use of acronyms and abbreviations, informal language and internet slang, use of hashtags, mentions and emojis, and use of non-complex and frequent words. Also pay attention to any verb tenses and use of named entities. Give me three tweet options in JSON format. The claim: {claim}")
            claim_rephrasings.append(response.text)
            
            # Free tier limit for Gemini 1.5 Flash is 15 requests per minute and 1.5K requests per day
            if len(claim) % 15 == 0:
                time.sleep(60 * 10) # Sleep for 10 minutes every 15 items
                
                
    data = {
        'claim_id': claim_ids,
        'claim': claims,
        'rephrasings': claim_rephrasings, 
        }

    climatefever_rephrased = pd.DataFrame(data)

    climatefever_rephrased.to_csv('climatefever_rephrased.csv')

if __name__ == "__main__":
    main()