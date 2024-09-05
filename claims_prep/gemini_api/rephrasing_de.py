import pandas as pd
import time
import argparse
from datasets import load_dataset
import google.generativeai as genai

"""
Use Gemini API to rephrase the Klimafakten and Correctiv datasets from 
formal text to tweets. 
"""

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--klimafakten_path", type=str, help="Path for Klimafakten dataset")
    parser.add_argument("--corrective_path", type=str, help="Path for Correctiv dataset")
    parser.add_argument("--google_api_key", type=str, help="Google API to use Gemini")

    args = parser.parse_args()

    klimafakten = pd.read_csv(args.klimafakten_path)

    GOOGLE_API_KEY = args.google_api_key

    genai.configure(api_key=GOOGLE_API_KEY)

    model = genai.GenerativeModel('gemini-1.5-flash')
    
    gemini_rephrasings = []
    
    for idx, row in klimafakten.iterrows():
        print("Rephrasing Klimafakten...")
        fact = row['Fact']
        try:
            response = model.generate_content(f"I will give you a fact about a common misinformation about climate change in German. Your task it to rephrase it as if you were 1. A layperson who believes this fact and is tweeting about it and 2. A layperson who does not believe this fact and is tweeting about it. Give me three tweet options in German for each scenario.  The tweets should be framed as claims and include common linguistic features of tweets such as use of abbreviations and acronyms, use of emojis, use of slang and celloqualisms, flexible use of capitalization, and use of common interjections. Fact: {fact}. Give your answer in a JSON format.")
            gemini_rephrasings.append(response.text)
        
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            gemini_rephrasings.append(f"Gemini Error: {error_type}, {error_message}")
            
        if (idx+1) % 15 == 0:
            time.sleep(60)
            
    klimafakten['rephrasings'] = gemini_rephrasings
    klimafakten.to_csv('klimafakten_rephrased.csv', index=False)
    print('Finished rephrasing Klimafakten and saved at klimafakten_rephrased.csv')

    correctiv = pd.read_csv(args.corrective_path)

    gemini_rephrasings = []
    for idx, row in correctiv.iterrows():
        claim = row['Behauptung']
        
        try:
            response = model.generate_content(f"I will give you a claim about climate change in German. Your task it to rephrase it as if you were a layperson who believes this fact and is tweeting about it. Give me three tweet options in German for each scenario. The tweets should be framed as claims and include common linguistic features of tweets such as use of abbreviations and acronyms, use of emojis, use of slang and celloqualisms, flexible use of capitalization, and use of common interjections. Claim: {claim}. Give your answer in a JSON format.")
            gemini_rephrasings.append(response.text)
            
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            gemini_rephrasings.append(f"Gemini Error: {error_type}, {error_message}")

        if (idx+1) % 15 == 0:
            time.sleep(60)
    
    correctiv['rephrasings'] = gemini_rephrasings
    correctiv.to_csv('corrective_rephrased.csv')
    print('Finished rephrasing Correctiv and saved at corrective_rephrased.csv')


if __name__ == "__main__":
    main()
