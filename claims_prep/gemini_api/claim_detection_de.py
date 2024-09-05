import pandas as pd
import time
import argparse
import google.generativeai as genai

"""
Script to detect whether the text from r/Klimawandel contains scientific claims using Gemini API. 

"""
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--klimawandel_path", type=str, help="Path for Klimawandel dataset")
    parser.add_argument("--google_api_key", type=str, help="Google API to use Gemini")

    args = parser.parse_args()

    klimawandel = pd.read_pickle(arg.sklimawandel_path)

    GOOGLE_API_KEY = args.google_api_key

    genai.configure(api_key=GOOGLE_API_KEY)

    model = genai.GenerativeModel('gemini-1.5-flash')

    claim_detect = []
    
    for idx, row in klimawandel.iterrows():
        text = row['text']
        try:
            response = model.generate_content(f"Claim Detection is a task in which, given a text, the model outputs a binary response of whether the text contains a check-worthy claim against a scientific publication. Meaning that the text contains a scientific claim that can be supported or refuted using scholarly articles as evidence. Perform claim detection on the following text: {text}. Give your answer in a JSON format including the structure: 'contains_claim': 'yes/no', 'claims': [claim1, claim2, ...]. Claims should be kept in the original format and language used.")
            claim_detect.append(response.text)
        
        except:
            claim_detect.append("Gemini Error")
            
        if (idx+1) % 15 == 0:
            time.sleep(60)

    klimawandel['claim_detection'] = claim_detect
    # Remove rows with Gemini Error
    klimawandel_with_claim_detection = klimawandel[klimawandel['claim_detection'] != 'Gemini Error']

    klimawandel_with_claim_detection.to_csv('klimawandel_with_claim_detection.csv', index=False)
    print('Saved Klimawandel data with claim detection at klimawandel_with_claim_detection.csv')


if __name__ == "__main__":
    main()
