import pandas as pd
from evaluate import load
import re
import ast
import json
from transformers import (AutoModelForSequenceClassification,
 AutoTokenizer, 
 pipeline, TextClassificationPipeline, 
 AutoModelForCausalLM)

"""
This script evaluates the synthetic results of claims created by Gemini API.

The evaluation is based on:
1. Fluency: perplexity score using gpt2 for English and dbmdz/german-gpt2 for German
2. Similarity to original claim: BERTScore
3. Style transfer quality: classifier result that distinguishes between tweets vs non-tweets (see style_classifier.py).
"""

def fix_missing_commas(json_string):
    # Regex to find where a comma is missing between key-value pairs
    # Matches situations where a closing quote of a value is followed by a quote of the next key without a comma
    corrected_json_string = re.sub(r'"\s*(\s*"[a-zA-Z_])', r'",\1', json_string)
    return corrected_json_string

def extract_json_from_text(text):

    # Search for the JSON in the text
    match = re.search(r'```json(.*?)```', text, re.DOTALL)

    if match:
        json_data = match.group(1)  # Extract the JSON part

        # Attempt to parse JSON
        try:
            json_dict = json.loads(json_data)  # Parse the JSON
            return json_dict
        except json.JSONDecodeError as e:
            # Try fixing the JSON and parsing again
            fixed_json_data = fix_missing_commas(json_data)
            try:
                json_dict = json.loads(fixed_json_data)
                return json_dict
            except json.JSONDecodeError as e2:
              try:
                # Remove non-breaking spaces
                cleaned_string = re.sub(r'\xa0', ' ', fixed_json_data)
                # Remove newlines and extra spaces around curly braces and brackets
                cleaned_string = re.sub(r'\s*(\{|\}|\[|\])\s*', r'\1', cleaned_string)
                # Remove any redundant spaces
                cleaned_string = re.sub(r'\s+', ' ', cleaned_string)
                json_dict = json.loads(cleaned_string)
                return json_dict
              except json.JSONDecodeError as e3:
                print(idx)
                print(cleaned_string)
                print(f"Error decoding JSON after fixing: {e3}")
                return None
    else:
        print("No JSON found in the text.")
        return None

def get_perplexity_scores(claims_data, model_id):

    perplexity = load("perplexity",  module_type= "metric")
    
    for idx, row in claims_data.iterrows():
        row_ppl = []
        if type(row["rephrasings_json"]) == list:
            for tweet in row["rephrasings_json"]:
                results = perplexity.compute(
                    model_id=model_name,
                    add_start_token=False,
                    predictions=[tweet])
                row_ppl.append(round(results["perplexities"][0], 2))
            ppl_tweets.append(row_ppl)
        else:
            ppl_tweets.append(None)

    claims_data['rephrasings_ppl'] = ppl_tweets

    return claims_data 

def get_bertscores(claims_data, lang):

    bertscore = load("bertscore")
    bertscores = []

    for idx, row in claims_data.iterrows():
        row_bertscore = []
        if type(row["rephrasings_json"]) == list:
            for dict in row["rephrasings_json"]:
                try:
                    tweet = dict["tweet"]
                except:
                    tweet = dict["text"]
                results = bertscore.compute(
                    predictions=[tweet], 
                    references=[row['claim']], 
                    lang=lang)
                row_bertscore.append(results)
            bertscores.append(row_bertscore)
        else:
            bertscores.append(None)
    
    claims_data['rephrasings_bertscores'] = bertscores
    return claims_data

def get_classifier_scores(model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, truncation=True, max_length=512)

    for idx, row in claims_data.iterrows():
        row_classification = []
        if type(row["rephrasings_json"]) == list:
            for dict in row["rephrasings_json"]:
                try:
                    tweet = dict["tweet"]
                except:
                    tweet = dict["text"]
                results = classifier(text)
                row_classification.append(results)
            class_scores.append(row_classification)
        else:
            class_scores.append(None)

    claims_data['style_scores'] = class_scores

    return claims_data

def get_final_scores(claims_data):

    final_scores = []
    
    for idx, row in claims_data.iterrows():
        row_final = []
        if type(row["rephrasings_json"]) == list:
            ppl = row['rephrasings_ppl']
            bertscore = row['rephrasings_bertscores']
            style = row['style_scores']
            
            for i in range(3):
                bertscore_i = bertscore[i]['f1'][0]*100
                
                if style[i][0]['label'] == 'LABEL_1':
                    style_i = style[i][0]['score']*100
                else:
                    style_i = 0
                    
                ppl_i = ppl[i]
                
                final_score = bertscore_i + style_i - ppl_i
                row_final.append(round(final_score, 2))
            
            final_scores.append(row_final)
        else:
            final_scores.append(None)

    claims_data['gemini_scores'] = final_scores
    return claims_data

def get_final_claims(claims_data):

    final_claims = []

    for idx, row in claims_data.iterrows():
        if type(row["rephrasings_json"]) == list:
            final_scores = row['gemini_scores']
            # get the index of the max value
            winner_idx = final_scores.index(max(final_scores))
            try:
                final_claims.append(row['rephrasings_json'][winner_idx]['tweet'])
            except:
                final_claims.append(row['rephrasings_json'][winner_idx]['text'])
        else:
            final_claims.append(None)

    claims_data["final_gemini_claim"] = final_claims
    return claims_data


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path for data containing the synthetic claims to be evaluated")
    parser.add_argument("--perplexity_model_id", type=str, help="HF model name that will be used to evaluate perplexity of text")
    parser.add_argument("--lang", type=str, help="language of dataset: en or de")
    parser.add_argument("--classifier_id", type=str, help="HF model name of the text style classifier")
    
    args = parser.parse_args()

    claims_data = pd.read_pickle(args.data_path)
    perplexity_model_id = args.perplexity_model_id
    lang = args.lang
    classifier_id = args.classifier_id

    # If English, get datasets with synthetic tweets and preprocess them
    if lang == 'en':
        rephrasings_json = []
        
        for idx, row in claims_data.iterrows():
            # check if row["rephrasing"] is not nan
            if type(row["rephrasings"]) == str:
                rephrasings_json.append(extract_json_from_text(row["rephrasings"]))
            else:
                rephrasings_json.append(None)
        
        claims_data['rephrasings_json'] = rephrasings_json


    # Same for German
    elif lang == 'de':
        rephrasings_json = []

        for idx, row in claims_data.iterrows():
            try:
                if type(row["gemini_rephrasings"]) == str:
                    rephrasings_json.append(ast.literal_eval(row["gemini_rephrasings"]))
                else:
                    rephrasings_json.append(None)
            except:
                if type(row["rephrasings"]) == str:
                    rephrasings_json.append(ast.literal_eval(row["rephrasings"])['tweets'])
                else:
                    rephrasings_json.append(None)

        claims_data['rephrasings_json'] = rephrasings_json
    
    else:
        raise ValueError("Expected lang to be either 'en' or 'de'.")


    claims_data = get_perplexity_scores(claims_data, model_id)
    claims_data = get_bertscores(claims_data, lang)
    claims_data = get_classifier_scores(claims_data, classifier_id)

    claims_data = get_final_scores(claims_data)
    claims_data = get_final_claims(claims_data)

    claims_data.to_pickle("claims_data_evaluated.pkl")


if __name__ == "__main__":
    main()