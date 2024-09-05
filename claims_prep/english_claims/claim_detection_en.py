from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, TextClassificationPipeline
from transformers.pipelines.pt_utils import KeyDataset
import pandas as pd
import argparse

"""
Script that relies on an exiting environmental claim detection model finetuned on climateBERT (https://huggingface.co/climatebert/environmental-claims) 
to classify extracted input text as containing an environmental claim or not.
This script was used for filtering ClimaConvo and DEBAGREEMENT data. 

When running the script the following arguments should be given:
--data_path: the path to the data that is to be classified. 
--data_source: accepts only two variants 'ClimaConvo' or 'DEBAGREEMENT'. 

Text classified as claims + text classified as not claims with a probablity of less than 0.8 are extracted to be examined manually. 
"""

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path for dataset that will be classified")
    parser.add_argument("--data_source", type=str, help="Should be either ClimaConvo or DEBAGREEMENT")
    args = parser.parse_args()

    data_path = args.data_path
    data_source = args.data_source

    model_name = "climatebert/environmental-claims"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)

    data_df = pd.read_csv(data_path)
    if data_source == 'ClimaConvo':
        data_text = climaconvo_data[climaconvo_data['text'].notnull()]
    elif data_source == 'DEBAGREEMENT':
        data_text = climaconvo_data[climaconvo_data['body'].notnull()]
    else:
        raise ValueError('Data source should be either ClimaConvo or DEBAGREEMENT')

    claim_label = []
    claim_prob = []

    for idx, row in data_text.iterrows():
        if data_source == 'ClimaConvo':
            text = row['text']
        else:
            text = row['body']
        result = classifier(tweet)
        claim_label.append(result[0]['label'])
        claim_prob.append(result[0]['score'])

    data_text['claim_label'] = claim_label
    data_text['claim_prob'] = claim_prob

    data_text_no_75 = data_text[(data_text['claim_prob'] <= 0.8) & (data_text['claim_label'] == 'no')]

    data_text_yes = data_text[data_text['claim_label'] == 'yes']

    data_text_claims = pd.concat([data_text_no_75, data_text_yes])
    
    if data_source == 'ClimaConvo':
        data_text_claims.to_csv('climaconvo_text_claims.csv', index=False)
    else:
        data_text_claims.to_csv('debagreement_text_claims.csv', index=False)

if __name__ == "__main__":
    main()