from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, TextClassificationPipeline
from transformers.pipelines.pt_utils import KeyDataset
import pandas as pd

"""
Script that relies on an exiting environmental claim detection model finetuned on climateBERT (https://huggingface.co/climatebert/environmental-claims) to classify extracted tweets from the ClimaConvo dataset.

Tweets classifies as claims + tweets classifies as not claims with a probablity of less than 0.8 are extracted to be examined manually. 
"""
def main():

    model_name = "climatebert/environmental-claims"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)

    climaconvo_data = pd.read_csv('climaconvo_relevants_with_text_and_errors.csv')
    climaconvo_text = climaconvo_data[climaconvo_data['text'].notnull()]

    claim_label = []
    claim_prob = []

    for idx, row in climaconvo_text.iterrows():
        text = row['text']
        result = classify_tweet(text)
        claim_label.append(result[0]['label'])
        claim_prob.append(result[0]['score'])

    climaconvo_text['claim_label'] = claim_label
    climaconvo_text['claim_prob'] = claim_prob

    climaconvo_text.to_csv('climaconvo_relevants_claim_detected.csv', index=False)

    climaconvo_no_75 = climaconvo_text[(climaconvo_text['claim_prob'] <= 0.8) & (climaconvo_text['claim_label'] == 'no')]

    climaconvo_yes = climaconvo_text[climaconvo_text['claim_label'] == 'yes']

    climaconvo_text_claims = pd.concat([climaconvo_no_75, climaconvo_yes])

    climaconvo_text_claims.to_csv('climaconvo_text_claims.csv', index=False)


if __name__ == "__main__":
    main()