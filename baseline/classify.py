import argparse
import os
import re
import pickle
import torch
import csv
import json
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForCausalLM,
    pipeline
)
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
from collections import Counter
from openai import OpenAI

# API_KEY = os.environ.get('API_KEY')
BASE_URL = os.environ.get('BASE_URL')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

prompt_instructions = """You are an expert claim verification assistant with vast knowledge of climate change , climate science , environmental science , physics , and energy science.
    Your task is to check if the Claim is correct according to the Evidence. Generate 'Supports' if the Claim is correct according to the Evidence, or 'Refutes' if the
    claim is incorrect or cannot be verified. Or 'Not enough information' if you there is not enough information in the evidence to make an informed decision. Only return the verification verdict."""

def extract_prediction(text):
    """
    Extracts the prediction label ('supports', 'refutes', or 'not enough information') from the input text.
    Used to get the prediction of the text returned by causal models. 

    Args:
        text (str): The input text containing the prediction.

    Returns:
        str: The extracted label in lowercase ('supports', 'refutes', or 'not enough information').
             Returns 'unknown' if no valid label is found.
    """
    # Search for the prediction in the text
    match = re.search(r'\[(.*?)\]', text)
    if match:
        # Extract the content within the brackets and normalize to lowercase
        prediction = match.group(1).strip().lower()
        # Normalize known labels
        if prediction in {"supports", "'supports'", "\"supports\""}:
            return "supports"
        elif prediction in {"refutes", "'refutes'", "\"refutes\""}:
            return "refutes"
        elif prediction in {"not enough information", "'not enough information'", "\"not enough information\""}:
            return "not enough information"
    return "unknown"

def process_causal_lm(tokenizer, model, data, model_name, output_data_path, batch_size=16):
    """
    Processes Causal LMs by prompting them with the given claim-abstract pairs.
    Inputes:
    - tokenizer: an AutoTokenizer object.
    - model: an AutoModelForCausalLM object.
    - data: HF Dataset which includes the features: 'claim' and 'abstract'.
    - model_name: the cleaned model name for the purposes of saving intermediate results of batches as the model processes the inputs.
    - batch_size: batch size to be processes by the model. Default is 16.
    - output_data_path: path to save both predictions as batched when the model is running + a final pickle file of the data with predictions.
    """
    
    predictions = []
    
    all_prompts = [f"""You are an expert claim verification assistant with vast knowledge of climate change , climate science , environmental science , physics , and energy science.
    Your task is to check if the Claim is correct according to the Evidence. Generate 'Supports' if the Claim is correct according to the Evidence, or 'Refutes' if the
    claim is incorrect or cannot be verified. Or 'Not enough information' if you there is not enough information in the evidence to make an informed decision.
    Evidence: {row["abstract"]}
    Claim: {row["claim"]}
    Provide the final answer in a Python list format.
    Let's think step-by-step:""" for idx, row in data.iterrows()]

    all_messages = [[{"role": "user", "content": prompt}] for prompt in all_prompts]

    if "llama" in model_name.lower():
        tokenizer.pad_token_id = model.config.eos_token_id[0]
    else:
        tokenizer.pad_token_id = model.config.eos_token_id
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=500,
        do_sample=False,
        trust_remote_code=True,
        truncation=True,
    )

    # Make dir to save predictions to while the model is running
    os.makedirs(f"{output_data_path}/{model_name}", exist_ok=True)

    predictions = []
    for i in tqdm(range(0, len(all_messages), batch_size)):
        outputs = pipe(all_messages[i : i + batch_size], batch_size=batch_size)
        batch_predictions = [extract_prediction(item[0]["generated_text"]) for item in outputs]
        predictions.extend(batch_predictions)
        # Save predictions list as pickle
        with open(f'{output_data_path}/{model_name}/{model_name}_batch_{i}.pkl', 'wb') as file: 
            pickle.dump(batch_predictions, file)
        torch.cuda.empty_cache()  # Clear cached memory
        torch.cuda.ipc_collect()  # Reduce fragmentation
        with open(f'{output_data_path}/{model_name}_all_batches.pkl', 'wb') as file: 
            pickle.dump(predictions, file)

    print(f"All predction batches saved in {output_data_path}/{model_name}.")
    return predictions

def process_with_model(data, 
                       model_name, 
                       output_data_path,
                       evidentiary_threshold=3, 
                       abstracts_threshold=3):
    """
    Given a model name and type, this function processes 
    Inputs:
    - data: HF Dataset which includes the features: 'claim' and 'abstract'.
    - model_name: the name of the model to be processed (as written on HF).
    - model_type: either "sequence_classification" or "causal_lm".
    - evidentiary_threshold: the threshold of needed evidentiary predictions for each claim-abstract pair. This is used in order to not process a pair further if it already meets the threshold in order
    to save compute resources and time. Use case example: using 6 models for pooling, 3 evidentiary predictions are needed to inlclude a claim-abstract pair in the annotation data.
    - abstracts_threshold: the number of abstracts we want each claim to be connected to. If a claim reaches this threshold (e.g. already has 3 connected abstracts), it is not processed further to save
    compute resources and time.
    - output_data_path: path to save the intermediate + final data.
    """

    print(f"Processing with model: {model_name}")

    cleaned_model_name = model_name.split("/")[-1]

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    predictions = process_causal_lm(tokenizer, model, data, cleaned_model_name, output_data_path)

    # Free up memory by deleting model
    del model

    # Save model predictions in dataset
    data[cleaned_model_name] = predictions
    # data = data.add_column(cleaned_model_name, predictions)

    data.to_csv(f"{output_data_path}/classifications_{cleaned_model_name}.csv")

    # # Update votes for each row
    # votes_all_rows = []
    # for row in tqdm(data, desc="Adding counts"):
    #     total_votes = row["total_votes"]
    #     model_predictions = row[cleaned_model_name]

    #     total_votes[model_predictions] += 1
    #     votes_all_rows.append(total_votes)

    # data = data.remove_columns("total_votes")
    # data = data.add_column("total_votes", votes_all_rows)

    # # Remove rows with "evidentiary_threshold" evidentiary abstracts from the working dataset
    # annotation_dataset = []
    # remaining_data = []

    # for row in tqdm(data, desc="Filtering evidentiary abstracts"):
    #     # Count how many evidentiaty predictions exist for each claim-abstract; split the data to "annotation_data" which contains claims that already have the 
    #     # required number of abstract and "remaining_data" which will be processed further.
        
    #     evidentiary_count = row["total_votes"]["supports"] + row["total_votes"]["refutes"]
    #     if evidentiary_count >= evidentiary_threshold:
    #         annotation_dataset.append(row)
    #     else:
    #         remaining_data.append(row)

    # # Check if "abstracts_threshold" rows already exist with the same claim in the final dataset to remove it completely
    # remaining_data_filtered = []
    # annotation_data_claims = [row['claim'] for row in annotation_dataset]
    # annotation_data_claim_counts = Counter(annotation_data_claims)

    # for row in tqdm(remaining_data, desc="Filtering by claim limit"):
    #     claim = row["claim"]
    #     if annotation_data_claim_counts[claim] < abstracts_threshold:
    #         remaining_data_filtered.append(row)

    # # Save intermediate dataset -> this is the remaining data which will be passed on to the next model
    # remaining_data_filtered = Dataset.from_list(remaining_data_filtered)
    # remaining_data_filtered.save_to_disk(f"{output_data_path}/intermediate_data_{cleaned_model_name}.hf")

    # # Save the final dataset with filtered rows -> this is the dataset that we already know will be part of the annotatation corpus
    # # After each model, an annotations data will be save, then we will stick them all together and make sure every claim has only the top 3 abstracts
    # annotation_dataset = Dataset.from_list(annotation_dataset)
    # if len(annotation_dataset) > 0:
    #     annotation_dataset.save_to_disk(f"{output_data_path}/annotation_data_{cleaned_model_name}.hf")

    print(f"Finished processing with model: {model_name}")
    return 0

def get_classifcation(data, model, tokenizer, output_path):
    print(f"[INFO] Start classification for # {data.shape[0]} claim-abstract pairs")
    tokenizer.pad_token_id = model.config.eos_token_id[0]
    pipe = pipeline("text-generation", 
                    model=model,
                    tokenizer=tokenizer,
                    return_full_text=False,
                    max_new_tokens=500,
                    do_sample=False,
                    trust_remote_code=True,
                    truncation=True)
    
    # with open(output_path, 'wt') as out_file:
    #     tsv_writer = csv.writer(out_file, delimiter='\t')
    #     for idx, row in tqdm(data.iterrows()):
    #         messages=[{"role":"system","content":prompt_instructions},
    #                 {"role":"user","content":f"Claim: {row['claim']} \nEvidence: {row['abstract']}"}]
            
    #         output = pipe(messages)
    #         label_prediction = extract_prediction(output[0]["generated_text"])

    #         tsv_writer.writerow([row['claim_id'], row['claim'], row['abstract_id'], row['abstract'], row['rank'], label_prediction])

    for idx, row in tqdm(data.iterrows()):
        messages=[{"role":"system","content":prompt_instructions},
                    {"role":"user","content":f"Claim: {row['claim']} \nEvidence: {row['abstract']}"}]
            
        output = pipe(messages)
        # label_prediction = extract_prediction(output[0]["generated_text"])

        entry = {
            'claim_id': row['claim_id'],
            'claim': row['claim'],
            'abstract_id': row['abstract_id'],
            'abstract': row['abstract'],
            'rank': row['rank'],
            'label': output[0]['generated_text'],
        }
        with open(output_path, 'a') as f:
            f.write(json.dumps(entry))
            f.write('\n')

    return 0    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", type=str)
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    # parser.add_argument("--batch", type=int, default=16)

    args = parser.parse_args()

    model_name = args.model
    input_path = args.input
    output_path = args.output
    # batch_size = args.batch

    reranked_df = pd.read_csv(input_path)
    reranked_df['label'] = ""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    get_classifcation(reranked_df, model, tokenizer, output_path)
    # labeled_df.to_csv(output_path)

    # client = OpenAI(
    #     base_url = BASE_URL
    # )
    # print(f"[INFO] Start classification with {model_name}")
    # for idx, row in tqdm(reranked_df.iterrows()):
    #     chat_completion = client.chat.completions.create(
    #         messages=[{"role":"system","content":prompt_instructions},
    #                 {"role":"user","content":f"Claim: {row['claim']} \nEvidence: {row['abstract']}"}],
    #         model= model_name,
    #     )
    #     label = extract_prediction(chat_completion.choices[0].message.content)
    #     reranked_df.at[idx, 'label'] = label


    

    print("Processing complete.")
