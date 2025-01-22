import torch
from datasets import load_from_disk, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForCausalLM,
    pipeline
)
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
from collections import Counter
import re
import pickle
import argparse
import os


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define the models and their respective types
models_info = {
    "FacebookAI/roberta-large-mnli": "sequence_classification",
    "microsoft/deberta-v2-xxlarge-mnli": "sequence_classification",
    "joeddav/xlm-roberta-large-xnli": "sequence_classification",
    "01-ai/Yi-1.5-9B-Chat-16K": "causal_lm",
    "Qwen/Qwen1.5-14B-Chat": "causal_lm",
    "meta-llama/Llama-3.1-8B-Instruct": "causal_lm",
}

def get_previous_model(model_name):
  """
  Given a model name, returns the previous model in the list.
  If the given model is the first in the list, returns None.

  Args:
    model_name: The name of the model.

  Returns:
    The name of the previous model in the list, or None if the given model 
    is the first in the list.
  """

  model_list = list(models_info.keys()) 
  try:
    index = model_list.index(model_name)
    if index == 0:
      return None
    else:
      return model_list[index - 1]
  except ValueError:
    # Handle the case where the given model_name is not in the list
    return None


def process_sequence_classification(tokenizer, model, data):

  predictions = []
  labels = ["refutes", "not enough information", "supports"]

  for row in tqdm(data, desc="Processing rows"):
    # Tokenize the claim and abstract
    inputs = tokenizer(
        row['claim'],
        row['abstract'],
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    ).to('cuda')
    
    # Perform inference with the model
    with torch.no_grad():
      outputs = model(**inputs)
      logits = outputs.logits
      prediction = labels[torch.argmax(logits, dim=-1).item()]

    predictions.append(prediction)

  return predictions

def extract_prediction(text):
    """
    Extracts the prediction label ('supports', 'refutes', or 'not enough information') from the input text.

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

def process_causal_lm(tokenizer, model, data, model_name, batch_size=16):
    
    predictions = []
    all_prompts = [f"""You are an expert claim verification assistant with vast knowledge of climate change , climate science , environmental science , physics , and energy science.
    Your task is to check if the Claim is correct according to the Evidence. Generate ’Supports’ if the Claim is correct according to the Evidence, or ’Refutes’ if the
    claim is incorrect or cannot be verified. Or 'Not enough information' if you there is not enough information in the evidence to make an informed decision.
    Evidence: {row["abstract"]}
    Claim: {row["claim"]}
    Provide the final answer in a Python list format.
    Let’s think step-by-step:""" for row in data]

    all_messages = [[{"role": "user", "content": prompt}] for prompt in all_prompts]

    tokenizer.pad_token_id = model.config.eos_token_id[0]

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
    os.makedirs(f"/netscratch/abu/Shared-Tasks/ClimateCheck/data/claims/predictions_intermediate_results/{model_name}_10-20", exist_ok=True)

    predictions = []
    for i in tqdm(range(0, len(all_messages), batch_size)):
        outputs = pipe(all_messages[i : i + batch_size], batch_size=batch_size)
        batch_predictions = [extract_prediction(item[0]["generated_text"]) for item in outputs]
        predictions.extend(batch_predictions)
        # Save predictions list as pickle
        with open(f'/netscratch/abu/Shared-Tasks/ClimateCheck/data/claims/predictions_intermediate_results/{model_name}_10-20/{model_name}_batch_{i}.pkl', 'wb') as file: 
            pickle.dump(batch_predictions, file)
        torch.cuda.empty_cache()  # Clear cached memory
        torch.cuda.ipc_collect()  # Reduce fragmentation
        with open(f'/netscratch/abu/Shared-Tasks/ClimateCheck/data/claims/predictions_intermediate_results/{model_name}_all_batches_10-20.pkl', 'wb') as file: 
            pickle.dump(predictions, file)

    print(f"All predction batches saved in /netscratch/abu/Shared-Tasks/ClimateCheck/data/claims/predictions_intermediate_results/{model_name}_10-20.")
    return predictions

def process_with_model(data, model_name, model_type):

    print(f"Processing with model: {model_name}")

    cleaned_model_name = model_name.split("/")[-1]

    if model_type == "sequence_classification":
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        predictions = process_sequence_classification(tokenizer, model, data)
    elif model_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        predictions = process_causal_lm(tokenizer, model, data, cleaned_model_name)

    # Free up memory by deleting model
    del model

    # Save model predictions in dataset
    data = data.add_column(cleaned_model_name, predictions)

    # Update votes for each row
    votes_all_rows = []
    for row in tqdm(data, desc="Adding counts"):
        total_votes = row["total_votes"]
        model_predictions = row[cleaned_model_name]

        total_votes[model_predictions] += 1
        votes_all_rows.append(total_votes)

    data = data.remove_columns("total_votes")
    data = data.add_column("total_votes", votes_all_rows)

    # Remove rows with 4 evidentiary abstracts from the working dataset
    annotation_dataset = []
    remaining_data = []

    for row in tqdm(data, desc="Filtering evidentiary abstracts"):
        # Check if 4 or more evidentiary abstracts exist
        evidentiary_count = row["total_votes"]["supports"] + row["total_votes"]["refutes"]
        if evidentiary_count >= 3:
            annotation_dataset.append(row)
        else:
            remaining_data.append(row)

    # Check if 3 rows already exist with the same claim in the final dataset to remove it completely
    remaining_data_filtered = []
    annotation_data_claims = [row['claim'] for row in annotation_dataset]
    annotation_data_claim_counts = Counter(annotation_data_claims)

    for row in tqdm(remaining_data, desc="Filtering by claim limit"):
        claim = row["claim"]
        if annotation_data_claim_counts[claim] < 3:
            remaining_data_filtered.append(row)

    # Save intermediate dataset -> this is the remaining data which will be passed on to the next model
    remaining_data_filtered = Dataset.from_list(remaining_data_filtered)
    remaining_data_filtered.save_to_disk(f"/netscratch/abu/Shared-Tasks/ClimateCheck/data/claims/pooling_results/intermediate_data_{cleaned_model_name}_top10-20.hf")

    # Save the final dataset with filtered rows -> this is the dataset that we already know will be part of the annotations
    # After each model, an annotations data will be save, then we will stick them all together and make sure every claim has only the top 3 abstracts
    annotation_dataset = Dataset.from_list(annotation_dataset)
    if len(annotation_dataset) > 0:
        annotation_dataset.save_to_disk(f"/netscratch/abu/Shared-Tasks/ClimateCheck/data/claims/pooling_results/annotation_data_{cleaned_model_name}_top10-20.hf")

    print(f"Finished processing with model: {model_name}")
    return remaining_data_filtered


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        help="Name of model on HF.",
    )

    parser.add_argument(
        "--model_type",
        "-t",
        type=str,
        help="Model type: either 'sequence_classification' or 'causal_lm'."
    )

    args = parser.parse_args()

    model_name = args.model_name
    model_type = args.model_type

    previous_model = get_previous_model(model_name)

    if previous_model:
        # Load data left after finishing run on previous model
        previous_model = previous_model.split("/")[-1]
        data = load_from_disk(f"/netscratch/abu/Shared-Tasks/ClimateCheck/data/claims/pooling_results/intermediate_data_{previous_model}_top10-20.hf")
    else:
        data = load_from_disk("/netscratch/abu/Shared-Tasks/ClimateCheck/data/claims/final_english_claims_expanded_masmarco_top10-20.hf")

    remaining_data_filtered = process_with_model(data, model_name, model_type)

    print("Processing complete.")