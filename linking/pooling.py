import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForCausalLM, 
    pipeline
)
from tqdm import tqdm
import pandas as pd
import json
import re
import pickle

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define the models and their respective types
models_info = {
    "microsoft/deberta-v2-xxlarge-mnli": "sequence_classification",
    "joeddav/xlm-roberta-large-xnli": "sequence_classification",
    "FacebookAI/roberta-large-mnli": "sequence_classification",
    "Qwen/Qwen1.5-14B-Chat": "causal_lm",
    #"google/gemma-2-9b-it": "causal_lm", waiting for access
    "meta-llama/Llama-2-13b-chat-hf": "causal_lm",
    "mistralai/Mistral-Nemo-Instruct-2407": "causal_lm",
    "01-ai/Yi-1.5-9B-Chat": "causal_lm"
}

# Function to process sequence classification models
def process_sequence_classification_batch(tokenizer, model, claims, abstracts):
    inputs = tokenizer(
        claims,
        abstracts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    labels = ["refutes", "not enough information", "supports"]
    predictions = [labels[torch.argmax(logit).item()] for logit in logits]
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
        else:
            return prediction
    return "unknown"

def process_causal_lm_batch(pipe, claims, abstracts):
    batch_results = []
    for claim, abstract in zip(claims, abstracts):
        prompt = f"""You are an expert claim verification assistant with vast knowledge of climate change , climate science , environmental science , physics , and energy science.
        Your task is to check if the Claim is correct according to the Evidence. Generate ’Supports’ if the Claim is correct according to the Evidence, or ’Refutes’ if the claim is incorrect or cannot be verified. Or 'Not enough information' if you there is not enough information in the evidence to make an informed decision.
        Evidence: {abstract}
        Claim: {claim}
        Provide the final answer in a Python list format. 
        Let’s think step-by-step:"""

        messages = [
            { "role": "user", "content": prompt}
        ]
        
        response = pipe(messages)[0]["generated_text"]
        
        prediction = extract_prediction(response)
        
        batch_results.append((response, prediction))
        
    return batch_results

def main():

    data = pd.read_pickle('/netscratch/abu/Shared-Tasks/ClimateCheck/data/claims/final_english_claims_reranked_msmarco.pkl')
    results = [None] * len(data)

    # Preload models and tokenizers
    models = {}
    for model_name, model_type in models_info.items():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_type == "sequence_classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
            models[model_name] = (tokenizer, model)
        elif model_type == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
            pipe = pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                return_full_text=False,
                max_new_tokens=500,
                do_sample=False)
            models[model_name] = pipe

    for index, row in tqdm(data.iterrows(), total=len(data)):
        claim = row["atomic_claim"]
        ms_marco_results = row["ms_marco_reranking"]
        abstracts_list = [item[2] for item in ms_marco_results]
        abstracts_original_indices = [item[0] for item in ms_marco_results]
        evidentiary_abstracts = []

        n = 6 # number of abstracts in batch

        for idx in range(0, len(abstracts_list), n):  # Process abstracts in batches of n
            batch_claims = [claim] * len(abstracts_list[idx:idx+n])
            batch_abstracts = abstracts_list[idx:idx+n]
            batch_original_indices = abstracts_original_indices[idx:idx+n]

            batch_votes = [{"supports": 0, "refutes": 0, "not enough information": 0, "unknown": 0} for _ in batch_abstracts]
            
            for model_name, model_type in models_info.items():
                
                if model_type == "sequence_classification":
                    tokenizer, model = models[model_name]
                    preds = process_sequence_classification_batch(tokenizer, model, batch_claims, batch_abstracts)
                elif model_type == "causal_lm":
                    pipe = models[model_name]
                    batch_results = process_causal_lm_batch(pipe, batch_claims, batch_abstracts)
                    preds = [pred for _, pred in batch_results]

                #the overall number of predictions per abstract should be the number of models
                assert len(preds) == len(batch_abstracts), (
                    f"Unexpected predictions length: {len(preds)} for batch size: {len(batch_abstracts)}"
                )

                for pred_idx, pred in enumerate(preds):
                    batch_votes[pred_idx][pred] += 1
            
            for batch_idx, abstract_votes in enumerate(batch_votes):
                if abstract_votes["supports"] + abstract_votes["refutes"] >= 4:
                    evidentiary_abstracts.append({
                        "rank": idx + batch_idx,
                        "original_index": batch_original_indices[batch_idx],
                        "abstract": batch_abstracts[batch_idx],
                        "votes": abstract_votes
                    })
                    
            if len(evidentiary_abstracts) >= 3: # don't process more batches if top 3 are already found
                break
        
        # Sort and save top 3 abstracts
        evidentiary_abstracts.sort(key=lambda x: x["rank"])
        results[index] = evidentiary_abstracts
        
        with open('/netscratch/abu/Shared-Tasks/ClimateCheck/data/claims/pooling_results_list.pkl', 'wb') as f:
            pickle.dump(results, f)
        print(f'Saved results at index: {index}...')
        
    data["top_3_abstracts"] = results
    data.to_pickle('/netscratch/abu/Shared-Tasks/ClimateCheck/data/claims/final_english_claims_pooling.pkl')
    print('Processing complete.')

    
if __name__ == "__main__":
    main()
