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
def process_sequence_classification(tokenizer, model, claim, abstract):
    inputs = tokenizer(
        claim,
        abstract,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    labels = ["refutes", "not enough information", "supports"]
    prediction = labels[torch.argmax(logits, dim=-1).item()]

    return prediction

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

# Function to process causal language models
def process_causal_lm(tokenizer, model, claim, abstract):
    
    prompt = f"""You are an expert claim verification assistant with vast knowledge of climate change , climate science , environmental science , physics , and energy science.
    Your task is to check if the Claim is correct according to the Evidence. Generate ’Supports’ if the Claim is correct according to the Evidence, or ’Refutes’ if the 
    claim is incorrect or cannot be verified. Or 'Not enough information' if you there is not enough information in the evidence to make an informed decision.
    Evidence: {abstract}
    Claim: {claim}
    Provide the final answer in a Python list format. 
    Let’s think step-by-step:"""

    pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False
    )
    
    messages = [
    { "role": "user", "content": prompt}
    ]
    output = pipe(messages)
    
    response = output[0]["generated_text"]

    prediction = extract_prediction(response)
    
    return response, prediction

def main():

    data = pd.read_pickle('/netscratch/abu/Shared-Tasks/ClimateCheck/data/claims/final_english_claims_reranked_msmarco.pkl')
    results = [None] * len(data)

    for index, row in tqdm(data.iterrows(), total=len(data)):
        claim = row["atomic_claim"]
        ms_marco_results = row["ms_marco_reranking"] 
        abstracts_list = [item[2] for item in ms_marco_results]
        abstract_indices_list = [item[0] for item in ms_marco_results]
        evidentiary_abstracts = []

        for idx, abstract in enumerate(abstracts_list):
            rank = idx # lower index in list means higher rank from ms marco
            
            votes = {"supports": 0, "refutes": 0, "not enough information": 0, "unknown": 0}
            
            for model_name, model_type in models_info.items():
                
                print(f"Processing {model_name}...")
                
                tokenizer = AutoTokenizer.from_pretrained(model_name)

                if model_type == "sequence_classification":
                    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
                    pred = process_sequence_classification(tokenizer, model, claim, abstract)
                elif model_type == "causal_lm":
                    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
                    response, pred = process_causal_lm(tokenizer, model, claim, abstract)
                else:
                    raise ValueError("Unsupported model type")
                try:    
                    votes[pred] += 1
                except: 
                    votes["unknown"] +=1

                # Free up memory
                del model  

            # Check if abstract qualifies as evidentiary
            if votes["supports"] + votes["refutes"] >= 4:
                evidentiary_abstracts.append({"rank": rank, "idx": idx, "abstract": abstract, "votes": votes})

        # Rank top 3 abstracts by order in the list (ascending rank)
        evidentiary_abstracts.sort(key=lambda x: x["rank"])
        top_3_abstracts = evidentiary_abstracts[:3]

        results[index] = top_3_abstracts
        
        data['top_3_abstracts'] = results
        
        data.to_pickle('/netscratch/abu/Shared-Tasks/ClimateCheck/data/claims/final_english_claims_pooling.pkl')

        print(f'Saved data at index: {index}...')

if __name__ == "__main__":
    main()
