import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForCausalLM, 
    pipeline
)
import json
import pandas as pd
import re

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Define the models and their respective types
models_info = {
    "microsoft/deberta-v2-xxlarge-mnli": "sequence_classification",
    "joeddav/xlm-roberta-large-xnli": "sequence_classification",
    "FacebookAI/roberta-large-mnli": "sequence_classification",
    "facebook/bart-large-mnli": "sequence_classification",
    "Qwen/Qwen2.5-7B-Instruct": "causal_lm",
    #"google/gemma-2-9b-it": "causal_lm", waiting for access
    "meta-llama/Llama-3.1-8B-Instruct": "causal_lm",
    "mistralai/Mistral-Nemo-Instruct-2407": "causal_lm",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "causal_lm",
}

# Function to process sequence classification models
def process_sequence_classification(model_name, tokenizer, model, claim, abstract):
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
    # Define a regex pattern to capture the prediction
    pattern = r"\['(Supports|Refutes|Not Enough Information)'\]"
    
    # Search for the prediction in the text
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        # Return the matched label in lowercase
        return match.group(1).lower()
    else:
        # Return 'unknown' if no match is found
        return "unknown"

# Function to process causal language models
def process_causal_lm(model, tokenizer, claim, abstract):
    prompt = f"""You are an expert claim verification assistant.
    Verify the claim against the provided abstract:
    Claim: "{claim}"
    Abstract: "{abstract}"
    Does the abstract:
    - Support the claim
    - Refute the claim
    - Not provide enough information?
    Answer clearly with one of the following: "Supports", "Refutes", or "Not Enough Information".
    Return your answer in a python list format."""
    
    messages = [{"role": "user", "content": prompt}]
    input_text=tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs, 
        max_new_tokens=20, 
        temperature=0.1, 
        top_p=1, 
        do_sample=True, 
        eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    prediction = extract_prediction(response)
    
    return response, prediction

def main():

    data = pd.read_pickle('/netscratch/abu/Shared-Tasks/ClimateCheck/data/claims/pooling_test.pkl')

    predictions = {}
    
    for model_name, model_type in models_info.items():
        print(f"Processing {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if model_type == "sequence_classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        elif model_type == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct").to(device)
        else:
            raise ValueError("Unsupported model type")
        
        
        model_predictions = []
        for idx, row in data.iterrows(): 
            claim = row["atomic_claim"]
            abstracts = [item[0] for item in row['reranking_results'][:5]]
            for abstract in abstracts:
                if model_type == "sequence_classification":
                    pred = process_sequence_classification(model_name, tokenizer, model, claim, abstract)
                    model_predictions.append({"claim": claim, "abstract": abstract, "prediction": pred})
                elif model_type == "causal_lm":
                    response, pred = process_causal_lm(model, tokenizer, claim, abstract)
                    model_predictions.append({"claim": claim, "abstract": abstract, "response": response, "prediction": pred})
    
        predictions[model_name] = model_predictions
        try:
            del model  # Free up memory
        except:
            del text_gen_pipeline
    
        # Save predictions to JSON
        with open("model_predictions.json", "w") as f:
            json.dump(predictions, f, indent=4)
        
        print("Predictions saved to 'model_predictions.json'")

if __name__ == "__main__":
    main()
