import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import json
import pandas as pd

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Define the models and their respective types
models_info = {
    "microsoft/deberta-v2-xxlarge-mnli": "sequence_classification",
    "joeddav/xlm-roberta-large-xnli": "sequence_classification",
    "FacebookAI/roberta-large-mnli": "sequence_classification",
    "openlm-research/open_llama_13b": "causal_lm",
    "bigscience/bloom-3b": "causal_lm",
    "meta-llama/Llama-2-7b-hf": "causal_lm",
    "HuggingFaceTB/SmolLM-360M": "causal_lm",
    "microsoft/Phi-3-mini-4k-instruct": "causal_lm",
    "google/mt5-xl": "seq2seq",
}

# Mapping from logits indices to labels
label_mapping = {0: "Refutes", 1: "Not Enough Information", 2: "Supports"}

# Function to process sequence classification models
def process_sequence_classification(model_name, tokenizer, model, claim, abstract):
    input_text = f"{claim} </s></s> {abstract}" if "roberta" in model_name or "xlm-roberta" in model_name else f"{claim} [SEP] {abstract}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return label_mapping[prediction]

# Function to process causal language models
def process_causal_lm(model_name, tokenizer, model, claim, abstract):
    prompt = f"""You are a fact-checking assistant. Verify the claim against the provided abstract:
Claim: {claim}
Abstract: {abstract}
Does the abstract support, refute, or not provide enough information about the claim? Answer with "Supports", "Refutes", or "Not Enough Information".
Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=600, temperature=0.7, top_p=0.9)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to process seq2seq models
def process_seq2seq(model_name, tokenizer, model, claim, abstract):
    input_text = f"Claim: {claim} Abstract: {abstract} Predict the relationship: Supports, Refutes, or Not Enough Information."
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50, temperature=0.7, top_p=0.9)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

def main():

    data = pd.read_pickle('/netscratch/abu/Shared-Tasks/ClimateCheck/data/claims/pooling_test.pkl')

    predictions = {}
    
    for model_name, model_type in models_info.items():
        print(f"Processing {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if model_type == "sequence_classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        elif model_type == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
        elif model_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            raise ValueError("Unsupported model type")
        
        
        model_predictions = []
        for idx, row in data.iterrows(): 
            claim = row["atomic_claim"]
            abstracts = [item[0] for item in row['reranking_results'][:5]]
            for abstract in abstracts:
                if model_type == "sequence_classification":
                    model.to(device)
                    pred = process_sequence_classification(model_name, tokenizer, model, claim, abstract)
                elif model_type == "causal_lm":
                    pred = process_causal_lm(model_name, tokenizer, model, claim, abstract)
                elif model_type == "seq2seq":
                    model.to(device)
                    pred = process_seq2seq(model_name, tokenizer, model, claim, abstract)
                model_predictions.append({"claim": claim, "abstract": abstract, "prediction": pred})
    
        predictions[model_name] = model_predictions
        del model  # Free up memory
    
        # Save predictions to JSON
        with open("model_predictions.json", "w") as f:
            json.dump(predictions, f, indent=4)
        
        print("Predictions saved to 'model_predictions.json'")

if __name__ == "__main__":
    main()
