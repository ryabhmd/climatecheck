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
    "facebook/bart-large-mnli": "sequence_classification",
    "Qwen/Qwen2.5-7B": "causal_lm",
    "google/gemma-2-9b": "causal_lm",
    "meta-llama/Llama-3.1-8B": "causal_lm",
    "mistralai/Mistral-Nemo-Base-2407": "causal_lm",
    "intfloat/e5-mistral-7b-instruct": "causal_lm",
    "HuggingFaceTB/SmolLM2-1.7B": "causal_lm"
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
        
    labels = ["contradiction", "neutral", "entailment"]
    prediction = labels[torch.argmax(logits, dim=-1).item()]

    return prediction

# Function to process causal language models
def process_causal_lm(model_name, claim, abstract):
   prompt = (
        f"Claim: {claim}\n"
        f"Abstract: {abstract}\n\n"
        "Does the abstract support, refute, or provide no information about the claim? "
        "Answer with one of the following words: supports, refutes, not enough info."
    )

    text_gen_pipeline = pipeline("text-generation", model=model_name, device=0)

    output = text_gen_pipeline(prompt, max_length=50, num_return_sequences=1, do_sample=True)
    response = output[0]["generated_text"].lower()

    return response

def main():

    data = pd.read_pickle('/netscratch/abu/Shared-Tasks/ClimateCheck/data/claims/pooling_test.pkl')

    predictions = {}
    
    for model_name, model_type in models_info.items():
        print(f"Processing {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if model_type == "sequence_classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        elif model_type == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True, torch_dtype=torch.float16)
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
                    pred = process_causal_lm(model_name, claim, abstract)
                model_predictions.append({"claim": claim, "abstract": abstract, "prediction": pred})
    
        predictions[model_name] = model_predictions
        del model  # Free up memory
    
        # Save predictions to JSON
        with open("model_predictions.json", "w") as f:
            json.dump(predictions, f, indent=4)
        
        print("Predictions saved to 'model_predictions.json'")

if __name__ == "__main__":
    main()
