import argparse
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def rerank_with_cross_encoder(claim, abstracts, abstract_inds, model, tokenizer, top_k=50):
    """
    Re-ranks a list of abstracts for a given claim using a cross-encoder model.

    Parameters:
        claim (str): The claim text.
        abstracts (list): A list of abstract texts.
        model_name (str): The name of the cross-encoder model.
        top_k (int): The number of top-ranked abstracts to return.

    Returns:
        list: A list of tuples containing the top-k abstracts and their scores, sorted by relevance.
    """

    claim_texts = [claim] * len(abstracts)
    inputs = tokenizer(claim_texts, abstracts, padding=True, truncation=True, return_tensors="pt", max_length=512)

    batch_size = 32
    scores = []
    for i in range(0, len(abstracts), batch_size):
      batch_abstracts = abstracts[i:i+batch_size]
      inputs = tokenizer([claim] * len(batch_abstracts), batch_abstracts, padding=True, truncation=True, return_tensors="pt", max_length=512)
      inputs = {k: v.to('cuda') for k, v in inputs.items()}
      with torch.no_grad():
        outputs = model(**inputs)
        scores.extend(outputs.logits.squeeze().tolist())

    scored_abstracts = list(zip(abstract_inds, scores, abstracts))
    scored_abstracts.sort(key=lambda x: x[1], reverse=True)

    return scored_abstracts[:top_k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims_path", type=str, help="Path for claims dataset (pickle, including 'atomic_claim' column)")
    parser.add_argument("--output_path", type=str, help="Path for output")

    args = parser.parse_args()

    claims_path = args.claims_path
    output_path = args.output_path

    claims_df = pd.read_pickle(claims_path)

    model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to('cuda')

    reranked_abstracts = []
    for index, row in tqdm(claims_df.iterrows(), desc='Reranking abstracts'):
        r_abstract_inds = []
        r_abstracts = []
        for item in row['bm25_results']:
            r_abstract_inds.append(item[0])
            r_abstracts.append(item[2])

        row_top_abstracts = rerank_with_cross_encoder(row['atomic_claim'], r_abstracts, r_abstract_inds, model, tokenizer, top_k=50)
        reranked_abstracts.append(row_top_abstracts)
    
    claims_df['ms_marco_reranking'] = reranked_abstracts
    claims_df.to_pickle(output_path)
    print("Finished execution")