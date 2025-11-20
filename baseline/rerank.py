import time
import json
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def rerank_with_cross_encoder(claim, abstracts, abstract_inds, model, tokenizer, top_k=5):
    claim_texts = [claim] * len(abstracts)
    inputs = tokenizer(claim_texts, abstracts, padding=True, truncation=True, return_tensors="pt", max_length=512)

    batch_size = 32
    scores = []
    for i in range(0, len(abstracts), batch_size):
      batch_abstracts = abstracts[i:i+batch_size]
      inputs = tokenizer([claim] * len(batch_abstracts), batch_abstracts, padding=True, truncation=True, return_tensors="pt", max_length=512)
      inputs = {k: v.to(device) for k, v in inputs.items()}
      with torch.no_grad():
        outputs = model(**inputs)
        scores.extend(outputs.logits.squeeze().tolist())

    scored_abstracts = list(zip(abstract_inds, scores, abstracts))
    scored_abstracts.sort(key=lambda x: x[1], reverse=True)

    return scored_abstracts[:top_k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--topN', type=int)
    args = parser.parse_args()

    model_name = args.model
    topN = args.topN

    print(f"[INFO] Reranking predictions with {model_name}")

    predictions_df = pd.read_csv(args.input)

    rerank_tokenizer = AutoTokenizer.from_pretrained(model_name)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    rerank_model.to(device)

    prediction_groups = predictions_df.groupby(['claim_id', 'claim']).agg({'abstract':lambda x: list(x), 'abstract_id':lambda x: list(x), 'bm25_score':lambda x: list(x), 'cosine_sim':lambda x: list(x)}).reset_index()
    print(f"[INFO] predictions {prediction_groups.shape}")

    reranked_res = []
    for idx, row in tqdm(prediction_groups.iterrows()):
        claim = row['claim']
        abstract_ids = row['abstract_id']
        abstracts = row['abstract']

        new_abstracts = rerank_with_cross_encoder(claim, abstracts, abstract_ids, rerank_model, rerank_tokenizer, topN)
        rank = 1
        for pred in new_abstracts:
            reranked_res.append({
                "claim_id": row['claim_id'],
                "claim": row['claim'],
                "abstract_id": pred[0],
                "abstract": pred[2],
                "rerank_score": pred[1],
                "rank": rank
                }
            )
            rank += 1

    reranked_df = pd.DataFrame(reranked_res)
    reranked_df.to_csv(args.output)
    print("Finished execution")