import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict(df, abstract_embeddings, model, topN=20):
  output = []

  for claim_id in tqdm(df['claim_id'].unique().tolist()):
    claim = df[df.claim_id == claim_id]['claim'].values[0]
    query_embedding = model.encode([claim], convert_to_numpy=True)

    abstract_ids = df[df.claim_id == claim_id]['abstract_id'].tolist()
    abstracts = df[df.claim_id == claim_id]['abstract'].tolist()
    selected_abstract_embeddings = [abstract_embeddings[abstract_id] for abstract_id in abstract_ids]
    
    scores = cosine_similarity(query_embedding, selected_abstract_embeddings)[0]
    top_k_indices = np.argsort(scores)[::-1][:topN] 
    # top_k_results = [(abstract_ids[i], abstracts[i]) for i in top_k_indices]

    for i in top_k_indices:
      output.append({
        "claim_id": claim_id, 
        "claim": claim,
        "abstract_id": abstract_ids[i],
        "abstract": abstracts[i],
        'bm25_score': float("%0.2f" %(df[(df.claim_id == claim_id) & (df.abstract_id == abstract_ids[i])]['bm25_score'].values[0])),
        'cosine_sim': float("%0.2f" % (scores[i]*100))
        })
  
  return pd.DataFrame(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--topN', type=str)
    parser.add_argument('--type', type=str)
    args = parser.parse_args()

    topN = args.topN
    model_name = args.model
    run_type = args.type
    input_file = args.input

    # claims_test = load_dataset("rabuahmad/climatecheck", split='test')
    # pubs = load_dataset("rabuahmad/climatecheck_publications_corpus")
    file = 'msmarco'
    if model_name == 'sentence-transformers/msmarco-MiniLM-L-12-v3':
       file = 'msmarco'
    elif model_name == 'malteos/scincl':
       file = 'scincl'
    elif model_name == 'malteos/scincl':
       file = 'bge-base'

    model = SentenceTransformer(model_name, device='cuda')

   #  claims_train = load_dataset("rabuahmad/climatecheck", split=run_type)
   #  claims_df = pd.DataFrame(claims_train)
    claims_df = pd.read_csv(input_file)

    abstract_embeddings = np.load(f"data/abstract_embeddings_{file}.npy")

    predictions_df = predict(claims_df, abstract_embeddings, model, 20)

    predictions_df.to_csv(f"/storage/usmanova/climatecheck/data/test/predictions_{run_type}_{file}_top{topN}.csv")