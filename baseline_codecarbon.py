import os
import re
import pickle
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import string
import heapq
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForCausalLM,
    pipeline
)
from transformers.pipelines.pt_utils import KeyDataset
from collections import Counter

from codecarbon import track_emissions

nltk.download('punkt')
nltk.download('punkt_tab')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

prompt_instructions = """You are an expert claim verification assistant with vast knowledge of climate change , climate science , environmental science , physics , and energy science.
    Your task is to check if the Claim is correct according to the Evidence. Generate 'Supports' if the Claim is correct according to the Evidence, or 'Refutes' if the
    claim is incorrect or cannot be verified. Or 'Not enough information' if you there is not enough information in the evidence to make an informed decision. Only return the verification verdict."""

def preprocess(text):
	# Tokenize and remove punctuation
	text = text.lower()  # Lowercase the text
	tokens = word_tokenize(text)  # Tokenize
	tokens = [word for word in tokens if word not in string.punctuation]  # Remove punctuation
	return tokens

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

def rerank_with_cross_encoder(claim, abstracts, abstract_inds, model, tokenizer, top_k=10):
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


def retrieval(run_type="test", ):

	# Load the claims

	claims_train = load_dataset("rabuahmad/climatecheck", split=run_type)
	claims_df = pd.DataFrame(claims_train)
	claims_only = claims_df[['claim', 'claim_id']]
	claims_only = claims_only.drop_duplicates()

	print("[INFO] Testing claims loaded ...")

	pubs = load_dataset("rabuahmad/climatecheck_publications_corpus", split='train', columns=['abstract', 'abstract_id', 'abstract_lowered'])
	pubs_df = pd.DataFrame(pubs)

	print("[INFO] Publications dataset loaded ...")

	print("[STEP1] BM25 - getting top 1000 abstract per claim ...")

	claims_only['bm25_results'] = ''

	# Preprocess publications corpus
	chunk_size = 10_000
	tokenized_corpus = []
	abstract_index = []
	global_index = 0

	# Pre-tokenize and store corpus in chunks
	for idx, abstract in enumerate(pubs_df['abstract'].tolist()):
		tokenized_corpus.append(preprocess(abstract))
		abstract_index.append((global_index, abstract))  # Store index and original abstract
		global_index += 1

	# Initialize BM25 once with the entire preprocessed corpus
	bm25 = BM25Okapi(tokenized_corpus)

	# Prepare to store results
	top_abstracts = []
	bm25_results = []

	# Iterate through each claim
	for idx, row in tqdm(claims_only.iterrows(), desc='Processing Claims'):
		tokenized_query = preprocess(row['claim'])
		# Compute scores for the entire corpus
		scores = bm25.get_scores(tokenized_query)
		# Use a heap to keep track of the top 1000 scores
		top_1000 = heapq.nlargest(1000, zip(range(len(scores)), scores), key=lambda x: x[1])

		# Collect the top 1000 abstracts
	
		for i, score in top_1000:
		# for res in top_1000_abstracts:
			bm25_results.append({
				'claim': row['claim'],
				'claim_id': row['claim_id'],
				'abstract_id': abstract_index[i][0],
				'bm25_score': score,
				'abstract': abstract_index[i][1]
				})

	# Store results in the dataframe

	bm25_results_df = pd.DataFrame(bm25_results)

	print("[STEP2] Creating embeddings for publications ...")

	model = SentenceTransformer("sentence-transformers/msmarco-MiniLM-L-12-v3", device=device)
	model.max_seq_length = 512
	model.to(device)
	print(f"[INFO] Model loaded sentence-transformers/msmarco-MiniLM-L-12-v3")

	abstracts = pubs_df['abstract_lowered'].to_list()
	metadata = pubs_df['abstract_id'].to_list()

	print("[INFO] Start encoding abstracts")
	abstract_embeddings = []
	batch_size = 64
	for i in tqdm(range(0, len(abstracts), batch_size)):
		batch = abstracts[i:i+batch_size]
		batch_emb = model.encode(batch, device=device)
		abstract_embeddings.append(batch_emb)

	print("[INFO] Encoding finished")
	abstract_embeddings = np.vstack(abstract_embeddings)

	np.save("abstract_embeddings.csv", abstract_embeddings)
	print("[INFO] Embeddings saved")

	print("[STEP3] Cosine similarity - getting top 20 abstract per claim ...")
	
	predictions_df = predict(bm25_results_df, abstract_embeddings, model, 20)

	print("[STEP4] Reranker - getting top 10 abstract per claim ...")

	reranker_model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
	topN = 10

	print(f"[INFO] Reranking predictions with {reranker_model_name}")

	rerank_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
	rerank_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
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
	
	print("[INFO] Finished retrieval!")
      
	return reranked_df

def classification(retrieval_df, model_name="01-ai/Yi-1.5-9B-Chat-16K"):

	print("[STEP5] Getting label predictions per claim-abstract pair ...")
    
	retrieval_df['label'] = ""
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(model_name)
    
	print(f"[INFO] Start classification for # {retrieval_df.shape[0]} claim-abstract pairs")
	tokenizer.pad_token_id = model.config.eos_token_id
	pipe = pipeline("text-generation", 
				model=model,
				tokenizer=tokenizer,
				return_full_text=False,
				max_new_tokens=500,
				do_sample=False,
				trust_remote_code=True,
				truncation=True)
	
	class_res = []
      
	for idx, row in tqdm(retrieval_df.iterrows()):
            
		messages=[{"role":"system","content":prompt_instructions},
                      {"role":"user","content":f"Claim: {row['claim']} \nEvidence: {row['abstract']}"}]
        
		output = pipe(messages)
		class_res.append({
			'claim_id': row['claim_id'],
			'claim': row['claim'],
			'abstract_id': row['abstract_id'],
			'abstract': row['abstract'],
			'rank': row['rank'],
			'label': output[0]['generated_text'],
			}
		)
	
	class_df = pd.DataFrame(class_res)
	
	print("[INFO] Finished classification!")
	return class_df
            

@track_emissions(project_name="ClimateCheck2026", save_to_api=True)
def main():

	retrieval_results = retrieval()
	retrieval_results.to_csv("retrieval_results.csv")

	classification_results = classification(retrieval_results)
	classification_results.to_csv("classification_results.csv")

if __name__ == "__main__":
	
	main()

