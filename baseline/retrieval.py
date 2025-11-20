import argparse
import string
import heapq
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

def preprocess(text):
    # Tokenize and remove punctuation
    text = text.lower()  # Lowercase the text
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word not in string.punctuation]  # Remove punctuation
    return tokens


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, help="Run type")
    parser.add_argument("--output_path", type=str, help="Path for output")

    args = parser.parse_args()
    run_type = args.type
    output_path = args.output_path

    # Load the claims
    claims_train = load_dataset("rabuahmad/climatecheck", split=run_type)
    claims_df = pd.DataFrame(claims_train)
    claims_only = claims_df[['claim', 'claim_id']]
    claims_only = claims_only.drop_duplicates()
    claims_only['bm25_results'] = ''

    # queries = claims_df['claim'].unique().tolist()
    pubs = load_dataset("rabuahmad/climatecheck_publications_corpus", split='train', columns=['abstract', 'abstract_id'])
    pubs_df = pd.DataFrame(pubs)
    
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
        # top_1000_abstracts = [(abstract_index[i][0], score, abstract_index[i][1]) for i, score in top_1000]
        for i, score in top_1000:
        # for res in top_1000_abstracts:
            bm25_results.append({
                'claim': row['claim'],
                'claim_id': row['claim_id'],
                'abstract_id': abstract_index[i][0],
                'bm25_score': score,
                'abstract': abstract_index[i][1]
            })
        # claims_only.at[idx, 'bm25_results'] = top_1000_abstracts
        # top_abstracts.append(top_1000_abstracts)
        
    # Store results in the dataframe
    # claims_df['bm25_results'] = top_abstracts
    bm25_results_df = pd.DataFrame(bm25_results)
    
    # Save the updated claims dataframe
    bm25_results_df.to_csv(output_path)
