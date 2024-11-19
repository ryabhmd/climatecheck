from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import string
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
from tqdm import tqdm
import heapq
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

def preprocess(text):
    # Tokenize and remove punctuation; add more preprocessing? (stop word removal, lemmatization, etc.?)
    text = text.lower()  # Lowercase the text
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word not in string.punctuation]  # Remove punctuation
    return tokens

def main():

    # Load the claims
    claims = pd.read_pickle('/netscratch/abu/Shared-Tasks/ClimateCheck/data/claims/final_english_claims_reduced.pkl')
    en_queries = claims['atomic_claim'].tolist()
    
    # Load publications as a ParquetFile
    parquet_file = pq.ParquetFile('/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/merged_publications_only_en.parquet')
    
    # Preprocess publications corpus
    chunk_size = 10_000
    tokenized_corpus = []
    abstract_index = []
    
    # Pre-tokenize and store corpus in chunks
    for batch in tqdm(parquet_file.iter_batches(batch_size=chunk_size), desc='Preprocessing Corpus'):
        pubs = batch.to_pandas()
        for idx, abstract in enumerate(pubs['abstract'].tolist()):
            tokenized_corpus.append(preprocess(abstract))
            abstract_index.append((batch.offset + idx, abstract))  # Store index and original abstract
            
    # Initialize BM25 once with the entire preprocessed corpus
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Prepare to store results
    top_abstracts = []
    
    # Iterate through each claim
    for query in tqdm(en_queries, desc='Processing Claims'):
        tokenized_query = preprocess(query)
        # Compute scores for the entire corpus
        scores = bm25.get_scores(tokenized_query)
        
        # Use a heap to keep track of the top 1000 scores
        top_1000 = heapq.nlargest(1000, zip(range(len(scores)), scores), key=lambda x: x[1])
        
        # Collect the top 1000 abstracts
        top_1000_abstracts = [(abstract_index[i][0], score, abstract_index[i][1]) for i, score in top_1000]
        
        top_abstracts.append(top_1000_abstracts)
        
    # Store results in the dataframe
    claims['bm25_results'] = top_abstracts
    
    # Save the updated claims dataframe
    claims.to_pickle('/netscratch/abu/Shared-Tasks/ClimateCheck/data/claims/final_english_claims_reduced_bm25.pkl')
       
if __name__ == "__main__":
    main()




    

