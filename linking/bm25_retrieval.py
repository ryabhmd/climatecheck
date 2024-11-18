from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import string
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
from tqdm import tqdm
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

    claims = pd.read_pickle('/netscratch/abu/Shared-Tasks/ClimateCheck/data/claims/final_english_claims_reduced.pkl')
    en_queries = claims['atomic_claim'].tolist()

    chunk_size = 10_000
    parquet_file = pq.ParquetFile('/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/merged_publications_only_en.parquet')

    top_abstracts = []
    for query in tqdm(en_queries):
        tokenized_query = preprocess(query)
        all_scores = []

        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            # Process each batch, converting it to a pandas dataframe if necessary
            pubs = batch.to_pandas()
            pubs_corpus = pubs['abstract'].tolist()
            tokenized_corpus = [preprocess(doc) for doc in pubs_corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            # save all scores for a query
            scores = bm25.get_scores(tokenized_test_query)
            scores_abs = [(i, scores[i], pubs_corpus[i]) for i in range(len(scores))]
            all_scores.extend(scores_abs)
        
        top_1000_indices = sorted(range(len(all_scores)), key=lambda x: all_scores[x][1], reverse=True)[:1000]
        top_1000_abstracts = [all_scores[i] for i in top_1000_indices]

        top_abstracts.append(top_1000_abstracts)

    claims['bm25_results'] = top_abstracts

    claims.to_pickle('final_english_claims_reduced_bm25.pkl')
       
if __name__ == "__main__":
    main()




    

