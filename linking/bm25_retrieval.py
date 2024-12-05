from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import string
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
from tqdm import tqdm
import heapq
import nltk
import argparse
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

nltk.download('punkt')
nltk.download('punkt_tab')

def preprocess(text):
    # Tokenize and remove punctuation; add more preprocessing? (stop word removal, lemmatization, etc.?)
    text = text.lower()  # Lowercase the text
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word not in string.punctuation]  # Remove punctuation
    return tokens

def translate_claims_german_to_english(claims, tokenizer, model):
    """
    Function to translate a list of German claims to English
    """
    # Load the MBart model and tokenizer for translation
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    tokenizer.src_lang = "de_DE"
    target_lang = "en_XX"

    translated_claims = []
    for claim in tqdm(claims, desc='Translating Claims'):
        # Tokenize the German text
        inputs = tokenizer(claim, return_tensors="pt", max_length=512, truncation=True)
        # Generate translation
        translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])
        # Decode the translated text
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        translated_claims.append(translated_text)
    return translated_claims

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, help="Language of Claims")
    parser.add_argument("--claims_path", type=str, help="Path for claims dataset (pickle, including 'atomic_claim' column)")
    parser.add_argument("--pub_path", type=str, help="Path for Publications dataset (parquet)")
    parser.add_argument("--output_path", type=str, help="Path for output")

    args = parser.parse_args()

    lang = args.lang
    claims_path = args.claims_path
    pub_path = args.pub_path
    output_path = args.output_path

    # Load the claims
    claims = pd.read_pickle(claims_path)
    queries = claims['atomic_claim'].tolist()
    
    # Load publications as a ParquetFile
    parquet_file = pq.ParquetFile(pub_path)
    
    # Preprocess publications corpus
    chunk_size = 10_000
    tokenized_corpus = []
    abstract_index = []
    global_index = 0
    
    # Pre-tokenize and store corpus in chunks
    for batch in tqdm(parquet_file.iter_batches(batch_size=chunk_size), desc='Preprocessing Corpus'):
        pubs = batch.to_pandas()
        for idx, abstract in enumerate(pubs['abstract'].tolist()):
            tokenized_corpus.append(preprocess(abstract))
            abstract_index.append((global_index, abstract))  # Store index and original abstract
            global_index += 1
            
    # Initialize BM25 once with the entire preprocessed corpus
    bm25 = BM25Okapi(tokenized_corpus)

    if lang == 'de':
        queries = translate_claims_german_to_english(queries)
    
    # Prepare to store results
    top_abstracts = []
    
    # Iterate through each claim
    for query in tqdm(queries, desc='Processing Claims'):
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
    claims.to_pickle(output_path)
       
if __name__ == "__main__":
    main()




    

