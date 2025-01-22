import json
from tqdm import tqdm
import pandas as pd
import numpy as np

def get_abstract_metadata(parquet_file, abs_original_index: str, pub_path: str):
    """
    Efficiently fetch metadata for a specific abstract index.
    Inputs:
    - abs_original_index: the index in the publication corpus to retrieve.
    - pub_path: the path of the publications corpus (Parquet format).
    Output:
    - A dictionary with metadata: { 'doi': doi, 'title': title, 'url': url }
    """
    
    # Directly locate the row by index
    try:
        row = parquet_file.loc[abs_original_index]  # Use the index directly
    except KeyError:
        raise ValueError(f"No abstract found with index {abs_original_index}")
    
    return {
        'doi': row['doi'],
        'title': row['title'],
        'url': row['url']
    }


def main():

	"""
	Load annotation data.
	"""
	input_path = "/netscratch/abu/Shared-Tasks/ClimateCheck/data/annotation_data/annotation_data.json"
	with open(input_path, 'r') as f:
		annotation_data = json.load(f)

	"""
	Create dataframe where wach row is one claim-abstract pair.
	Columns: claim, claimID, abstract, abstract_original_index, abstract_msmarco_rank, abstract_votes
	"""
	claims = []
	claim_ids = []
	abstracts = []
	abstract_original_indexes = []
	abstract_msmarco_ranks = []
	abstract_votes = []

	for item in tqdm(annotation_data, desc='Creating claim-abstract dataframe'):
		for linked_abstract in item['linked_abstracts']:
			claims.append(item['claim'])
			claim_ids.append(item['claimID'])
			abstracts.append(linked_abstract['abstract'])
			abstract_original_indexes.append(linked_abstract['abstract_original_index'])
			abstract_msmarco_ranks.append(linked_abstract['msmarco_rank'])
			abstract_votes.append(linked_abstract['total_votes'])

	# create dataframe
	claim_abstract_pairs = pd.DataFrame({
		'claim': claims,
		'claimID': claim_ids,
		'abstract': abstracts,
		'abstract_original_index': abstract_original_indexes,
		'msmarco_rank': abstract_msmarco_ranks,
		'total_votes': abstract_votes})

	"""
	Create new IDs.
	The ID for each claim-abstract pair will consist of the index in the new dataframe.
	"""
	claim_abstract_pairs['pair_id'] = claim_abstract_pairs.index

	"""
	Load original abstracts data to add more info: title + authors + DOI. 
	"""
	publication_data_path = "/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/merged_publications_only_en_citations_v2.parquet"
	
	# Load the Parquet file
	parquet_file = pd.read_parquet(publication_data_path, engine="pyarrow")

	dois = []
	titles = []
	urls = []

	for idx, row in tqdm(claim_abstract_pairs.iterrows(), desc='Adding metadata of abstracts'):
		original_index = row['abstract_original_index']
		metadata = get_abstract_metadata(parquet_file, original_index, publication_data_path)
		dois.append(metadata['doi'])
		titles.append(metadata['title'])
		urls.append(metadata['url'])


	claim_abstract_pairs['abs_title'] = titles
	claim_abstract_pairs['abs_doi'] = dois
	claim_abstract_pairs['abs_url'] = urls

	"""
	Dump .txt files.
	"""
	txt_output_path = "/netscratch/abu/Shared-Tasks/ClimateCheck/data/annotation_data/annotation_txt_files"

	for idx, row in tqdm(claim_abstract_pairs.iterrows(), desc='Creating text files'):
		pair_id = row['pair_id']
		claim = row['claim']
		abstract = row['abstract']
		abs_title = row['abs_title']
		abs_url = row['abs_url']
		abs_doi = row['abs_doi']

		with open(f'{txt_output_path}/claim_abs_{pair_id}.txt', 'w') as f:
			f.write(f'Claim: {claim}' + '\n\n\n')
			f.write(f'Abstract: {abstract}' + '\n\n\n')
			f.write(f'Title: {abs_title}' + '\n\n\n')
			f.write(f'URL: {abs_url}' + '\n\n\n')
			f.write(f'DOI: {abs_doi}' + '\n\n\n')

	claim_abstract_pairs.to_pickle('/netscratch/abu/Shared-Tasks/ClimateCheck/data/annotation_data/annotation_dataframe.pkl')

if __name__ == "__main__":
	main()