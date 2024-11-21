import pandas as pd
import zipfile
import os
import langdetect
from langdetect import detect, detect_langs
import pyalex
from pyalex import Works, config
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np

CHUNK_SIZE = 10000
citation_cache = {}

def merge(open_alex_publlications, s2orc_publications_deduped):

	s2orc_publications_deduped.rename(columns=
		{
		'id': 's2orc_id',
		'url': 's2orc_url',
		'openAccessPdf': 'url',
		'fieldsOfStudy': 's2orc_fieldsOfStudy',
		'citationCount': 'citation_count'
		},
		inplace=True)
	
	s2orc_publications_deduped['source'] = 's2orc'

	s2orc_publications_deduped['url'] = [item['url'] for item in s2orc_publications_deduped['url'].values]

	open_alex_publlications.rename(columns=
		{
		'id': 'openalex_id',
		'oa_url': 'url',
		'topics': 'openalex_topics',
		'keywords': 'openalex_keywords',
		'concepts': 'openalex_concepts',
		'source_topic_id': 'openalex_topic_id'
		},
		inplace=True)

	open_alex_publlications['source'] = 'OpenAlex'

	merged_df = pd.concat([s2orc_publications_deduped, open_alex_publlications], ignore_index=True)
	
	# remove rows from merged_df that don't have a value in 'title'
	merged_df = merged_df.dropna(subset=['title'])
	merged_df = merged_df.dropna(subset=['abstract'])
	merged_df = merged_df.dropna(subset=['url'])

	# deduplicate rows based on doi, title (lowered), and abstract (lowered)
	merged_df['title_lowered'] = [title.lower() for title in merged_df['title'].tolist()]
	merged_df['abstract_lowered'] = [abstract.lower() for abstract in merged_df['abstract'].tolist()]

	merged_df = merged_df.drop_duplicates(subset=['doi'])
	merged_df = merged_df.drop_duplicates(subset=['title_lowered'])
	merged_df = merged_df.drop_duplicates(subset=['abstract_lowered'])

	# remove rows with 'title' of less than 10 chars
	merged_df = merged_df[merged_df['title'].str.len() > 10]
	merged_df = merged_df[merged_df['abstract'].str.len() > 30]

	return merged_df

def fetch_citation_count(row):
    """Fetch citation count for a single row."""
    doi = row['doi']
    
    # Check cache first
    if doi in citation_cache:
        return citation_cache[doi]

    # Fetch from OpenAlex API
    if row['source'] == 'OpenAlex':
        try:
            citation_count = Works()[doi]['cited_by_count']
        except Exception as e:
            print(f"Error fetching citation for DOI {doi}: {e}")
            citation_count = None
    else:
        citation_count = row['citation_count']

    # Cache the result
    citation_cache[doi] = citation_count
    return citation_count

def process_chunk(chunk):
    """Process a chunk of the DataFrame."""
    with ThreadPoolExecutor(max_workers=8) as executor:
        citation_counts = list(
            tqdm(
                executor.map(fetch_citation_count, [row for _, row in chunk.iterrows()]),
                total=chunk.shape[0],
                desc="Fetching citation counts",
            )
        )
    chunk['citation_count'] = citation_counts
    return chunk

def chunk_dataframe(df, chunk_size):
    """Split DataFrame into chunks."""
    num_chunks = int(np.ceil(len(df) / chunk_size))
    for i in range(num_chunks):
        yield df[i * chunk_size:(i + 1) * chunk_size]

def filter_citations(file_path):
    """Filter Parquet file in manageable chunks."""
    # Load full Parquet file
    full_df = pd.read_parquet(file_path, engine="pyarrow")
    
    filtered_rows = []
    for i, chunk in enumerate(chunk_dataframe(full_df, CHUNK_SIZE)):
        print(f"Processing chunk {i+1}/{len(full_df)//CHUNK_SIZE + 1}...")
        processed_chunk = process_chunk(chunk)
        # Apply filtering condition (e.g., citation_count >= 10)
        filtered_chunk = processed_chunk[processed_chunk['citation_count'] >= 10]
        filtered_rows.append(filtered_chunk)

    # Combine all filtered chunks
    final_df = pd.concat(filtered_rows, ignore_index=True)
    return final_df

def filter_non_en(merged_df):

	non_en_indices = []
	for index, row in tqdm(merged_df.iterrows(), total=merged_df.shape[0], desc='Filtering non EN'):
		abstract = row['abstract']
		title = row['title']
		# Detect language of abstract using langdetect
		try:
			abstract_lang = detect(abstract)
			title_lang = detect(title)
		except langdetect.lang_detect_exception.LangDetectException:
			print(f"Error detecting language for abstract at index {index}")
			non_en_indices.append(index)
			continue
		if abstract_lang != 'en' or title_lang != 'en':
			non_en_indices.append(index)

	filtered_df = merged_df.drop(non_en_indices)

	return filtered_df



def main():

	s2orc_publications = pd.read_pickle('/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/s2orc_publications_v3_citations.pkl')
	s2orc_publications['doi'] = 'https://doi.org/' + s2orc_publications['doi']

	open_alex_publlications = pd.read_pickle('/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/combined_climate_works_filtered.pkl')
	
	print(f'Number of S2ORC publications: {len(s2orc_publications)}.')
	print(f'Number of OpenAlex publications: {len(open_alex_publlications)}.')

	s2orc_publications_deduped = s2orc_publications[~s2orc_publications['doi'].isin(open_alex_publlications['doi'])]
	print(f'Number of S2ORC publications after deduplication: {len(s2orc_publications_deduped)}.')

	s2orc_publications_deduped.to_pickle('/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/s2orc_publications_v3_citations_deduplicated.pkl')

	merged_df = merge(open_alex_publlications, s2orc_publications_deduped)
	merged_df.to_parquet('/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/merged_publications_v2.parquet')
	print(f'Number of merged publications: {len(merged_df)}.')

	# drop non EN rows (titles or abstracts)
	filtered_df = filter_non_en(merged_df)
	filtered_df.to_parquet('/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/merged_publications_only_en_v2.parquet')
	print(f'Number of merged publications after removing non English ones: {len(filtered_df)}.')
	
	# drop rows with less than 10 citations
	filtered_df = filter_citations('/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/merged_publications_only_en_v2.parquet')
	filtered_df.to_parquet(
    '/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/merged_publications_only_en_citations_v2.parquet')
	print(f"Number of merged publications after removing those with less than 10 citations: {len(filtered_df)}.")

if __name__ == "__main__":
    main()




