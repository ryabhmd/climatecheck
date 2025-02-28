from datasets import load_from_disk, concatenate_datasets, Dataset
import pandas as pd
from tqdm import tqdm
import json
import argparse


# Read models info
with open('pooling_models.json', 'r') as f:
    models_info = json.load(f)

def load_datasets(path):
	"""
	Loads results of pooling and creates a dataframe of all results. 
	"""
	concatenated_datasets = Dataset.from_dict({})
	
	for model, _ in tqdm(models_info.items(), desc='Collecting data from models'):
		clean_name = model.split("/")[-1]
		try:
			data_top6 = load_from_disk(path + "/annotation_data_" + clean_name + "_top6.hf")
		except:
			data_top6 = Dataset.from_dict({})
			print(f'{clean_name} top 6 was not found. Initiated empty dataset.')

		try:	
			data_top6_10 = load_from_disk(path + "/annotation_data_" + clean_name + "_top6-10.hf")
		except:
			data_top6_10 = Dataset.from_dict({})
			print(f'{clean_name} top 6-10 was not found. Initiated empty dataset.')

		try:	
			data_top10_20 = load_from_disk(path + "/annotation_data_" + clean_name + "_top10-20.hf")
		except:
			data_top10_20 = Dataset.from_dict({})
			print(f'{clean_name} top 10-20 was not found. Initiated empty dataset.')

		concatenated_datasets = concatenate_datasets([concatenated_datasets,
							      data_top6,
                            data_top6_10,
                            data_top10_20])

	df = concatenated_datasets.to_pandas()

	return df


def correct_ranks(annotation_data, original_msmarco_path):
	"""
	Correct the msmarco rank for each abstract according to the original data (a .pkl file with the original rankings).
	This function was added due to an error in the pooling code which did not correctly save the msmarco rankings.
	It returns the annotation data with the correct rankings.  
 	The function also adds the claimIDs.
	"""
	original_data = pd.read_pickle(original_msmarco_path)

	for item in tqdm(annotation_data, desc='Correcting ranks'):
		claim = item['claim']
		linked_abstracts = item['linked_abstracts']

		for abs_idx, linked_abstract in enumerate(linked_abstracts):
			for idx, row in original_data.iterrows():
				if claim == row['atomic_claim']:
					# Add info about claimID
					item['claimID'] = row['claimID']
					msmarco_results = row['ms_marco_reranking']
					for msmarco_rank, msmarco_result in enumerate(msmarco_results):
						if msmarco_result[2] == linked_abstract['abstract']:
							item['linked_abstracts'][abs_idx]['msmarco_rank'] = msmarco_rank

	return annotation_data

def create_annotation_data(concatenated_dataframe):
	"""
	Gets the dataframe of pooling results as input and creates a new dataframe with annotation data. 
	"""
	all_claims = concatenated_dataframe['claim'].unique()

	# Initiate empty annotation data where each item is one claim
	annotation_data = [{'claim': claim,
						'claimID': None,
						'linked_abstracts': []} for claim in all_claims]


	for item in tqdm(annotation_data, desc='Creating annotation data'):
		claim = item['claim']
		for idx, row in concatenated_dataframe.iterrows():
			if claim == row['claim']:
				abstract = row['abstract']
				abs_original_idx = row['abstract_original_index']
				msmarco_rank = row['msmarco_rank']
				total_votes = row['total_votes']
				item['linked_abstracts'].append({'abstract': abstract, 'abstract_original_index': abs_original_idx, 'msmarco_rank': msmarco_rank, 'total_votes': total_votes})

	annotation_data = correct_ranks(annotation_data, "../final_english_claims_reranked_msmarco.pkl")

	return annotation_data 

def main():

	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

	parser.add_argument(
        "--pooling_path",
        "-m",
        type=str,
        help="Path to directory where pooling results are saved.",
	)

	parser.add_argument(
        "--output_path",
        "-m",
        type=str,
        help="Path to save the annotation data.",
	)

	parser.add_argument(
        "--top_n",
        "-m",
        type=str,
	default=5,
        help="Name of model on HF.",
	)

	args = parser.parse_args()
	pooling_path = args.pooling_path
	output_path = args.output_path
	top_n = args.top_n
	
	data = load_datasets(pooling_path)
	annotation_data = create_annotation_data(data)

	# Get top_n linked abstract for each claim
	for item in annotation_data:
		item['linked_abstracts'] = sorted(item['linked_abstracts'], key=lambda x: x['msmarco_rank'])[:top_n]

	print(f'Created annotation data with {len(annotation_data)} claims.')

	with open(output_path + '/annotation_data.json', 'w') as f:
		json.dump(annotation_data, f, indent=4)

	print(f'Saved annotation data at {output_path}/annotation_data.json.')

	# Count how many claims have less than n abstracts
	claims_with_less_than_n_abstracts = []
	for item in annotation_data:
		if len(item['linked_abstracts']) < top_n:
			claims_with_less_than_n_abstracts.append(item)

	print(f'{len(claims_with_less_than_n_abstracts)} claims have less than n abstracts.')

	with open(output_path + '/claims_with_less_than_n_abstracts.json', 'w') as f:
		json.dump(claims_with_less_than_n_abstracts, f, indent=4)

	print(f'Saved claims with less than {n} abstracts at {output_path}/claims_with_less_than_n_abstracts.json.')


if __name__ == "__main__":
	main()
	




