import pandas as pd
from datasets import Dataset


def prepare_data(input_path, output_path):
    """
    Prepares data for claim-abstract pairs and saves it as a Hugging Face dataset.

    Args:
        input_path (str): Path to the input pickle file containing claims and ranked abstracts.
        output_path (str): Path to save the Hugging Face dataset.
    """

    # Load the input pickle file
    data = pd.read_pickle(input_path)
    
    # Expand the dataset into claim-abstract pairs
    expanded_data = []
    for idx, row in data.iterrows():
        claim_id = row["claimID"]
        claim = row["atomic_claim"]
        original_idx = idx
        ms_marco_results = row["ms_marco_reranking"][10:20]

        for rank, (abs_idx, msmarco_score, abstract) in enumerate(ms_marco_results):
            expanded_data.append({
                "claimID": claim_id,
                "claim": claim,
                "abstract": abstract,
                "abstract_original_index": abs_idx,
                "msmarco_rank": rank,  
                "total_votes": {
                "supports": 0,
                "refutes": 0,
                "not enough information": 0,
                "unknown": 0
                }
            })
    
    # Convert to Pandas DataFrame
    expanded_df = pd.DataFrame(expanded_data)
    print(f"Prepared dataset with {len(expanded_df)} rows.")
    
    # Convert to Hugging Face dataset
    hf_dataset = Dataset.from_pandas(expanded_df)
    
    # Save the dataset
    hf_dataset.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}.")


if __name__ == "__main__":
    
    input_path = "../final_english_claims_reranked_msmarco.pkl"  
    output_path = "../final_english_claims_expanded_masmarco_top10-20.hf"  

    prepare_data(input_path, output_path)
