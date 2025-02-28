import json
import pandas as pd
from tqdm import tqdm
import os

def iterate_json_files(directory, pub_id):
    
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                try:
                    json_file = json.load(f)
                    json_data = json_file['data']
                    for publication in json_data:
                        if publication['paperId'] == pub_id:
                            return publication['abstract']
                            
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {filename}: {e}")
                    
    print(f"File not found {pub_id}.")
    return None


def main():

    directory = '.../S2ORC'

    s2orc_publications = pd.read_pickle('.../s2orc_publications_deduplicated.pkl')
    s2orc_publications['abstract'] = None
    
    for index, row in tqdm(s2orc_publications.iterrows(), total=s2orc_publications.shape[0]):
        pub_id = row['id']
        abstract = iterate_json_files(directory, pub_id)
        s2orc_publications.at[index, 'abstract'] = abstract
        s2orc_publications.to_pickle(f'.../s2orc_publications_deduplicated_with_abstracts.pkl')


if __name__ == "__main__":
    main()

