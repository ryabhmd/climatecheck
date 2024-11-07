import json
import pandas as pd
from tqdm import tqdm

def iterate_json_files(directory, pub_id):
    
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                try:
                    json_file = json.load(f)
                    json_data = json_file['data']
                    for publication in json_data['data']:
                        if publication['paperId'] == pub_id:
                            print(f'Found abstract for publication {pub_id}.')
                            return publication['abstract']
                            
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {filename}: {e}")
                    
    print(f"File not found {pub_id}.")
    return None


def main():

    directory = '/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/S2ORC'
    all_abstracts = []

    s2orc_publications = pd.read_pickle('/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/s2orc_publications.pkl')
    
    for index, row in tqdm(s2orc_publications.iterrows(), total=s2orc_publications.shape[0]):
        pub_id = row['id']
        absrtact = iterate_json_files(directory, pub_id)
        all_abstracts.append(abstract)

    s2orc_publications['abstract'] = all_abstracts

    s2orc_publications.to_pickle('/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/s2orc_publications_with_abstracts.pkl')


if __name__ == "__main__":
    main()

