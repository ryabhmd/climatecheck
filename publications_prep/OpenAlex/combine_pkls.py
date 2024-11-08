import pandas as pd
import glob
from pathlib import Path
from tqdm import tqdm

def extract_topic_id_from_filename(filename):
    return filename.split('_')[-1].replace('.pkl', '')

def process_pkls(directory='.'):
    
    # Find all climate work PKL files
    pkl_files = glob.glob(f"{directory}/climate_works_*.pkl")
    print(f"\nFound {len(pkl_files)} PKL files to process")
    

    combined_data = []
    
    # Process each PKL file
    for pkl_file in tqdm(pkl_files, desc="Processing PKL files"):
        topic_id = extract_topic_id_from_filename(pkl_file)
        
        # load pkl file
        df = pd.read_pickle(pkl_file)
        print(f"\nProcessing {pkl_file} with {len(df)} works")
        
        # process each work in the DataFrame
        for _, work in df.iterrows():
            try:
                # convert abstract from inverted index to text
                abstract = None
                if work.get('abstract_inverted_index'):
                    inv_index = work['abstract_inverted_index']
                    max_position = max(position for positions in inv_index.values() 
                                    for position in positions)
                    words = [''] * (max_position + 1)
                    
                    for word, positions in inv_index.items():
                        for position in positions:
                            words[position] = word
                    
                    abstract = ' '.join(words).strip()
                
                work_data = {
                    'id': work.get('id'),
                    'doi': work.get('doi'),
                    'title': work.get('title'),
                    'abstract': abstract,  
                    'oa_url': (work.get('open_access', {}) or {}).get('oa_url'),
                    'topics': work.get('topics'),
                    'keywords': work.get('keywords'),
                    'concepts': work.get('concepts'),
                    'source_topic_id': topic_id
                }
                combined_data.append(work_data)
            except Exception as e:
                print(f"Error processing work from {pkl_file}: {str(e)}")
                continue
    
    # convert to DF
    combined_df = pd.DataFrame(combined_data)
    
    # remove duplicates based on work ID
    combined_df.drop_duplicates(subset=['id'], inplace=True)
    
    # save combined data
    output_file = 'combined_climate_works.pkl'
    combined_df.to_pickle(output_file)
    
    # summary
    print("\nSummary:")
    print(f"Total works processed: {len(combined_data):,}")
    print(f"Unique works after deduplication: {len(combined_df):,}")
    print(f"Data saved to: {output_file}")
    
    # Print some abstract statistics
    non_null_abstracts = combined_df['abstract'].notna().sum()
    print(f"Works with abstracts: {non_null_abstracts:,} ({non_null_abstracts/len(combined_df)*100:.1f}%)")
    
    return combined_df

def main():
    
    # process all pkl files
    df = process_pkls()
    
    # print sample 
    print("\nSample of combined data:")
    print(df.head())
    
    # stats
    print(f"Number of unique topics: {df['source_topic_id'].nunique()}")
    print("\nWorks per source topic:")
    print(df['source_topic_id'].value_counts())

if __name__ == "__main__":
    df = main()