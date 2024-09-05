import os
import requests
import json
import time
from twitter_utils import (create_url, bearer_oauth, connect_to_endpoint, fetch_tweet_info)


"""
Code to retrieve tweet texts for the ClimaConvo data using Twitter API.
Bearer token needs to be replaced in twitter utils.py.
The result of the code is a .csv file of the ClimaConvo data with retrieved text of the tweets. 
If the text could not be retrieved, the error is logged in a separate column.
"""

def main():
    # Get tweet texts for ClimaConvo data
    climaconvo = pd.read_csv('id_dataset.csv')
    # Get only relevant tweets
    climaconvo_relevants = climaconvo[climaconvo['Relevance'] == 1.0]

    tweet_ids = [str(int(float(id))) for id in climaconvo_relevants['ID'].values]

    output_file = 'climaconvo_tweet_data.json'
    tweets, errors = fetch_tweet_info(tweet_ids, output_file)
    print(f"Fetched {len(tweets)} tweets.")
    print(f"Data saved to {output_file}.")

    # add text + error data into the climaconvo dataframe
    tweets_df = pd.DataFrame(tweets)
    errors_df = pd.DataFrame(errors)
    
    tweets_df = tweets_df[['id', 'text']]  # Select only 'id' and 'text' columns
    tweets_df.rename(columns={'id': 'ID'}, inplace=True)
    errors_df.rename(columns={'id': 'ID'}, inplace=True)

    # Ensure both source_id columns are of type string
    climaconvo_relevants['ID'] = climaconvo_relevants['ID'].astype(str)
    merged_df = pd.merge(climaconvo_relevants, tweets_df, on='ID', how='left')
    merged_df = pd.merge(merged_df, errors_df, on='ID', how='left')

    merged_df.to_csv('climaconvo_relevants_with_text_and_errors.csv', index=False)

if __name__ == "__main__":
    main()

