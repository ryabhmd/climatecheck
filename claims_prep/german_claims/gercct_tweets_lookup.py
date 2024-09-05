import os
import requests
import json
import time
from twitter_utils import (create_url, bearer_oauth, connect_to_endpoint, fetch_tweet_info)


"""
Code to retrieve tweet texts for the GerCCT data using Twitter API.
Bearer token needs to be replaced in twitter utils.py.
The result of the code is a .csv file of the GerCCT data with retrieved text of the source 
and reply tweets. If the text could not be retrieved, the error is logged in a separate column.
"""

def main():
    # Get tweet texts for GerCCT data
    gercct = pd.read_csv('gercct_class_annotations.csv')
    # Get only tweets initially annotated as verifiable claims
    gercct_verifiable_claims = gercct[gercct['verifiable_claim'] == 1]

    tweet_ids = list(gercct_verifiable_claims['source_id'].values)
    reply_ids = list(gercct_verifiable_claims['reply_id'].values)

    output_file = 'source_tweet_data.json'
    source_tweets, source_errors = fetch_tweet_info(tweet_ids, output_file)
    print(f"Fetched {len(source_tweets)} tweets.")
    print(f"Data saved to {output_file}.")

    # Same process for reply tweets in original data
    output_file = 'reply_tweet_data.json'
    reply_tweets, reply_errors = fetch_tweet_info(reply_ids, output_file)
    print(f"Fetched {len(reply_tweets)} tweets.")
    print(f"Data saved to {output_file}.")

    # Create DataFrames from the data
    source_tweet_df = pd.DataFrame(source_tweets)
    source_tweet_df = tweet_df[['id', 'text']]  # Select only 'id' and 'text' columns
    source_tweet_df.rename(columns={'id': 'source_id', 'text': 'source_text'}, inplace=True)

    reply_tweet_df = pd.DataFrame(reply_tweets)
    reply_tweet_df = reply_tweet_df[['id', 'text']]  # Select only 'id' and 'text' columns
    reply_tweet_df.rename(columns={'id': 'reply_id', 'text': 'reply_text'}, inplace=True)

    source_errors_df = pd.DataFrame(source_errors)
    reply_errors_df = pd.DataFrame(reply_errors)

    source_errors_df.rename(columns={'id': 'source_id', 'error': 'source_error'}, inplace=True)
    reply_errors_df.rename(columns={'id': 'reply_id', 'error': 'reply_error'}, inplace=True)

    # Ensure id columns are of type string
    gercct_verifiable_claims['source_id'] = gercct_with_text['source_id'].astype(str)
    source_tweet_df['source_id'] = source_tweet_df['source_id'].astype(str)
    source_errors_df['source_id'] = source_errors_df['source_id'].astype(str)

    gercct_verifiable_claims['reply_id'] = gercct_with_text['reply_id'].astype(str)
    reply_tweet_df['reply_id'] = reply_tweet_df['reply_id'].astype(str)
    reply_errors_df['reply_id'] = reply_errors_df['reply_id'].astype(str)

    # drop dupes in tweet dataframes
    source_tweet_df = source_tweet_df.drop_duplicates(subset='source_id')
    reply_tweet_df = reply_tweet_df.drop_duplicates(subset='reply_id')

    # Merge original data with the four created dataframes
    merged_df = pd.merge(gercct_verifiable_claims, source_tweet_df, on='source_id', how='left')
    merged_df = pd.merge(merged_df, reply_tweet_df, on='reply_id', how='left')
    merged_df = pd.merge(merged_df, source_errors_df, on='source_id', how='left')
    merged_df = pd.merge(merged_df, reply_errors_df, on='reply_id', how='left')

    merged_df.to_csv('gercct_with_text_and_errors.csv', index=False)

if __name__ == "__main__":
    main()
