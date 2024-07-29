import os
import requests
import json
import time

"""
Code to retrieve tweet texts for the GerCCT data using Twitter API.
Bearer token needs to be replaced. 
Some functions are taken from: https://github.com/xdevplatform/Twitter-API-v2-sample-code/blob/main/Tweet-Lookup/get_tweets_with_bearer_token.py
"""

bearer_token = os.environ.get("BEARER_TOKEN")

def create_url(ids):
    tweet_fields = "tweet.fields=entities,geo,id,lang,source,text,username"
    url = f"https://api.twitter.com/2/tweets?ids={ids}&{tweet_fields}"
    return url

def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2TweetLookupPython"
    return r

def connect_to_endpoint(url):
    response = requests.request("GET", url, auth=bearer_oauth)
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Content: {response.text}")  # Print the response content
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()

def fetch_tweet_info(tweet_ids, output_file):

    # Load existing data if the file exists
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            all_tweet_data = json.load(f)
    else:
        all_tweet_data = []

    for i in range(0, len(tweet_ids), 100):
        ids_batch = tweet_ids[i:i+100]
        ids_string = ','.join(map(str, ids_batch))  # Convert numpy.int64 to str
        url = create_url(ids_string)
        try:
            json_response = connect_to_endpoint(url)
            if json_response and 'data' in json_response and json_response['data']:
                all_tweet_data.extend(json_response['data'])
            else:
                print(f"No data found for IDs: {ids_batch}")
            # Save the data incrementally
            with open(output_file, 'w') as f:
                json.dump(all_tweet_data, f, indent=4, sort_keys=True)

        except Exception as e:
            print(f"Error fetching tweets: {e}")
        
        # Rate limit handling
        if (i // 100 + 1) % 15 == 0:
            print("Rate limit reached. Saving progress and sleeping for 15 minutes.")
            with open(output_file, 'w') as f:
                json.dump(all_tweet_data, f, indent=4, sort_keys=True)
            time.sleep(15 * 60)  # Sleep for 15 minutes
    
    return all_tweet_data

def main():
    # Get tweet texts for GerCCT data
    gercct = pd.read_csv('gercct_class_annotations.csv')
    # Get only tweets initially annotated as verifiable claims
    gercct_verifiable_claims = gercct[gercct['verifiable_claim'] == 1]

    tweet_ids = list(gercct_verifiable_claims['source_id'].values)
    reply_ids = list(gercct_verifiable_claims['reply_id'].values)

    output_file = 'tweet_data.json'
    tweet_data = fetch_tweet_info(tweet_ids, output_file)
    print(f"Fetched {len(tweet_data)} tweets.")
    print(f"Data saved to {output_file}.")

    # Create a DataFrame from the tweet data
    tweet_df = pd.DataFrame(tweet_data)
    tweet_df = tweet_df[['id', 'text']]  # Select only 'id' and 'text' columns
    tweet_df.rename(columns={'id': 'reply_id'}, inplace=True)

    # Ensure both source_id columns are of type string
    gercct_verifiable_claims['source_id'] = gercct_verifiable_claims['source_id'].astype(str)
    tweet_df['source_id'] = tweet_df['source_id'].astype(str)
    # Remove dups from tweet_df
    tweet_df = tweet_df.drop_duplicates(subset='source_id')

    gercct_verifiable_claims_with_text = pd.merge(gercct_verifiable_claims, tweet_df, on='source_id', how='left')

    # Same process for reply tweets in original data
    output_file = 'reply_tweet_data.json'
    tweet_data = fetch_tweet_info(reply_ids, output_file)
    print(f"Fetched {len(tweet_data)} tweets.")
    print(f"Data saved to {output_file}.")

    # same for reply IDs
    gercct_verifiable_claims_with_text['reply_id'] = gercct_verifiable_claims_with_text['reply_id'].astype(str)
    tweet_df['reply_id'] = tweet_df['reply_id'].astype(str)
    # Remove dups from tweet_df
    tweet_df = tweet_df.drop_duplicates(subset='reply_id')

    gercct_verifiable_claims_with_text = pd.merge(gercct_verifiable_claims_with_text, tweet_df, on='reply_id', how='left')

    gercct_verifiable_claims_with_text.to_csv('gercct_with_text_with_nan.csv', index=False)

    # Remove rows for which no tweet text could be retrieved for source or reply
    gercct_with_text = gercct_verifiable_claims_with_text[gercct_verifiable_claims_with_text['text_x'].notna() | gercct_verifiable_claims_with_text['text_y'].notna()]

    gercct_with_text.to_csv('gercct_with_text.csv', index=False)


if __name__ == "__main__":
    main()
