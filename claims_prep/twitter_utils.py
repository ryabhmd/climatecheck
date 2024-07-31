import os
import requests
import json
import time

"""
Function to collect Tweet info from the X Developer API using tweet IDs. 
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
    all_tweet_data = []
    error_data = []

    for i in range(0, len(tweet_ids), 100):
        ids_batch = tweet_ids[i:i+100]
        ids_string = ','.join(map(str, ids_batch))  # Convert numpy.int64 to str
        url = create_url(ids_string)
        
        # Rate limit handling
        if i > 0 and i % 1500 == 0:  # 15 requests (1500 tweets) per 15 mins
            print("Rate limit reached. Saving progress and sleeping for 15 minutes.")
            time.sleep(15 * 60)  # Sleep for 15 minutes

        try:
            response = connect_to_endpoint(url)
            if "error" in response:
                for tweet_id in ids_batch:
                    error_data.append({"id": tweet_id, "error": response["message"]})
                print(f"Error fetching tweets for IDs: {ids_batch}. Error: {response['message']}")
            else:
                if 'data' in response:
                    all_tweet_data.extend(response['data'])
                if 'errors' in response:
                    for error in response['errors']:
                        error_data.append({"id": error['resource_id'], "error": error['title']})
        except Exception as e:
            for tweet_id in ids_batch:
                error_data.append({"id": tweet_id, "error": str(e)})
            print(f"Exception occurred for IDs: {ids_batch}. Exception: {str(e)}")

        # Save data incrementally
        with open(output_file, 'w') as f:
            json.dump({"tweets": all_tweet_data, "errors": error_data}, f, indent=4, sort_keys=True)
    
    return all_tweet_data, error_data