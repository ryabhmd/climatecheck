import asyncio
import aiohttp
import pandas as pd
import json
import pickle
import os
import argparse
import random
import time

url = 'https://api.semanticscholar.org/graph/v1/paper/search/bulk'

parser = argparse.ArgumentParser()
parser.add_argument("--s2orc_key", type=str, help="S2ORC API token")
args = parser.parse_args()

s2orc_key = args.s2orc_key

query_params = {
    'query': 'climate change',
    'limit': 100,
    'fieldsOfStudy': 'Environmental Science',
    'openAccessPdf': "True",
    'fields': 'externalIds,title,year,abstract,url,fieldsOfStudy,s2FieldsOfStudy,openAccessPdf'
}

headers = {'x-api-key': s2orc_key}
errors = []

# Directory to save responses
save_dir = "/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/S2ORC/"
os.makedirs(save_dir, exist_ok=True)

async def fetch(session, url, params, retries=5):
    for attempt in range(retries):
        try:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 429:  # Too Many Requests
                    retry_after = int(response.headers.get("Retry-After", 2))  # Get the 'Retry-After' header value if available
                    wait_time = retry_after + random.uniform(0, 1)  # Add jitter to avoid synchronized retrying
                    print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
                    continue  # Retry request
                response_json = await response.json()
                idx = 0 
                params['token'] = response_json['token']
                
                # Save response to a local JSON file
                file_path = os.path.join(save_dir, f"s2orc_{idx}.json")
                with open(file_path, 'w') as f:
                    json.dump(response_json, f)

                return response_json
        except Exception as e:
            errors.append((params['offset'], str(e)))
            print(params['offset'])
            print(str(e))
            return None
        await asyncio.sleep(2 ** attempt)  # Exponential backoff

async def fetch_all_data(total_responses):
    tasks = []
    semaphore = asyncio.Semaphore(10)  # Limit concurrency to 10 requests

    async def sem_fetch(session, url, params):
        async with semaphore:
            return await fetch(session, url, params)

    async with aiohttp.ClientSession() as session:
        for offset in range(0, total_responses, query_params['limit']):
            query_params['offset'] = offset
            task = asyncio.ensure_future(sem_fetch(session, url, query_params.copy()))
            tasks.append(task)
        return await asyncio.gather(*tasks)

async def main():
    # Initial request to get total number of responses
    async with aiohttp.ClientSession() as session:
        initial_response = await fetch(session, url, query_params)
        if not initial_response:
            print("Failed to fetch initial data.")
            return

        total_responses = initial_response.get('total', 0)

        # Fetch all data
        all_data = await fetch_all_data(total_responses)

        # Collect data for DataFrame creation
        ids, dois, titles, abstracts, urls, openAccessPdfs, fieldsOfStudy, s2FieldsOfStudy = [], [], [], [], [], [], [], []
        for response_json in all_data:
            if response_json and 'data' in response_json:
                for publication in response_json['data']:
                    ids.append(publication['paperId'])
                    dois.append(publication['externalIds'].get('DOI', ''))
                    titles.append(publication['title'])
                    abstracts.append(publication['abstract'])
                    urls.append(publication['url'])
                    openAccessPdfs.append(publication['openAccessPdf'])
                    fieldsOfStudy.append(publication['fieldsOfStudy'])
                    s2FieldsOfStudy.append(publication['s2FieldsOfStudy'])

        # Save collected data to a CSV file
        s2orc_publications = pd.DataFrame({
            'id': ids,
            'doi': dois,
            'title': titles,
            'abstract': abstracts,
            'url': urls,
            'openAccessPdf': openAccessPdfs,
            'fieldsOfStudy': fieldsOfStudy,
            's2FieldsOfStudy': s2FieldsOfStudy
        })
        s2orc_publications.to_csv(os.path.join(save_dir, 's2orc_publications.csv'))

        # Save errors
        with open('/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/s2orc_errors.pkl', 'wb') as f:
            pickle.dump(errors, f)

        print("Data fetching complete.")

# Run the main function
asyncio.run(main())
