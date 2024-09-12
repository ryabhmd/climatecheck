import asyncio
import aiohttp
import pandas as pd
import json
import pickle
import os
from huggingface_hub import HfApi, Repository
import argparse


async def fetch(session, url, params):
    try:
        async with session.get(url, params=params, headers=headers) as response:
            response_json = await response.json()
            offset = params['offset']

            # Save response to a local JSON file
            file_path = os.path.join(local_repo_path, f"s2orc_{offset}.json")
            with open(file_path, 'w') as f:
                json.dump(response_json, f)

            # Upload the file to Hugging Face Hub
            repo.git_add(file_path)
            repo.git_commit(f"Add data batch at offset {offset}")
            repo.git_push()
            print(f"Uploaded data for offset {offset} to Hugging Face.")

            return response_json
    except Exception as e:
        errors.append((params['offset'], str(e)))
        return None

async def fetch_all_data(total_responses):
    tasks = []
    async with aiohttp.ClientSession() as session:
        for offset in range(0, total_responses, query_params['limit']):
            query_params['offset'] = offset
            task = asyncio.ensure_future(fetch(session, url, query_params.copy()))
            tasks.append(task)
        return await asyncio.gather(*tasks)

async def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--s2orc_key", type=str, help="S2ORC API token")
    parser.add_argument("--hf_key", type=str, help="Hugging Face API token")
    args = parser.parse_args()

    s2orc_key = args.s2orc_key

    url = 'https://api.semanticscholar.org/graph/v1/paper/search'
    
    query_params = {
        'query': 'climate change',
        'limit': 100,
        'fieldsOfStudy': 'Environmental Science',
        'openAccessPdf': True,
        'fields': 'externalIds,title,year,abstract,url,fieldsOfStudy,s2FieldsOfStudy,openAccessPdf',
        'offset': 0
        }
        
    headers = {'x-api-key': s2orc_key}
    errors = []
    
    # Hugging Face setup
    hf_token = args.hf_key
    repo_id = "datasets/rabuahmad/climatecheck-publications"
    local_repo_path = "/netscratch/abu/Shared-Tasks/ClimateCheck/climatecheck/publications_prep/climatecheck-publications"
    
    api = HfApi()
    repo = Repository(local_repo_path, clone_from=repo_id, token=hf_token)

    # Ensure Git LFS is properly configured
    os.system("git lfs install")
    
    try:
        repo.git_pull()  # Pull latest changes from the repository
    except subprocess.CalledProcessError as e:
        print(f"Error pulling the repository: {e}")

    # Initial request to get total number of responses
    async with aiohttp.ClientSession() as session:
        initial_response = await fetch(session, url, query_params)
        if not initial_response:
            print("Failed to fetch initial data.")
            return

        total_responses = initial_response.get('total', 0)

        # Fetch all data and upload to Hugging Face
        all_data = await fetch_all_data(total_responses)

        # Collect data for DataFrame creation
        ids, dois, titles, abstract, urls, openAccessPdfs, fieldsOfStudy, s2FieldsOfStudy = [], [], [], [], [], [], [], []
        for response_json in all_data:
            if response_json and 'data' in response_json:
                for publication in response_json['data']:
                    ids.append(publication['paperId'])
                    dois.append(publication['externalIds'].get('DOI', ''))
                    titles.append(publication['title'])
                    abstract.append(publication['abstract'])
                    urls.append(publication['url'])
                    openAccessPdfs.append(publication['openAccessPdf'])
                    fieldsOfStudy.append(publication['fieldsOfStudy'])
                    s2FieldsOfStudy.append(publication['s2FieldsOfStudy'])

        # Save collected data to a CSV file
        csv_file_path = os.path.join(local_repo_path, 's2orc_publications.csv')
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
        s2orc_publications.to_csv(csv_file_path)

        # Upload CSV file to Hugging Face
        repo.git_add(csv_file_path)
        repo.git_commit("Add aggregated data CSV")
        repo.git_push()
        print("Uploaded aggregated data to Hugging Face.")

        # Save errors if any
        errors_file_path = os.path.join(local_repo_path, 's2orc_errors.pkl')
        with open(errors_file_path, 'wb') as f:
            pickle.dump(errors, f)

        # Upload errors file to Hugging Face
        repo.git_add(errors_file_path)
        repo.git_commit("Add error log")
        repo.git_push()
        print("Uploaded error log to Hugging Face.")

# Run the main function
asyncio.run(main())
