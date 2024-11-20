import requests
import argparse
import time
import pandas as pd
import json
import pickle


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, help="Hugging Face API token")
    args = parser.parse_args()

    s2orc_key = args.api_key

    url = 'https://api.semanticscholar.org/graph/v1/paper/search/bulk'

    query_params = {'query': 'climate change', 
                'limit': 10, 
                'fieldsOfStudy': 'Environmental Science',
                'openAccessPdf': "True",
                'fields': 'externalIds,title,year,abstract,url,fieldsOfStudy,s2FieldsOfStudy,openAccessPdf,citationCount'}

    headers = {'x-api-key': s2orc_key}

    response = requests.get(url, params=query_params, headers=headers)

    total_responses = response.json()['total']

    ids, dois, titles, abstracts, urls, openAccessPdfs, fieldsOfStudy, s2FieldsOfStudy, citationCounts = [], [], [], [], [], [], [], [], []

    idx = 0
    errors = []

    while len(ids)  <= total_responses:
        try:
            ids.extend([publication['paperId'] for publication in response.json()['data']])
            dois.extend([publication['externalIds']['DOI'] for publication in response.json()['data']])
            titles.extend([publication['title'] for publication in response.json()['data']])
            abstracts.extend([publication['abstract'] for publication in response.json()['data']])
            urls.extend([publication['url'] for publication in response.json()['data']])
            openAccessPdfs.extend([publication['openAccessPdf'] for publication in response.json()['data']])
            fieldsOfStudy.extend([publication['fieldsOfStudy'] for publication in response.json()['data']])
            s2FieldsOfStudy.extend([publication['s2FieldsOfStudy'] for publication in response.json()['data']])
            citationCounts.extend([publication['citationCount'] for publication in response.json()['data']])
            print(f"Got data for idx {idx}")

        except Exception as error:
            errors.append((idx, error))
            print(f"Error for {idx}: {error}")
            
        # Save response.json() to a json file
        with open(f"/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/S2ORCV2/s2orc_{idx}.json", 'w') as f:
            json.dump(response.json(), f)
        print(f"Saved response of {idx}")

        token = response.json()['token']
        query_params['token'] = token
            
        idx += 1
        response = requests.get(url, params=query_params, headers=headers)
        time.sleep(10)


    s2orc_publications = pd.DataFrame({
        'id': ids, 
        'doi': dois, 
        'title': titles, 
        'abstract': abstracts,
        'url': urls,
        'openAccessPdf': openAccessPdfs, 
        'fieldsOfStudy': fieldsOfStudy, 
        's2FieldsOfStudy': s2FieldsOfStudy,
        'citationCount': citationCounts
        })
        
    s2orc_publications.to_pickle('/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/s2orc_publications_v3_citations.pkl')

    with open('/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/s2orc_errors.pkl', 'wb') as f:
        pickle.dump(errors, f)



if __name__ == "__main__":
    main()
