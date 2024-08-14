import requests
import argparse
import time
import pandas as pd
import json

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, help="Hugging Face API token")
    args = parser.parse_args()

    s2orc_key = args.api_key

    url = 'https://api.semanticscholar.org/graph/v1/paper/search'

    query_params = {'query': 'climate change', 
                'limit': 10, 
                'fieldsOfStudy': 'Environmental Science',
                'openAccessPdf': True,
                'fields': 'externalIds,title,year,abstract,url,fieldsOfStudy,s2FieldsOfStudy,openAccessPdf', 
                'offset': 0}

    headers = {'x-api-key': s2orc_key}

    response = requests.get(url, params=query_params, headers=headers)

    total_responses = response.json()['total']

    ids = [publication['paperId'] for publication in response.json()['data']]
    dois = [publication['externalIds']['DOI'] for publication in response.json()['data']]
    titles = [publication['title'] for publication in response.json()['data']]
    openAccessPdfs = [publication['openAccessPdf'] for publication in response.json()['data']]
    fieldsOfStudy = [publication['fieldsOfStudy'] for publication in response.json()['data']]
    s2FieldsOfStudy = [publication['s2FieldsOfStudy'] for publication in response.json()['data']]

    # save response.json() to a json file
    with open(f"/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/S2ORC/s2orc_{query_params['offset']}.json", 'w') as f:
        json.dump(response.json(), f)

    offset = 10
    while offset <= total_responses:
        query_params['offset'] = offset
        response = requests.get(url, params=query_params, headers=headers)

        ids.extend([publication['paperId'] for publication in response.json()['data']])
        dois.extend([publication['externalIds']['DOI'] for publication in response.json()['data']])
        titles.extend([publication['title'] for publication in response.json()['data']])
        openAccessPdfs.extend([publication['openAccessPdf'] for publication in response.json()['data']])
        fieldsOfStudy.extend([publication['fieldsOfStudy'] for publication in response.json()['data']])
        s2FieldsOfStudy.extend([publication['s2FieldsOfStudy'] for publication in response.json()['data']])
        
        # Save response.json() to a json file
        with open(f"/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/S2ORC/s2orc_{query_params['offset']}.json", 'w') as f:
            json.dump(response.json(), f)
        print(f"Saved response of offset {offset}")
            
        offset += 10
        time.sleep(5)


    s2orc_publications = pd.DataFrame({
        'id': ids, 
        'doi': dois, 
        'title': titles, 
        'openAccessPdf': openAccessPdfs, 
        'fieldsOfStudy': fieldsOfStudy, 
        's2FieldsOfStudy': s2FieldsOfStudy
        })
        
    s2orc_publications.to_csv('/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/s2orc_publications.csv')


if __name__ == "__main__":
    main()