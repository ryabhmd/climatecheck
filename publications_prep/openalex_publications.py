import pyalex
from pyalex import Works, Topics
from pyalex import config

import pandas as pd
import json

EMAIL = "raia.abu_ahmad@dfki.de"

def main():
    """
    Create an API request that filters:
        1. topics.id: at least one of the topics that includes "climate change" in its display_name
        2. has_oa_accepted_or_published_version:true
        3. has_fulltext:true
    """

    pyalex.config.email = EMAIL
    config.max_retries = 0
    config.retry_backoff_factor = 0.1
    config.retry_http_codes = [429, 500, 503]

    # Get all IDs from https://api.openalex.org/topics?filter=display_name.search:climate+change (should be 32)
    pager = Topics().search_filter(display_name='climate change').paginate(per_page=None)
    
    topics_response = []
    
    for page in pager:
        for topic in page:
            topics_response.append(topic)

    # Get string of IDs for filtering Works()
    climate_topic_ids = []
    for topic in topics_response:
        climate_topic_ids.append(topic['id'])

    id_text = [id + "|" for id in climate_topic_ids]
    id_text = "".join(id_text)[:-1]

    pager_works = Works().filter(topics={"id": id_text}, has_oa_accepted_or_published_version=True, has_fulltext=True).paginate(n_max=None)

    ids = []
    dois = []
    titles = []
    oa_urls = []
    topics = []
    keywords = []
    concepts = []
    
    for idx, page in enumerate(pager_works):
        relevant_works = []
        for work in page:
            relevant_works.append(work)
            ids.append(work['id'])
            dois.append(work['doi'])
            titles.append(work['title'])
            oa_urls.append(work['open_access']['oa_url'])
            topics.append(work['topics'])
            keywords.append(work['keywords'])
            concepts.append(work['concepts'])
        # save as json
        with open(f'/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/OpenAlex/OpenAlex_ClimateCheck_{idx}.json', 'w') as f:
            json.dump(relevant_works, f)
            print(f'Saved page {idx}')

    openalex_publications = pd.DataFrame({
        'id': ids, 
        'doi': dois, 
        'title': titles, 
        'oa_url': oa_urls, 
        'topics': topics, 
        'keywords': keywords, 
        'concepts': concepts
        })
    
    openalex_publications.to_csv('/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/openalex_publications.csv')
    

if __name__ == "__main__":
    main()