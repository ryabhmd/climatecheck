import asyncio
import aiohttp
import pandas as pd
import json
import os
import pyalex
from pyalex import Topics, Works
from pyalex import config

# Setup the API email for OpenAlex
EMAIL = "raia.abu_ahmad@dfki.de"
config.email = EMAIL
config.max_retries = 3
config.retry_backoff_factor = 0.1
config.retry_http_codes = [429, 500, 503]

# Directory to save JSON data
save_dir = "/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/OpenAlex/"
os.makedirs(save_dir, exist_ok=True)

async def fetch_topics(session):
    """Fetch all topics related to 'climate change' from OpenAlex API."""
    pager = Topics().search_filter(display_name='climate change').paginate(per_page=None)
    topics_response = []
    for page in pager:
        topics_response.extend(page)
    return [topic['id'] for topic in topics_response]

async def fetch_works(session, topic_ids, idx):
    """Fetch works related to the specified topics."""
    pager_works = Works().filter(
        topics={"id": "|".join(topic_ids)},
        has_oa_accepted_or_published_version=True,
        has_fulltext=True
    ).paginate(n_max=None)

    for idx, page in enumerate(pager_works, start=idx):
        relevant_works = [work for work in page]
        
        # Save each page as JSON
        file_path = f"{save_dir}OpenAlex_ClimateCheck_{idx}.json"
        with open(file_path, 'w') as f:
            json.dump(relevant_works, f)
            print(f"Saved page {idx}")
        
        await asyncio.sleep(5)  # Non-blocking sleep

async def main():
    """Main function to fetch topics and works concurrently."""
    async with aiohttp.ClientSession() as session:
        # Fetch topics asynchronously
        topic_ids = await fetch_topics(session)
        
        # Fetch works asynchronously
        await fetch_works(session, topic_ids, idx=38669)

        # Aggregate data into a DataFrame
        aggregated_data = {
            'id': [],
            'doi': [],
            'title': [],
            'abstract': [],
            'oa_url': [],
            'topics': [],
            'keywords': [],
            'concepts': []
        }
        
        # Load and aggregate all saved JSON files
        for file_name in os.listdir(save_dir):
            if file_name.endswith(".json"):
                with open(os.path.join(save_dir, file_name), 'r') as f:
                    relevant_works = json.load(f)
                    for work in relevant_works:
                        aggregated_data['id'].append(work['id'])
                        aggregated_data['doi'].append(work.get('doi', ''))
                        aggregated_data['title'].append(work['title'])
                        aggregated_data['abstract'].append(work['abstract'])
                        aggregated_data['oa_url'].append(work.get('open_access', {}).get('oa_url', ''))
                        aggregated_data['topics'].append(work.get('topics', []))
                        aggregated_data['keywords'].append(work.get('keywords', []))
                        aggregated_data['concepts'].append(work.get('concepts', []))

                # Save aggregated data to CSV
                openalex_publications = pd.DataFrame(aggregated_data)
                openalex_publications.to_pickle('/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/openalex_publications.pkl', index=False)
                print(f"Aggregated data saved to pkl until file {file_name}.")

if __name__ == "__main__":
    asyncio.run(main())
