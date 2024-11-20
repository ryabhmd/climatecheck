import pyalex
from pyalex import Works, Authors, Sources, Institutions, Topics, Publishers, Funders
from pyalex import config
import json
import pandas as pd
from tqdm import tqdm
import time
from pathlib import Path

# Configure pyalex for polite pool
pyalex.config.email = "nikolas.rauscher@dfki.de"
config.max_retries = 3
config.retry_backoff_factor = 0.1
config.retry_http_codes = [429, 500, 503]

def format_number(num):
    return f"{num:,}".replace(",", ".")

def get_climate_topics():

    pager = Topics().search_filter(display_name='climate change').paginate(per_page=None)
    topics_response = []
    
    for page in pager:
        for topic in page:
            topics_response.append(topic)
    
    # Extract topic IDs
    climate_topic_ids = [topic['id'] for topic in topics_response]
    id_text = "|".join(climate_topic_ids)
    
    # topics info
    print(f"\nFound {len(topics_response)} climate change related topics:")
    for topic in topics_response:
        works_count = format_number(topic['works_count'])
        print(f"ID: {topic['id']} - {topic['display_name']} (Works count: {works_count})")
    
    return topics_response, id_text

def fetch_and_save_works_by_topic(topics):
    
    print("\nFetching works for each topic...")
    
    for topic in tqdm(topics):
        topic_id = topic['id']
        topic_name = topic['display_name']
        filename = f"climate_works_{topic_id.replace('/', '_')}.pkl"
        
        # Check if the PKL file already exists
        if Path(filename).exists():
            print(f"\nSkipping {topic_name} as {filename} already exists.")
            continue
        
        # filters for open access and full text
        works_query = Works().filter(
            topics={"id": topic_id},
            has_oa_accepted_or_published_version=True,
            has_fulltext=True
        )
        
        # Get total count for topic
        total_works = works_query.count()
        print(f"\nFetching {format_number(total_works)} works for topic: {topic_name}")
        
        try:
            # Use paginate 
            all_works = []
            for works_page in works_query.paginate(per_page=200, n_max=None):
                all_works.extend(works_page)
                print(f"Fetched {format_number(len(all_works))} of {format_number(total_works)} works", end='\r')
                time.sleep(0.1)
            
            # Convert to DF
            if all_works:
                df = pd.DataFrame(all_works)
                df.to_pickle(filename)
                print(f"\nSaved {format_number(len(df))} works to {filename}")
            
        except Exception as e:
            print(f"\nError processing topic {topic_name}: {str(e)}")
            continue
        
        # delay for rate limit
        time.sleep(1)
    
    return True

def analyze_topics(id_text):
    print("\nAnalyzing works with open access and full text...")
    
    pager_works = Works().filter(
        topics={"id": id_text},
        has_oa_accepted_or_published_version=True,
        has_fulltext=True
    )
    
    total_results = pager_works.count()
    print(f"\nTotal results found: {format_number(total_results)}")
    
    return total_results

def main():
    # Get climate change topics
    topics, id_text = get_climate_topics()
    
    # Get total matching works
    total_works = analyze_topics(id_text)
    
    # Fetch and save works for each topic
    print("\nStarting to fetch and save works for each topic...")
    fetch_and_save_works_by_topic(topics)
    
    # Save 
    results = {
        'total_topics': len(topics),
        'total_matching_works': total_works,
        'total_matching_works_formatted': format_number(total_works),
        'topic_ids': id_text
    }
    
    with open('climate_topics_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to climate_topics_analysis.json")

if __name__ == "__main__":
    main()