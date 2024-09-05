import requests
from bs4 import BeautifulSoup
import json
import time

"""
Script to scrape claims about Climate Change from https://correctiv.org/faktencheck/klima/ 
including Behauptung, Bewertung, and Resources. 
"""

def get_html(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None

def extract_articles_correctiv(html):
    soup = BeautifulSoup(html, 'html.parser')
    articles = []
    for article in soup.find_all('a', class_='teaser__item'):
        link = article['href']
        articles.append(link)
    return articles

def extract_behauptung(soup):
    """Extracts the 'Behauptung' text from the article."""
    behauptung_box = soup.find('div', class_='detail__box-title', string='\n            Behauptung        ')
    if behauptung_box:
        behauptung_content = behauptung_box.find_next('div', class_='detail__box-content')
        return behauptung_content.text.strip()
    return None

def extract_bewertung(soup):
    """Extracts the 'Bewertung' text and its assessment from the article."""
    bewertung_title_div = soup.find('div', class_='detail__box-title', string='\n            Bewertung        ')
    if bewertung_title_div:
        bewertung_rating_text = bewertung_title_div.find_next('div', class_='detail__rating-text').strong.text.strip()
        # Move up to the parent <div> with class "detail__box"
        bewertung_box_div = bewertung_title_div.find_parent('div', class_='detail__box')

        if bewertung_box_div:
            # Extract and return all the text content within this <div>
            bewertung_content = bewertung_box_div.get_text(strip=True, separator=' ')
            return bewertung_rating_text, bewertung_content
    return None, None

def extract_resources(soup):
    """Extracts the resources from the article."""
    resources = []
    resource_header = soup.find('h4', string='Die wichtigsten, öffentlichen Quellen für diesen Faktencheck:')
    if resource_header:
        resource_list = resource_header.find_next('ul')
        if resource_list:
            resource_items = resource_list.find_all('li')
            for item in resource_items:
                resource_text = item.text.strip()
                resource_link = item.find('a')['href'] if item.find('a') else None
                resources.append({'text': resource_text, 'link': resource_link})
    return resources

def get_article_details(url):
    """Fetches the article's details like 'Behauptung', 'Bewertung', and 'Resources'."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    behauptung = extract_behauptung(soup)
    bewertung_rating_text, bewertung_content = extract_bewertung(soup)
    resources = extract_resources(soup)

    return {
        'URL': url,
        'Behauptung': behauptung,
        'Bewertung': bewertung_rating_text,
        'Bewertung Content': bewertung_content,
        'Resources': resources
    }

def main():

    base_url = 'https://correctiv.org/faktencheck/klima/'
    html = get_html(base_url)
    article_links = extract_articles_correctiv(html)

    articles = []

    # Loop through each article URL and get the details
    for url in article_links:
        article_details = get_article_details(url)
        articles.append(article_details)

    # Convert to DataFrame and save to a CSV file
    df = pd.DataFrame(articles)

    # Normalize the 'Resources' field for better CSV formatting
    df['Resources'] = df['Resources'].apply(lambda x: "; ".join([f"{r['text']} ({r['link']})" for r in x]))

    df.to_csv('correctiv.csv', index=False)
    print('Scraped Correctiv data and saved at correctiv.csv')
    
if __name__ == "__main__":
    main()