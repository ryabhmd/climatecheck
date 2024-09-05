import requests
from bs4 import BeautifulSoup
import json

"""
Code that iterates through the articles on climate that are available on the climatefeedback website
and retrieves: title, link, key_takeaway, verdict, claim, source, date, verdict_detail, and references
for each article. Data is saved incrementally in JSON format. 
"""

def get_html(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None

def extract_articles(html):
    soup = BeautifulSoup(html, 'html.parser')
    articles = []
    for article in soup.find_all('a', class_='story__img-wrapper'):
        link = article['href']
        title = article['aria-label']
        articles.append({'title': title, 'link': link})
    return articles

def extract_key_takeaway(html):
    soup = BeautifulSoup(html, 'html.parser')
    key_takeaway_header = soup.find('h2', string='Key takeaway')
    if key_takeaway_header:
        key_takeaway_paragraph = key_takeaway_header.find_next('p')
        if key_takeaway_paragraph:
            return key_takeaway_paragraph.text.strip()
    return None

def extract_verdict_claim_source_date(html):
    soup = BeautifulSoup(html, 'html.parser')
    content_div = soup.find('div', class_=lambda value: value and value.startswith('sfc-review-reviewed-content'))
    if not content_div:
        return None, None, None, None

    verdict_label = content_div.find('p', class_='reviewed-content__label', string='Verdict:')
    if verdict_label:
        verdict = verdict_label.find_next_sibling('div').text.strip()
    else:
        verdict = None

    claim_label = content_div.find('p', class_='reviewed-content__label', string='Claim:')
    if claim_label:
        claim = claim_label.find_next('blockquote').text.strip()
    else:
        claim = None

    figcaption = content_div.find('figcaption', class_='reviewed-content__figcaption line-height-md flex gap-xs')
    if figcaption:
        source_date = figcaption.text.strip()
        source_and_date_parts = source_date.split(',')
        sources = ', '.join(source_and_date_parts[:-1]).strip()
        date = source_and_date_parts[-1].strip()
    else:
        sources = None
        date = None

    return verdict, claim, sources, date

def extract_verdict_detail(html):
    soup = BeautifulSoup(html, 'html.parser')
    verdict_detail_header = soup.find('h2', string='Verdict detail')
    if verdict_detail_header:
        verdict_detail_paragraph = verdict_detail_header.find_next('p')
        if verdict_detail_paragraph:
            return verdict_detail_paragraph.text.strip()
    return None

def extract_references(html):
    soup = BeautifulSoup(html, 'html.parser')
    references_header = soup.find('h4', class_='wp-block-heading is-style-large-uppercase bsab4s-toc-content__target', id='references')
    if references_header:
        references_list = references_header.find_next('ul', class_='wp-block-list')
        if references_list:
            references = []
            for li in references_list.find_all('li'):
                references.append(li.text.strip())
            return references
    return None

def scrape_all_articles(base_url):
    all_articles = []
    for page in range(1, 36):  # Currently there are 35 pages
        print(f'Scraping page {page}')
        url = f'{base_url}&_pagination={page}'
        html = get_html(url)
        if html:
            articles = extract_articles(html)
            for article in articles:
                title = article['title'][17:-1]
                print(f'Scraping article at {title}')
                article_html = get_html(article['link'])
                if article_html:
                    key_takeaway = extract_key_takeaway(article_html)
                    verdict, claim, source, date = extract_verdict_claim_source_date(article_html)
                    verdict_detail = extract_verdict_detail(article_html)
                    references = extract_references(article_html)
                    article['title'] = title
                    article['key_takeaway'] = key_takeaway
                    article['verdict'] = verdict
                    article['claim'] = claim
                    article['source'] = source
                    article['date'] = date
                    article['verdict_detail'] = verdict_detail
                    article['references'] = references
                    time.sleep(60)  
                else:
                    print(f'Failed to retrieve article at {article["link"]}')
            all_articles.extend(articles)
            with open('articles.json', 'w') as f:
              json.dump(all_articles, f, indent=4)

        else:
            print(f'Failed to retrieve page {page}')
        time.sleep(1) 
    return all_articles


def main():
    base_url = 'https://science.feedback.org/reviews/?_topic=climate'
    scrape_all_articles()



