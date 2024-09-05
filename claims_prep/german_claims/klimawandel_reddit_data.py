import praw
import requests
import pandas as pd
import re
from langdetect import detect
from datetime import datetime, timedelta
import time
import argparse

"""
Script that uses PRAW to get data from r/klimawandel on Reddit. 
Data from the last two years is retrieved, filtered, and saved in klimawandel_data.pkl. 
User should pass the following arguments to access the Reddit API:
1. reddit_client_id
2. reddit_client_secret
3. reddit_user_agent

Submissions are filtered to the following:
- Contain at least one keyword from a predefined list of German keywords. 
- Are mostly German text (checked using the langdetect library)

"""
def translate_keywords_to_german():
    """Translate English keywords to German."""
    # A dictionary of keywords translated to German
    german_keywords = {
        "climate change": "Klimawandel",
        "global warming": "globale Erwärmung",
        "greenhouse gas": "Treibhausgas",
        "carbon emission": "CO2-Emission",
        "fossil fuel": "fossiler Brennstoff",
        "ozone": "Ozon",
        "air pollution": "Luftverschmutzung",
        "carbon dioxide emissions": "Kohlendioxidemissionen",
        "deforestation": "Abholzung",
        "industrial pollution": "industrielle Verschmutzung",
        "rising sea levels": "steigender Meeresspiegel",
        "extreme weather": "extremes Wetter",
        "melting glaciers": "schmelzende Gletscher",
        "ocean acidification": "Ozeanversauerung",
        "biodiversity loss": "Verlust der biologischen Vielfalt",
        "ecosystem disruption": "Störung des Ökosystems",
        "carbon capture": "Kohlenstoffabscheidung",
        "carbon storage": "Kohlenstoffspeicherung",
        "soil carbon": "Bodenkohlenstoff",
        "renewable energy": "erneuerbare Energien",
        "sustainable practices": "nachhaltige Praktiken",
        "paris agreement": "Pariser Abkommen",
        "kyoto protocol": "Kyoto-Protokoll",
        "carbon tax": "CO2-Steuer",
        "emissions trading schemes": "Emissionshandelssysteme",
        "green technology": "grüne Technologie",
        "sustainable technology": "nachhaltige Technologie",
        "environmental change": "Umweltveränderung"
    }
    return list(german_keywords.values())

def contains_german_keywords(text, keywords):
    """Check if the text contains any of the German keywords."""
    text = text.lower()
    for keyword in keywords:
        if keyword.lower() in text:
            return True
    return False

def is_mostly_german(text):
    """Check if the majority of the text is in German."""
    try:
        return detect(text) == "de"
    except:
        return False

def clean_text(text):
    """Remove paragraph breaks and hyperlinks from the text."""
    text = re.sub(r'\n+', ' ', text)  # Remove paragraph breaks
    text = re.sub(r'http\S+', '', text)  # Remove hyperlinks
    return text.strip()

def fetch_submissions_and_comments(reddit):
    """Fetch submissions and comments from the r/klimawandel subreddit."""

    subreddit_name = 'klimawandel'
    german_keywords = translate_keywords_to_german()
    two_years_ago = datetime.now() - timedelta(days=2*365)
    processed_data = []

    # Fetch submissions from the subreddit
    subreddit = reddit.subreddit(subreddit_name)
    for submission in subreddit.new(limit=None):
        submission_date = datetime.utcfromtimestamp(submission.created_utc)
        if submission_date < two_years_ago:
            break

        # Process submission text
        submission_text = submission.selftext or ''
        if (contains_german_keywords(submission_text, german_keywords) and
            is_mostly_german(submission_text)):
            clean_submission_text = clean_text(submission_text)
            if clean_submission_text:
                processed_data.append({
                    'type': 'submission',
                    'id': submission.id,
                    'author': submission.author.name if submission.author else '[deleted]',
                    'date': submission_date,
                    'text': clean_submission_text
                })

        # Fetch comments for each submission
        submission.comments.replace_more(limit=None)
        for comment in submission.comments.list():
            if (comment.body and
                not comment.author in [None, '[deleted]'] and
                not re.search(r'http\S+', comment.body) and
                contains_german_keywords(comment.body, german_keywords) and
                is_mostly_german(comment.body)):
                clean_comment_text = clean_text(comment.body)
                if clean_comment_text:
                    processed_data.append({
                        'type': 'comment',
                        'id': comment.id,
                        'author': comment.author.name,
                        'date': datetime.utcfromtimestamp(comment.created_utc),
                        'text': clean_comment_text
                    })

        time.sleep(10) # not to overload

    # Convert fetched data to dataframe
    df = pd.DataFrame(processed_data)
    return df

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--reddit_client_id", type=str, help="Personal reddit API client ID")
    parser.add_argument("--reddit_client_secret", type=str, help="Personal reddit API client secret")
    parser.add_argument("--reddit_user_agent", type=str, help="Personal reddit API user agent")
    args = parser.parse_args()

    # Initialize Reddit API credentials
    client_id = args.reddit_client_id
    client_secret = args.reddit_client_secret
    user_agent = args.reddit_user_agent

    # Create Reddit instance
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        requestor_kwargs={'timeout': 10}
        )

    data_frame = fetch_submissions_and_comments(reddit)
    data_frame.to_pickle('klimawandel_data.pkl')


if __name__ == "__main__":
    main()