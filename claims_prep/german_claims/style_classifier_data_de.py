import pandas as pd
from datasets import load_dataset
import requests
from bs4 import BeautifulSoup
import nltk
import re

nltk.download('punkt')
"""
Script to prepare German data for the Text Style Classifier. 
Tweet data from: GerCCT + r/Kliawandel german subset
Non-tweet data from: Sentence-tokenized Wikipedia articles

"""

def prep_tweets_data():

    german_data = pd.read_pickle('final_german_claims.pkl')

    gercct = german_data[german_data['source'] == 'GerCCT']
    r_klimawandel = german_data[german_data['source'] == 'Klimawandel-Subreddit']

    klimawandel_sentences = []
    for sentence in r_klimawandel['claim']:
        split_sentences = nltk.tokenize.sent_tokenize(sentence, language='german')
        for sentence in split_sentences:
            if len(sentence) >= 50:
                klimawandel_sentences.append(sentence)

    klimawandel_sentences = pd.DataFrame(klimawandel_sentences, columns=['claim'])

    gercct = gercct[['claim']]
    gercct = gercct.rename(columns={'claim': 'text'})

    klimawandel_sentences = klimawandel_sentences[['claim']]
    klimawandel_sentences = klimawandel_sentences.rename(columns={'claim': 'text'})

    klimawandel_sentences = klimawandel_sentences.drop_duplicates(subset=['text'])
    gercct = gercct.drop_duplicates(subset=['text'])

    social_media_de_df = pd.concat([gercct, klimawandel_sentences], ignore_index=True)

    labels_tweets = ['social_media'] * len(social_media_de_df)
    social_media_de_df['label'] = labels_tweets

    return social_media_de_df

def prep_non_tweets_data():

    urls = [
    "https://de.wikipedia.org/wiki/Klimawandel",
    "https://de.wikipedia.org/wiki/Globale_Erw%C3%A4rmung",
    "https://de.wikipedia.org/wiki/Forschungsgeschichte_des_Klimawandels",
    "https://de.wikipedia.org/wiki/Klimahysterie",
    "https://de.wikipedia.org/wiki/Klimawandelleugnung",
    "https://de.wikipedia.org/wiki/Folgen_der_globalen_Erw%C3%A4rmung_in_der_Arktis#Schrumpfendes_arktisches_Meereis",
    "https://de.wikipedia.org/wiki/Folgen_der_globalen_Erw%C3%A4rmung",
    "https://de.wikipedia.org/wiki/Klimamodell",
    "https://de.wikipedia.org/wiki/Anpassung_an_die_globale_Erw%C3%A4rmung",
    "https://de.wikipedia.org/wiki/Kontroverse_um_die_globale_Erw%C3%A4rmung",
    "https://de.wikipedia.org/wiki/UN-Klimakonferenz_in_Dubai_2023",
    "https://de.wikipedia.org/wiki/Umweltbewegung#Klimaschutz",
    "https://de.wikipedia.org/wiki/Treibhausgas",
    "https://de.wikipedia.org/wiki/Treibhauseffekt",
    "https://de.wikipedia.org/wiki/Klimaschutz"
    ]

    texts = []
    for url in URLs:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for p in soup.find_all('p'):
            texts.append(p.text)

    sentences = []
    for text in texts:
        split_sentences = nltk.tokenize.sent_tokenize(text, language='german')
        # add sentences only if they are not numbers in square brackets such as [42]
        for sentence in split_sentences:
            if not re.search(r'\[.*?\]', sentence):
                sentences.append(sentence)

    non_social_media_de_df = pd.DataFrame(sentences, columns=['text'])
    labels_non_tweets = ['non_social_media'] * len(non_social_media_de_df)
    non_social_media_de_df['label'] = labels_non_tweets
        
    return non_social_media_de_df


def main():

    social_media_de_df = prep_tweets_data()

    # Save tweets data
    social_media_de_df.to_pickle('social_media_de_df.pkl')

    non_social_media_de_df = prep_non_tweets_data()

    # Save non-tweets data
    non_social_media_de_df.to_pickle('non_social_media_de_df.pkl')


if __name__ == "__main__":
    main()
