import pandas as pd
from datasets import load_dataset

"""
Script to prepare English data for the Text Style Classifier. 
Tweet data from: ClimaConvo + Sentiment140.
Non-tweet data from: news-category-dataset + arxiv-abstracts-2021 + wikipedia + enron_aeslc_emails + bookcorpus

Path to the ClimaConvo dataset should be given using --climaconvo_path.
"""

def prep_tweets_data(climaconvo_path):

    climaconvo = pd.read_csv('full_climaconvo_dataset.csv')

    # Dedup according to Tweet column
    climaconvo = climaconvo.drop_duplicates(subset=['Tweet'])
    climaconvo_tweets = climaconvo[['Tweet']]
    # rename column to 'text'
    climaconvo_tweets = climaconvo_tweets.rename(columns={'Tweet': 'text'})

    # Sentiment 140 - load from "stanfordnlp/sentiment140" huggingface
    sentiment140 = load_dataset("stanfordnlp/sentiment140", trust_remote_code=True)

    # Get random 15K instnace from sentiment140
    sentiment140_sample = sentiment140['train'].shuffle(seed=42).select(range(15500))

    sentiment140_sample = sentiment140_sample.to_pandas()
    # remove everything but 'text' column
    sentiment140_sample = sentiment140_sample[['text']]

    # deduplicate
    sentiment140_sample = sentiment140_sample.drop_duplicates(subset=['text'])

    # merge sentiment140_sample and climaconvo_tweets and shuffle
    tweets_df = pd.concat([sentiment140_sample, climaconvo_tweets])

    labels = ['tweet'] * len(tweets_df)
    tweets_df['label'] = labels

    return tweets_df

def prep_non_tweets_data():

    # News Articles (short description): get random 6K
    news_cat = load_dataset("heegyu/news-category-dataset")
    news_cat_sample = news_cat['train'].shuffle(seed=42).select(range(6000))
    news_cat_sample = news_cat_sample.to_pandas()
    news_cat_sample = news_cat_sample[['short_description']]
    news_cat_sample = news_cat_sample.rename(columns={'short_description': 'text'})
    news_cat_sample = news_cat_sample.drop_duplicates(subset=['text'])

    # Academic text: get random 6K
    arxiv = load_dataset("gfissore/arxiv-abstracts-2021")
    arxiv_sample = arxiv['train'].shuffle(seed=42).select(range(6000))
    arxiv_sample = arxiv_sample.to_pandas()
    arxiv_sample = arxiv_sample[['abstract']]
    arxiv_sample = arxiv_sample.rename(columns={'abstract': 'text'})
    arxiv_sample = arxiv_sample.drop_duplicates(subset=['text'])

    # Wiki articles: get random 6K
    wiki = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)
    wiki_sample = wiki['train'].shuffle(seed=42).select(range(6000))
    wiki_sample = wiki_sample.to_pandas()
    wiki_sample = wiki_sample[['text']]
    wiki_sample = wiki_sample.drop_duplicates(subset=['text'])

    # Emails: get random 6K
    emails = load_dataset("snoop2head/enron_aeslc_emails")
    emails_sample = emails['train'].shuffle(seed=42).select(range(6000))
    emails_sample = emails_sample.to_pandas()
    emails_sample = emails_sample[['text']]
    emails_sample = emails_sample.drop_duplicates(subset=['text'])

    # Books: get random 6K
    bookcorpus = load_dataset("bookcorpus/bookcorpus", trust_remote_code=True)
    # these will then be filtered according to length
    bookcorpus_sample = bookcorpus['train'].shuffle(seed=42).select(range(500000))
    bookcorpus_sample = bookcorpus_sample.to_pandas()
    bookcorpus_sample = bookcorpus_sample[['text']]
    # remove rows with 'text' less than 300 character
    bookcorpus_sample = bookcorpus_sample[bookcorpus_sample['text'].str.len() >= 300]
    bookcorpus_sample = bookcorpus_sample.drop_duplicates(subset=['text'])
    bookcorpus_sample = bookcorpus_sample.sample(n=6000, random_state=42)

    # concat and make new indices
    non_tweets_df = pd.concat(
        [news_cat_sample,
        arxiv_sample,
        wiki_sample,
        emails_sample,
        bookcorpus_sample], ignore_index=True)
    
    labels = ['non-tweet'] * len(non_tweets_df)
    non_tweets_df['label'] = labels

    return non_tweets_df


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--climaconvo_path", type=str, help="Path for the ClimaConvo dataset")
    args = parser.parse_args()

    climaconvo_path = args.climaconvo_path

    tweets_df = prep_tweets_data(climaconvo_path)

    # Save tweets data
    tweets_df.to_pickle('tweets_df.pkl')

    non_tweets_df = prep_non_tweets_data()

    # Save non-tweets data
    non_tweets_df.to_pickle('non_tweets_df.pkl')


if __name__ == "__main__":
    main()