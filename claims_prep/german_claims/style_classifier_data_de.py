import pandas as pd
from datasets import load_dataset

"""
Script to prepare German data for the Text Style Classifier. 
Tweet data from: Alienmaster/SB10k + cardiffnlp/tweet_sentiment_multilingual german subset
Non-tweet data from: wikipedia + community-datasets/gnad10

"""

def prep_tweets_data():

    sb10k = load_dataset('Alienmaster/SB10k')
    tweet_sentiment = load_dataset('cardiffnlp/tweet_sentiment_multilingual', 'german')

    # Process sb10k
    sb10k_train = sb10k['train'].remove_columns(['Sentiment', 'Normalized', 'POS-Tags', 'Dependency Labels', 'additional Annotations'])
    sb10k_test = sb10k['test'].remove_columns(['Sentiment', 'Normalized', 'POS-Tags', 'Dependency Labels', 'additional Annotations'])
    sb10k_dev = sb10k['dev'].remove_columns(['Sentiment', 'Normalized', 'POS-Tags', 'Dependency Labels', 'additional Annotations'])

    sb10k_train = sb10k_train.to_pandas()
    sb10k_test = sb10k_test.to_pandas()
    sb10k_dev = sb10k_dev.to_pandas()
    sb10k_df = pd.concat([sb10k_train, sb10k_test, sb10k_dev], ignore_index=True)

    sb10k_df = sb10k_df.drop(columns=['ID'])
    sb10k_df = sb10k_df.rename(columns={'Text': 'text'})
    sb10k_df = sb10k_df.drop_duplicates(subset=['text'])

    # Process tweet_sentiment 
    tweet_sentiment_train = tweet_sentiment['train'].remove_columns(['label'])
    tweet_sentiment_test = tweet_sentiment['test'].remove_columns(['label'])
    tweet_sentiment_dev = tweet_sentiment['validation'].remove_columns(['label'])
    
    tweet_sentiment_train = tweet_sentiment_train.to_pandas()
    tweet_sentiment_test = tweet_sentiment_test.to_pandas()
    tweet_sentiment_dev = tweet_sentiment_dev.to_pandas()
    tweet_sentiment_df = pd.concat([tweet_sentiment_train, tweet_sentiment_test, tweet_sentiment_dev], ignore_index=True)
    tweet_sentiment_df = tweet_sentiment_df.drop_duplicates(subset=['text'])

    # merge sb10k_df and tweet_sentiment_df
    tweets_df = pd.concat([sb10k_df, tweet_sentiment_df], ignore_index=True)
    labels_tweets = ['tweet'] * len(tweets_df)
    tweets_df['label'] = labels_tweets

    # keep random 10K and make new index
    tweets_df = tweets_df.sample(n=10000, random_state=42)
    tweets_df = tweets_df.reset_index(drop=True)

    return tweets_df

def prep_non_tweets_data():

    wiki_de = load_dataset("wikipedia", "20220301.de", trust_remote_code=True)
    news_de = load_dataset("community-datasets/gnad10")

    # get random 5K from wiki_de
    wiki_de_sample = wiki_de['train'].shuffle(seed=42).select(range(5000))
    # get random 5K from news_de['train']
    news_de_sample = news_de['train'].shuffle(seed=42).select(range(5000))

    # convert wiki_de_sample to dataframe and drop id, url, and title
    wiki_de_sample = wiki_de_sample.to_pandas()
    wiki_de_sample = wiki_de_sample.drop(columns=['id', 'url', 'title'])
    # dedup
    wiki_de_sample = wiki_de_sample.drop_duplicates(subset=['text'])
    # convert news_de_sample to dataframe and drop label
    news_de_sample = news_de_sample.to_pandas()
    news_de_sample = news_de_sample.drop(columns=['label'])
    # dedup
    news_de_sample = news_de_sample.drop_duplicates(subset=['text'])

    # merge wiki_de_sample and news_de_sample
    non_tweets_df = pd.concat([wiki_de_sample, news_de_sample], ignore_index=True)

    labels = ['non-tweet'] * len(non_tweets_df)
    non_tweets_df['label'] = labels

    return non_tweets_df


def main():

    tweets_df = prep_tweets_data()

    # Save tweets data
    tweets_df.to_pickle('tweets_df_de.pkl')

    non_tweets_df = prep_non_tweets_data()

    # Save non-tweets data
    non_tweets_df.to_pickle('non_tweets_df_de.pkl')


if __name__ == "__main__":
    main()