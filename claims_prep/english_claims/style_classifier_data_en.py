import pandas as pd
from datasets import load_dataset

"""
Script to prepare English data for the Text Style Classifier. 
Tweet data from: ClimaConvo + DEBAGREEMENT.
Non-tweet data from: G11/climate_adaptation_abstracts + rlacombe/ClimateX + pierre-pessarossi/wikipedia-climate-data

Path to the ClimaConvo dataset should be given using --climaconvo_path.
Path to the DEBAGREEMENT dataset should be given using --debagreement_path.
"""

def prep_tweets_data(climaconvo_path, deb_path):

    climaconvo = pd.read_csv(climaconvo_path)

    # Dedup according to Tweet column
    climaconvo = climaconvo.drop_duplicates(subset=['Tweet'])
    climaconvo_tweets = climaconvo[['Tweet']]
    # rename column to 'text'
    climaconvo_tweets = climaconvo_tweets.rename(columns={'Tweet': 'text'})

    climaconvo_tweets = climaconvo_tweets.sample(n=3000, random_state=42, ignore_index=True)

    labels = ['social_media'] * len(climaconvo_tweets)
    climaconvo_tweets['label'] = labels

    debagreement = pd.read_csv(deb_path)
    debagreement = debagreement.drop_duplicates(subset=['body'])
    debagreement = debagreement[['body']]
    debagreement = debagreement.rename(columns={'body': 'text'})

    debagreement = debagreement.sample(n=3000, random_state=42, ignore_index=True)

    social_media_df = pd.concat(
        [climaconvo_tweets,
        debagreement], ignore_index=True)

    labels = ['social_media'] * len(social_media_df)
    social_media_df['label'] = labels

    return social_media_df

def prep_non_tweets_data():

    # Abstracts (short description): get random 2K
    abs = load_dataset("G11/climate_adaptation_abstracts")
    abstracts = abs['full'].to_pandas()
    abstracts = abstracts[['text']]
    abstracts = abstracts.drop_duplicates(subset=['text'])
    abstracts = abstracts.sample(n=2000, random_state=42, ignore_index=True)

    # Academic text: get random 6K
    climatex = load_dataset("rlacombe/ClimateX", trust_remote_code=True)
    climatex = climatex['train'].to_pandas()
    climatex = climatex[['statement']]
    climatex = climatex.rename(columns={'statement': 'text'})
    climatex = climatex.drop_duplicates(subset=['text'])
    climatex = climatex.sample(n=2000, random_state=42, ignore_index=True)

    # Wiki articles: get random 2K
    wiki = load_dataset("pierre-pessarossi/wikipedia-climate-data", trust_remote_code=True)
    wiki = wiki['train'].to_pandas()
    wiki = wiki[['content']]
    wiki = wiki.rename(columns={'content': 'text'})
    wiki = wiki.drop_duplicates(subset=['text'])
    wiki = wiki.sample(n=2000, random_state=42, ignore_index=True)

    # concat and make new indices
    non_social_media_df = pd.concat(
        [abstracts,
        climatex,
        wiki], ignore_index=True)
    
    labels = ['non_social_media'] * len(non_social_media_df)
    non_social_media_df['label'] = labels

    return non_tweets_df


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--climaconvo_path", type=str, help="Path for the ClimaConvo dataset")
    parser.add_argument("--debagreement_path", type=str, help="Path for the ClimaConvo dataset")
    args = parser.parse_args()

    climaconvo_path = args.climaconvo_path
    deb_path = args.debagreement_path

    social_media_df = prep_tweets_data(climaconvo_path, deb_path)

    # Save tweets data
    social_media_df.to_pickle('social_media_df.pkl')

    non_social_media_df = prep_non_tweets_data()

    # Save non-tweets data
    non_social_media_df.to_pickle('non_social_media_df.pkl')


if __name__ == "__main__":
    main()
