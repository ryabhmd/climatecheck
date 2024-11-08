import pandas as pd
import zipfile
import os


def main():

	s2orc_publications = pd.read_pickle('/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/s2orc_publications.pkl')
	s2orc_publications['doi'] = 'https://doi.org/' + s2orc_publications['doi']

	open_alex_publlications = pd.read_pickle('/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/combined_climate_works_filtered.pkl')

	print(f'Number of S2ORC publications: {len(s2orc_publications)}.')
	print(f'Number of OpenAlex publications: {len(open_alex_publlications)}.')

	s2orc_publications_deduped = s2orc_publications[~s2orc_publications['doi'].isin(open_alex_publlications['doi'])]
	print(f'Number of S2ORC publications after deduplication: {len(s2orc_publications_deduped)}.')

	s2orc_publications_deduped.to_pickle('/netscratch/abu/Shared-Tasks/ClimateCheck/data/publications/s2orc_publications_deduplicated.pkl')


if __name__ == "__main__":
    main()




