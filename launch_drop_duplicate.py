import pandas as pd
import json
import os

path_conf_json = os.path.normpath(os.path.join(os.getcwd(), "get_tweet", "twitter_api", "config.json"))

with open(path_conf_json, 'r') as config_file:
    config = json.load(config_file)

list_subject = config["twitter_token"].keys()

for subject in list_subject:
    data = pd.read_csv('data/tweet_{}.csv'.format(subject))
    data.drop_duplicates(subset=['id'], inplace=True)
    data.to_csv('data/tweet_{}.csv'.format(subject), index=False)
