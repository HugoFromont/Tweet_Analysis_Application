import datetime
import os
import json
from get_tweet.twitter_api import make_request

start_date = str(datetime.date.today() - datetime.timedelta(days=1))
end_date = str(datetime.date.today())

#start_date = "2021-01-11"
#end_date = "2021-01-12"

# Collect request configuration
path_conf_json = os.path.normpath(os.path.join(os.getcwd(), "get_tweet", "twitter_api", "config.json"))

with open(path_conf_json, 'r') as config_file:
    config = json.load(config_file)

# List of all subject to collect
list_subject = config["twitter_token"].keys()

# Collect tweets subject
for subject in list_subject:
    try:
        make_request.request_parameters(subject, start_date, end_date)
        print("Done : ", subject)
    except:
        print("KO : ", subject)
