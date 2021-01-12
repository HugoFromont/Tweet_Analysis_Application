import os
import json
from get_tweet.nlp_tools import extract_information, text_cleaning, sentiment_analysis
import tweepy as tw
import pandas as pd
import spacy



def request_parameters(subject, start_date, end_date):
    """This function collect and process tweet. It apply the following operations :
            - Collect a tweet on twitter API
            - Collect information about tweet (hashtag, emoji...)
            - Clean the tweet
            - Calculate sentiment of tweet
            - Save tweet

    Parameters
    ----------
    subject : str
    start_date : str (exemple "2020-12-25")
    end_date : str (exemple "2020-12-25")
    """

    # Collect token and query on config file about subject
    path_conf_json = os.path.normpath(os.path.join(os.getcwd(), "get_tweet", "twitter_api", "config.json"))

    with open(path_conf_json, 'r') as config_file:
        config = json.load(config_file)

    consumer_key = config['twitter_token'][subject]['consumer_key']
    consumer_secret = config['twitter_token'][subject]['consumer_secret']
    access_token = config['twitter_token'][subject]['access_token']
    access_token_secret = config['twitter_token'][subject]['access_token_secret']
    query = config['twitter_token'][subject]['q']

    # Twitter authentication
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)

    # API request
    tweets = tw.Cursor(api.search,
                       q=query + " until:" + end_date + " since:" + start_date,
                       tweet_mode="extended"
                       ).items(5000000000)

    nlp = spacy.load("fr_core_news_sm")

    for tweet in tweets:

        tweet_process = text_cleaning.cleaning_text(tweet.full_text, nlp)
        note_sentiment, sentiment = sentiment_analysis.calculate_sentiment(tweet_process)

        tweet_result = {
            "created_at": [tweet.created_at],
            "id": [tweet.id],
            "full_init": [tweet.full_text],
            "user_id": [tweet.user.id_str],
            "user_name": [tweet.user.name],
            "user_followers_count": [tweet.user.followers_count],
            "retweet_count": [tweet.retweet_count],
            "favorite_count": [tweet.favorite_count],
            "emoji": [extract_information.extract_emoji(tweet.full_text)],
            "url": [extract_information.extract_url(tweet.full_text)],
            "hashtag": [extract_information.extract_hashtag(tweet.full_text)],
            "identification": [extract_information.extract_identification(tweet.full_text)],
            "tweet_process": [tweet_process],
            "note_sentiment": [note_sentiment],
            "sentiment": [sentiment]
        }

        # append data if csv exist or create csv file
        data_path = os.path.normpath(os.path.join(os.getcwd(), "data/tweet_" + subject + ".csv"))
        if not os.path.exists(data_path):
            pd.DataFrame(tweet_result).to_csv(data_path, index=False)
        else:
            pd.DataFrame(tweet_result).to_csv(data_path, mode='a', header=False, index=False)
