import re
import os
import json
import unicodedata
from get_tweet.nlp_tools.extract_information import tweet_is_retweet
import spacy

def cleaning_text(tweet, nlp):
    """This function clean the tweet. It apply the following operations :
            - Removing unnecessary information (hashtag, identification, emojis, url)
            - Cleaning text (space, punctuation, accents...)
            - Lemmatising the tweet

        Parameters
        ----------
        tweet : str

        Returns
        -------
        str : The tweet processed
        """

    # Remove_hashtag
    EE_hashtag = r"#\S+"
    tweet = re.sub(EE_hashtag, "", tweet)

    # Remove_url
    EE_lien_url = r"http\S+"
    tweet = re.sub(EE_lien_url, "", tweet)

    # Remove_identification
    EE_identification = r"@[\w]{1,}\b"
    tweet = re.sub(EE_identification, "", tweet)

    # Remove_emoji
    EE_emoji = u'['u'\U0001F300-\U0001F5FF'u'\U0001F600-\U0001F64F'u'\U0001F680-\U0001F6FF'u'\u2600-\u26FF\u2700-\u27BF]+'
    tweet = re.sub(EE_emoji, "", tweet)

    # remove RT if tweet is retweet
    if tweet_is_retweet(tweet):
        tweet = tweet[2:]

    tweet = tweet.lower()
    tweet = tweet.replace("'", " ")
    tweet = tweet.replace("\n", " ")
    tweet = unicodedata.normalize("NFKD", tweet).encode("ASCII", "ignore").decode("utf-8")
    tweet = re.sub(r"[^a-z\s]", " ", tweet)

    # Collect word to remove
    path_conf_json = os.path.normpath(os.path.join(os.getcwd(), "get_tweet", "nlp_tools", "config.json"))

    with open(path_conf_json, 'r') as config_file:
        config = json.load(config_file)

    word_to_remove = config['word_to_remove']

    new_tweet = []
    for token in nlp(tweet):
        if not token.is_stop and not token.is_punct and not token.is_space and token.lemma_ not in word_to_remove and len(token.lemma_) > 1:
                new_tweet.append(token.lemma_)

    tweet_process = " ".join(new_tweet)

    return tweet_process