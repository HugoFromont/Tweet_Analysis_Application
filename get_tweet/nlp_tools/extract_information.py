import re

def extract_emoji(tweet):
    """Extract emoji from the tweet

    Parameters
    ----------
    tweet : str

    Returns
    -------
    string : A string with all emoji in the tweet
    """

    # Regular expression for identify emoji
    EE_emoji = u'['u'\U0001F300-\U0001F5FF'u'\U0001F600-\U0001F64F'u'\U0001F680-\U0001F6FF'u'\u2600-\u26FF\u2700-\u27BF]+'
    # Extract emoji and concat them
    list_emoji = "".join(re.findall(EE_emoji, tweet))

    return list_emoji

def extract_hashtag(tweet):
    """Extract hashtag from the tweet

    Parameters
    ----------
    tweet : str

    Returns
    -------
    string : A string with all hashtag in the tweet
    """

    # Regular expression for extract hashtag
    EE_hashtag = r"#\S+"
    # Put a space befor hashtag for identify 2 hashtags not speparate by a space
    tweet = tweet.replace("#", " #")
    tweet = tweet.replace("]", " ]")
    # Extract hashtag and concat them in a string
    list_hashtag = " ".join(re.findall(EE_hashtag, tweet))

    return list_hashtag

def extract_identification(tweet):
    """Extract identification from the tweet

    Parameters
    ----------
    tweet : str

    Returns
    -------
    string : A string with all identification in the tweet
    """

    # Regular expression for identify identification
    EE_identification = r"@[\w]{1,}\b"
    # Put a space before identification for identify 2 identification not speparate by a space
    tweet = tweet.replace("]", " ]")
    tweet = tweet.replace("@", " @")
    # Extract identification and concat them in a string
    list_identification = " ".join(re.findall(EE_identification, tweet))

    return list_identification


def extract_url(tweet):
    """Extract url from the tweet

    Parameters
    ----------
    tweet : str

    Returns
    -------
    string : A string with all url in the tweet
    """
    # Regular expression for identify url
    EE_lien_url = r"http\S+"
    # Extract url and concat them in a string
    list_lien_url = " ".join(re.findall(EE_lien_url, tweet))

    return list_lien_url

def tweet_is_retweet(tweet):
    """Check if tweet is a Retweet

        Parameters
        ----------
        tweet : str

        Returns
        -------
        Booleen : True if tweet is a retweet false else
        """

    if tweet[:2] == "RT":
        is_retweet = True
    else:
        is_retweet = False

    return is_retweet
