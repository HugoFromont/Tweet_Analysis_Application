from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer


def calculate_sentiment(tweet, model="textblob"):
    """Extract the sentiment of the tweet

        Parameters
        ----------
        tweet : str
        model : str

        Returns
        -------
        float : The sentiment note of the tweet between -1 and 1
        str : The sentiment of the tweet (Positive, neutral or negative)
        """

    if tweet=="":
        tweet = " "

    if model == "textblob":
        note = TextBlob(tweet, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer()).sentiment[0]

    # Calculate the tweet sentiment
    if note <-0.1:
        sentiment="negative"
    elif note<0.1:
        sentiment="neutral"
    else:
        sentiment="positive"

    return (note,sentiment)


