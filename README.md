# Tweets Application analysis

## Overview

In this project, we created an application to retrieve, process, view and analyze tweets about a subject identified by the user.
This project is composed of two parts:
* A script to execute a request on the twitter API. The script allows you to retrieve tweets through the Twitter API. The tweets are then processed using Natural Language Processing techniques.
* A second script to deploy a Dash application to visualize and analyze the data collected.

# Installation

To install and run the application just clone the repository.
```{linux}
git clone https://github.com/HugoFromont/app_tweet_analysis.git
```
You must then install the necessary libraries for this project.
```{linux}
pip install -r requirements.txt
```

# Setting
Before running the tweet collection script, you need to fill in the topics you want to follow.
For each subject, you must enter your credentials to connect to the Twitter API as well as a request to send to the API.

You must enter this information in the followig file 'get_tweet/twitter_api/conf.json'.

Here is the shape it should have:

```{json}
    "subject_1": {
      "consumer_key": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      "consumer_secret": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      "access_token": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      "access_token_secret": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      "q": "#hashtag_to_follow OR @identification_to_follow lang:fr "
    }
```

# Execution
To launch the tweet collection script, just run :

```{python}
script_get_tweet.py
```
The tweet will be save in the subject csv on data file.

To launch the dash app, you just need to run :
```{python}
python app\app.py
```

![demo](demo.gif)