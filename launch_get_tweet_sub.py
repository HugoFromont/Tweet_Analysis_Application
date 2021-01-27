from get_tweet.twitter_api import make_request

start_date = "2021-01-22"
end_date = "2021-01-24"

subject = "melenchon"
try:
    make_request.request_parameters(subject, start_date, end_date)
    print("Done : ", subject)
except:
    print("KO : ", subject)
