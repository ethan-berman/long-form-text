import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords

#load inputs
training = pd.read_csv('tweets.csv', header=0)
#print(training["text"])
pruned = []
limit = int(training["text"].size)

def clean_tweet(tweet):
    letters = re.sub("[^a-zA-Z]", " ", tweet)
    words = letters.lower().split()
    stops = set(stopwords.words("english"))
    good_words = [w for w in words if not w in stops]

    return( " ".join( good_words ))

for i in range(0, limit):
    if training["is_retweet"][i] == False:
        cleaned = clean_tweet(training["text"][i])
        user = training["handle"][i]
        retweets = training["retweet_count"][i]
        favorites = training["favorite_count"][i]
        pruned.append({user : {(retweets, favorites) : cleaned}})

print(pruned)
#clean tweets add the text to a dictionary corresponding to which person tweeted it
#backwards propogation



#classify
