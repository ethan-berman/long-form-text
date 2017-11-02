import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords

#load inputs
training = pd.read_csv('tweets.csv', header=0)
#print(training["text"])
pruned = []
bag = []
limit = int(training["text"].size)

def clean_tweet(tweet):
    #improve me, i make ugly text and I get rid ofthe wrong characters
    letters = re.sub("[^a-zA-Z]", " ", tweet)
    words = letters.lower().split()
    stops = set(stopwords.words("english"))
    good_words = [w for w in words if not w in stops]

    return( " ".join( good_words ))

for i in range(0, limit):
    if training["is_retweet"][i] == False:
        cleaned = clean_tweet(training["text"][i])
        #cleaned should be vectorized before appending to pruned
        user = training["handle"][i]
        retweets = training["retweet_count"][i]
        favorites = training["favorite_count"][i]
        entry = [user, cleaned, retweets, favorites]
        pruned.append(entry)

#print(pruned)

for line in pruned:
    words = line[1].split()
    for w in words:
        if(w not in bag):
            bag.append(w)

print(bag)
def vectorize(tweet):
    vectors = []
    words = tweet.split()
    for w in words:
        print(w)
        indexes = []
        indexes.append(bag.index(w))
        vectors.append(indexes)
    return(vectors)
print(vectorize(pruned[0][1]))

#the above code can be used for all three approaches, although edits should be made to the clean_tweet function
#Next Step for bag of words is to Vectorize texts before going into pruned

#clean tweets add the text to a dictionary corresponding to which person tweeted it
#backwards propogation



#classify
