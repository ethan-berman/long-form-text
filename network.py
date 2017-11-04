

#load inputs
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords

#load inputs
training = pd.read_csv('tweets2.csv', header=0, encoding='iso-8859-1')
#print(training["text"])
pruned = []
bag = []
limit = int(training["text"].size)

def clean_tweet(tweet):
   #improve me, i make ugly text and I get rid ofthe wrong characters

	cutLinks = re.sub(r"http\S+", '', tweet)
	cutAts = re.sub("@[a-zA-z]+",  "", cutLinks)
	letters = re.sub("[^a-zA-Z-#]", " ", cutAts)
	words = letters.lower().split()
	stops = set(stopwords.words("english"))
	good_words = [w for w in words if not w in stops]
	print(good_words)
	return( " ".join(good_words))

   

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

# print(bag)


def vectorize(tweet):
	vectors = []
	words = tweet.split()
	for w in words:
		#print(w)
		indexes = []
		indexes.append(bag.index(w))
		vectors.append(indexes)
	return(vectors)

# print(vectorize(pruned[3][1]))



def onehot(vector):
    binary = []
    for num in vector:
    	print(num)
    	sample = [0] * len(bag)
    	sample[num[0]] = 1
    	binary.append(sample)
    #for loop until index reaches the index of the word in the bag
    return binary

#print(onehot(vectorize(pruned[0][1])))
#the above code can be used for all three approaches, although edits should be made to the clean_tweet function
#Next Step for bag of words is to Vectorize texts before going into pruned

#clean tweets add the text to a dictionary corresponding to which person tweeted it
#backwards propogation




#classify

#forward propogation


#calculate error



#backwards propogation



#classify
