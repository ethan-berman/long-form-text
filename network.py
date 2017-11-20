

#load inputs
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
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
	#print(good_words)
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
def tokenize(tweet):
    words = tweet.split()
    return(words)

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
    binary = [0] * len(bag) 
    for num in vector:
    	#print(num)
    	binary[num[0]] = 1
    #for loop until index reaches the index of the word in the bag
    return binary

cleaned_tweets = []
trump_tweets = []
clinton_tweets = []
for entry in pruned:
    cleaned_tweets.append(tokenize(entry[1]))
    if entry[0] == "realDonaldTrump":
        trump_tweets.append(tokenize(entry[1]))
    else:
        clinton_tweets.append(tokenize(entry[1]))
#print(cleaned_tweets)
model = Word2Vec(cleaned_tweets, min_count=1)
#print(model)
trump_model = Word2Vec(trump_tweets, min_count=1)
clinton_model = Word2Vec(clinton_tweets, min_count=1)
#print(trump_tweets)
words = list(model.wv.vocab)
#print(words)
#print(model['trump'])
Y = model[trump_model.wv.vocab]
Z = model[clinton_model.wv.vocab]

X = model[model.wv.vocab]
pca = PCA(n_components=2)
print(model['america'])
trump_res = pca.fit_transform(Y)
clinton_res = pca.fit_transform(Z)
result = pca.fit_transform(X)
plt.scatter(trump_res[:,0], trump_res[:,1])
plt.scatter(clinton_res[:,0], clinton_res[:,1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i,0], result[i,1]))
model.save('model.bin')
plt.show()
def build(tweet):
    location = []
    for index in tweet:
        #find vector value for every word in tweet, take the sum
        location.extend(model[bag[index[0]]])
    return(location)
print(build(vectorize(pruned[0][1])))
spaced = []
for tweet in pruned:
    spaced.append(build(vectorize(tweet[1])))

#print(model['trump'])
#new_model = Word2Vec.load('model.bin')
#print(new_model)
#print(onehot(vectorize(pruned[245][1])))
#the above code can be used for all three approaches, although edits should be made to the clean_tweet function
#Next Step for bag of words is to Vectorize texts before going into pruned

#clean tweets add the text to a dictionary corresponding to which person tweeted it
#backwards propogation




#classify

#forward propogation


#calculate error



#backwards propogation



#classify
