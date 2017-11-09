

#load inputs
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
import datetime
import time
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

#load inputs
training = pd.read_csv('tweets2.csv', header=0, encoding='iso-8859-1')
#print(training["text"])
pruned = []
bag = []
output=[]
limit = int(training["text"].size)
amount_of_users=2
bowtraining=[]
userclass=[]
userclass.append("realDonaldTrump")
userclass.append("HillaryClinton")

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

#print(bag)


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
#

   
#
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
# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)
 




def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):
    print("STARTING TO TRAIN")

    
    print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
    print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, amount_of_users) )

    np.random.seed(1)

    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, amount_of_users)) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)  #Return an array of zeros with the same shape and type as a given array.
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)
        
    for j in iter(range(epochs+1)):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))
                
        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # how much did we miss the target value?
        layer_2_error = y - layer_2

        if (j% 10000) == 0 and j > 5000:
            # if this 10k iteration's error is greater than the last iteration, break out
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print ("delta after "+str(j)+" iterations:" + str(np.mesan(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break
                
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
        
        if(j > 0):
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))        
        
        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update
        
        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    now = datetime.datetime.now()

    # persist synapses
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'user': user
              }
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    #print ("saved synapses to:", synapse_file)



output_empty = [0] * amount_of_users

for prune in pruned:
        # initialize our bag of words
        tempbag = []
        # list of tokenized words for the pattern
        pattern_words = prune[1]
        # stem each word
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
        # create our bag of words array
        for w in bag:
            tempbag.append(1) if w in pattern_words else tempbag.append(0)

        bowtraining.append(tempbag)
        # output is a '0' for each tag and '1' for current tag
        output_row = list(output_empty)
       
        output_row[userclass.index(prune[0])] = 1

        
        output.append(output_row)
        print(str(output_row))

print("this is bowtraining" +str(bowtraining))
#pruned is a list of training data

X = np.array(bowtraining)
y = np.array(output)

start_time = time.time()

train(X, y, hidden_neurons=20, alpha=0.1, epochs=100000, dropout=False, dropout_percent=0.2)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")


# probability threshold
ERROR_THRESHOLD = 0.2
# load our calculated synapse values
synapse_file = 'synapses.json' 
with open(synapse_file) as data_file: 
    synapse = json.load(data_file) 
    synapse_0 = np.asarray(synapse['synapse0']) 
    synapse_1 = np.asarray(synapse['synapse1'])



#classify("this poem is about the color red", show_details=True)
#classify("I love this country.Iâ€™m proud of this country.I want to be a leader who brings people together.", show_details=True)


#classify

#forward propogation


#calculate error



#backwards propogation



#classify
