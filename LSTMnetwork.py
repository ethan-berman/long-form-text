import numpy as np
import pandas as pd
import nltk
import re
import tensorflow as tf
from nltk.corpus import stopwords
import random
from random import shuffle





#load inputs
training = pd.read_csv('tweets2.csv', header=0, encoding='iso-8859-1')
#print(training["text"])
NUM_EXAMPLES = 721
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

""" # THIS WAS FOR BAG OF WORDS, WAS VECTORIZING LIKE IN A BAG BEFORE and TRIED TO FEED IT IN FOR RNN, THIS CODE REFLECTED THAT
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

#print(vectorize(pruned[3][1]))



def onehot(vector):
    binary = [0] * len(bag)
    for num in vector:
    #	print(num)
    	binary[num[0]] = 1
    	# BAD binary.append(sample)
    #for loop until index reaches the index of the word in the bag
    return binary

train_input = []
for entry in pruned:
  train_input.append(entry[1])


#print(onehot(vectorize(pruned[0][1])))
training = []
train_output = []
for i in range(0, len(pruned)):
  train_output.append(onehot(vectorize(pruned[i][1])))

#print(train_output)
"""
train_input = []
for entry in pruned:
  train_input.append(entry[1])

train_input = [map(int,i) for i in train_input]
ti  = []
for i in train_input:
    temp_list = []
    temp_list.append([i])
    ti.append(np.array(temp_list))
train_input = ti

train_output = []
for i in train_input:
    count = 0
    for j in i:
        if j[0] == 1:
            count+=1
    temp_list = ([0]*5722)
    temp_list[count]=1
    train_output.append(temp_list)

test_input = []
test_output = []

test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:] #everything beyond 10,000
 
train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES] #till 10,000

#print(test_output)

#http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/ -- tensor flow RNN network from here. Comments written by me. ran into errors. realized I was passing in data incorrectly. Didn't have time to fix it.
#hold input and target data
data = tf.placeholder(tf.float32, [100, 20,1])
target = tf.placeholder(tf.float32, [100, 21])
#create RNN/LSTM cell
num_hidden = 24
cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
#pass data, store output as val
val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
#switch batch with tweet size
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)
#define weights and biases, and multiply the output with the weights and add the bias values to it.
weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
#probability score, using softmax regression activation
prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
#calculate cross entropy loss (degree of incorrectness)
cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
#minimize entropy loss
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)
# error = how many tweets defined incorrectly
mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
# CREATE MODEL
init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

batch_size = 100
no_of_batches = (len(train_input)) / batch_size
num_batches = int (no_of_batches)
epoch = 1000
for i in range(epoch):
    ptr = 0
    for j in range(num_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    #print ("Epoch",str(i))
incorrect = sess.run(error,{data: test_input, target: test_output})
print (sess.run(predictioerrorn,{data: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]}))
#print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()










#print(onehot(vectorize(pruned[0][1])))


