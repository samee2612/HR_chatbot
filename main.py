import nltk
#stemming words to get root words
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

#read intents.json file contents
with open("intents.json") as file:
    data = json.load(file)#json data will be stored in the variable data

#to avoid doing the preprocessing steps again
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    #loop through all intents
    for intent in data["intents"]:
        #loop through all patterns in an intent
        for pattern in intent["patterns"]:
            #get single root words
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)#list of root words of patterns
            docs_x.append(wrds)#list of root words of patterns
            docs_y.append(intent["tag"]) #list of all intents

        #to get all tags of the intents
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))#remove duplicate lowercase words

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))] #initialize with 0

    #creating bag of words(freq of words in a sentence)
    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        #loop through all words in patterns
        for w in words:
            if w in wrds:
                bag.append(1) #represents tags of word
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training) #form taken by neural model
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.compat.v1.reset_default_graph() #resetting
                                    #model should take this length
net = tflearn.input_data(shape=[None, len(training[0])])#input layer(initialization)
net = tflearn.fully_connected(net, 8)#8 neurons in the hidden layer
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #output layer(softmax gets us probabilities of neurons)
net = tflearn.regression(net) #applying regression

model = tflearn.DNN(net)#train the model
#try:
#   model.load("model.tflearn")
#except:
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)#epoch is no of times it sees that data
model.save("model.tflearn")#Save that trained data
    
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

def get_response(msg):
    results = model.predict([bag_of_words(msg, words)]) [0]#give all the words with the input words and returns probabilities of each labels
    results_index = numpy.argmax(results) #pick greatest probability and return its index
    tag = labels[results_index] #returns the correct tag

    if results[results_index]>=0.6:
        for tg in data["intents"]:#pick the response corresponding to the tag
            if tg['tag'] == tag:
                responses = tg['responses']

        return (random.choice(responses))
    else:
        return ("I didn't understand. You can try rephrasing.")


   
#def chat():
    #print("Hello Lucy here. How can I help you? (To quit please type quit)")
    
    #while True:
    #    inp = input("You: ")
    #    if inp.lower() == "quit":
    #        break

    '''    results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))'''

#chat()