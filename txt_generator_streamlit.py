# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 18:13:41 2020

@author: rcxsm
"""
import streamlit as st
import numpy as np
import random
import re


# https://github.com/jsvine/markovify
#document = "joost.txt"
document = ["meditation.txt","chopra.txt"]
#document = "maxhavelaar.txt"
#document = "chopra.txt"
#document = ["taylorswiftlyrics.txt"]


def review_generator():
    # https://medium.com/analytics-vidhya/making-a-text-generator-using-markov-chains-e17a67225d10
    text = document
    for d in document:
        r = open(d, encoding='utf8').read()
        reviews= reviews + r

    #reviews = ''.join([i for i in reviews if not i.isdigit()]).replace("\n", " ").split(' ')
    
    index = 1
    chain = {}
    count = 100
    for word in reviews[index:]:
        key = reviews[index-1]
        if key in chain:
            chain[key].append(word)
        else:
            chain[key] = [word]
        index += 1
    
    word1 = random.choice(list(chain.keys()))
    message = word1.capitalize()

    while len(message.split(' ')) < count:
        word2 = random.choice(chain[word1])
        word1 = word2
        message += ' ' + word2
    if message [-1] != ".":
        message += "."
    message2= message.replace("e", ".XX")
    return message2




# ==============================================================================

# https://towardsdatascience.com/simulating-text-with-markov-chains-in-python-1a27e6d13fc6

# Trump's speeches here: https://github.com/ryanmcdermott/trump-speeches

reviews= ""
text = document
for d in document:
        r = open(d, encoding='utf8').read()
        reviews= reviews + r


#trump = open(document, encoding='utf8').read()
trump = reviews
corpus = trump.split()


def make_pairs(corpus):
    for i in range(len(corpus)-1):
        yield (corpus[i], corpus[i+1])
        
    
def generate ():
    pairs = make_pairs(corpus)
    word_dict = {}
    #print (pairs)
    
    for word_1, word_2 in pairs:
        if word_1 in word_dict.keys():
            word_dict[word_1].append(word_2)
        else:
            word_dict[word_1] = [word_2]
    
    first_word = np.random.choice(corpus)
    
    while first_word.islower():
        first_word = np.random.choice(corpus)
    
    chain = [first_word]
    
    n_words = 300
    
    for i in range(n_words):
        chain.append(np.random.choice(word_dict[chain[-1]]))
    
    ' '.join(chain)
    #print (chain)
    txt = "" 
    
    for i in chain:
        txt += i + " "
    if txt [-2] != " .":
      
        txt +=  "."

    #txt2= txt.replace(".", ".\n\n")
    txt2=txt
    txt3 =  re.split('(?=•)|(?=[A-Z])', txt2)

    for t in txt3:
        st.write (t)
    
if st.sidebar.button('GENERATE'):
    generate()
