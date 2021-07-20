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
# https://medium.com/analytics-vidhya/making-a-text-generator-using-markov-chains-e17a67225d10
# https://towardsdatascience.com/simulating-text-with-markov-chains-in-python-1a27e6d13fc6
# Trump's speeches here: https://github.com/ryanmcdermott/trump-speeches


def review_generator(document):
    for d in document:
        r = open("input/"+ d, encoding='utf8').read()
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
    
def make_pairs(corpus):
    for i in range(len(corpus)-1):
        yield (corpus[i], corpus[i+1])


def generate (corpus):
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
    st.title ("Meditation script of the day")
    for t in txt3:
        st.write (t)

def main():
    st.title('Meditation generator')
    reviews= ""
    document_ = ["meditation.txt","chopra.txt", "maxhavelaar.txt", "taylorswiftlyrics.txt"]

    document = st.sidebar.multiselect(
            "What to show left-axis (multiple possible)", document_, ["meditation.txt"]
        )
    text = document
    for d in document:
            r = open("input/"+d, encoding='utf8').read()
            reviews= reviews + r



    trump = reviews
    corpus = trump.split()
    if st.sidebar.button('GENERATE'):
        generate(corpus)

    tekst = (
        '<hr>Made by Rene Smit. (<a href=\'http://www.twitter.com/rcsmit\'>@rcsmit</a>) <br>'
        'Scripts are used from various meditation videos on Youtube<br>'
        'Sourcecode : <a href=\"https://github.com/rcsmit/">github.com/rcsmit</a>' )

    st.sidebar.markdown(tekst, unsafe_allow_html=True)

if __name__ == "__main__":

    main()
