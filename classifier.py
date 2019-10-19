# -*- coding: utf-8 -*-

import functools
import operator
import re
import twokenize

import emoji
import string
import spacy

def split_emoji(tweet):
    em_split_emoji = emoji.get_emoji_regexp().split(tweet)
    em_split_whitespace = [substr.split() for substr in em_split_emoji]
    em_split = functools.reduce(operator.concat, em_split_whitespace)
    return em_split

tweets = []
labels = []

nlp = spacy.load("en_core_web_sm")

with open('dev.tsv', encoding='utf-8') as f:
    line = f.readline()
    print('Reading tweets...')
    while True:
        if line == '':
            break
       
        labels.append(line.split()[0])
        doc = nlp(line)

        line = line.translate(str.maketrans('', '', string.punctuation.replace('#', '')))
        tweet = twokenize.tokenize(str(" ".join(split_emoji(line.lower()))))[1:]
         
        for e in doc.ents:
            if e.text in tweet:
                tweet[tweet.index(e.text)] = str(e.text + " " + e.label_)

        tweets.append(tweet)
        line = f.readline()
           

print('Done.')
with open('test.txt', encoding='utf-8', mode='w') as f:
    for t, l in zip(tweets, labels):
        f.write(" ".join(t))
        f.write(" __label__")
        f.write(l)
        f.write("\n")
