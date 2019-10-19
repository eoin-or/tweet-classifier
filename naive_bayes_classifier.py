#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from collections import Counter, defaultdict
import emoji
import twokenize

PATH_TO_CORPUS = '/home/eoin/college/snlp/project/offensive_dataset/'

class NB_Classifier:
    def __init__(self, d, d_k, n_k_counts, vocab):
        self.vocab = vocab
        ## Since the priors are assumed to be uniform, the values are hardcoded in to save from having to calculate them.
        self.priors = [0.5, 0.5]
        self.likelihoods = defaultdict(float)

        for i in range(2):
            class_sum = 0
            for v in vocab:
                class_sum += int(n_k_counts[(i, v)])

            ## class_sum is the number of times a word in the vocabulary occurs in a given class.
            for v in vocab:
                self.likelihoods[(i, v)] =  float(1 + n_k_counts[(i, v)]) / (len(vocab) + class_sum)

    def predict_class(self, d):
        "Takes as input a document as a list of words d and returns the predicted class of the document"
        preds = []

        for i in range(2):
            log_sum = 0
            for word in d:
                if (i, word) in self.likelihoods:
                    log_sum += math.log(self.likelihoods[(i, word)], 2)
            preds.append((math.log(self.priors[i], 2) + log_sum))
        max_pred = max(preds)
        return preds.index(max_pred)

def open_file(filename):
    with open(filename) as f:
        return f.read()

def seperate_emoji(tok_tweet, index):
    "Seperates out multi-emoji expressions into its component tokens"
    return tok_tweet[:index] + list(tok_tweet[index]) + tok_tweet[index + 1:]

def preprocess_corpus(corpus_name):
    "Tokenizes tweets in a given corpus and splits up multi-emoji expressions. Returns a list of lists of strings."
    tok_tweets = []
    with open(PATH_TO_CORPUS + corpus_name, encoding='utf-8') as f:
        while True:
            tweet = f.readline()
            if tweet == '':
                break
            tokenized_tweet = twokenize.tokenize(tweet)
            for word in tokenized_tweet:
                if word[0] in emoji.UNICODE_EMOJI and len(word) > 1:
                    tokenized_tweet = seperate_emoji(tokenized_tweet, tokenized_tweet.index(word))
            tok_tweets.append(tokenized_tweet)
        return tok_tweets
    

#### Training the classifier
tok_tweets = preprocess_corpus('train.tsv')

## Creating counts of a given word in a specific class
n_k_counts = defaultdict(int)
for tweet in tok_tweets:
    if tweet[0] == 'NOT':
        index = 0
    else:
        index = 1
    for word in tweet[1:]:
        n_k_counts[(index, word)] += 1

## Creating vocabulary
vocab = Counter()
for tweet in tok_tweets:
    vocab.update(tweet[1:])

## Limiting vocabulary to words used 2 or more times; otherwise the accuracy drops by about 5%
common_vocab = list(filter(lambda x: vocab[x] >= 2, list(vocab)))

def d(corpus_name):
    "Gets the total number of docs (tweets) in a given corpus"
    doc_count = 0
    with open(PATH_TO_CORPUS + corpus_name, mode='r', encoding='utf-8') as f:
        while True:
            tweet = f.readline()
            if tweet == '':
                break
            doc_count += 1
    return doc_count

doc_count = d('train.tsv')

## Based on the assumption of uniform priors, the number of documents in each class must be the same
d_k = doc_count / 2
nb = NB_Classifier(doc_count, d_k, n_k_counts, common_vocab)

total_correct = 0

false_pos = 0
false_neg = 0
#### Testing on the dev set
dev_set = preprocess_corpus('dev.tsv')
for tweet in dev_set:
    prediction = nb.predict_class(tweet[1:])
    if prediction == 0 and tweet[0] == 'NOT':
        total_correct += 1
    elif prediction == 1 and tweet[0] == 'OFF':
        total_correct += 1
    else:
        if prediction == 0:
            false_neg += 1
        if prediction == 1:
            false_pos += 1

print('False positives: ', false_pos, '\tFalse negatives: ', false_neg)


