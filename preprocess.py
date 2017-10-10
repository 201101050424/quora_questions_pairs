# encoding=utf-8
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
import argparse
import logging
import pandas as pd
import sys
import csv

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from keras.optimizers import SGD

from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from gensim.models import Word2Vec
from conf import *

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return text.split()

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument("-t", help="是否调试", action="store_true")
    # args = parser.parse_args()
    # if args.t:
    #     wv_path = './wv'
    #     train_path = 'train.csv_small'

    wv = KeyedVectors.load_word2vec_format(wv_path, binary=True)
    reader = csv.reader(open(train_path))

    word_index = {}
    word_index['<<PAD>>'] = 0
    f_data_output = open(train_preprocess_path, 'w')
    for row in tqdm(list(reader)):
        assert(len(row) == 6)
        nl = []

        q1 = text_to_wordlist(row[3])
        q2 = text_to_wordlist(row[4])

        def get_nq(q):
            nq = []
            for word in q1:
                if word not in wv.vocab:
                    continue
                if word not in word_index:
                    word_index[word] = len(word_index)
                nq.append(str(word_index[word]))
            while len(nq) < max_seq_len:
                nq.append('0')
            return nq[:max_seq_len]
        
        nq1 = get_nq(q1)
        nq2 = get_nq(q2)

        nl.append(','.join(nq1))
        nl.append(','.join(nq2))
        nl.append(row[-1])
        f_data_output.write('\t'.join(nl)+'\n')


    embedding_weight = np.zeros((len(word_index), word_vector_dim))
    for k in tqdm(word_index):
        if k == '<<PAD>>':
            continue
        embedding_weight[word_index[k]] = wv[k].astype(float)
    np.save(embedding_weight_path, embedding_weight)
