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

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from keras.optimizers import SGD

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from gensim.models import Word2Vec

parser = argparse.ArgumentParser(description='')
parser.add_argument("-word2vec", type=str, help="word2vec地址（所有词）",
                    default="/home/xueguoqing01/.keras/models/GoogleNews-vectors-negative300.bin")
parser.add_argument("-word2vec_small", type=str, help="word2vec地址（部分词）",
                    default="./wv")
parser.add_argument("-data", type=str, help="全部数据", default="./train.csv")
parser.add_argument("-data_preprocess", type=str,
                    help="预处理过后的数据", default="./data_preprocess")
parser.add_argument("-max_seq_len", type=int, help="最大单词数", default=120)
parser.add_argument("-word_vector_dim", type=int, help="词向量维度", default=300)
parser.add_argument("-lstm_output_dim", type=int,
                    help="lstm输出维度", default=1000)
# parser.add_argument("-m", "--date", type=str,
#                     help="", default=today)
# parser.add_argument("-e", help="", nargs='+',
#                     action="store", required=True)
# parser.add_argument("-ne", help="", nargs='+',
#                     action="store", default=[])
# parser.add_argument("-t", type=str,
#                     help="", choices=['kmtl', 'msu'], required=True)
args = parser.parse_args()

word_set = set([])

if __name__ == "__main__":
    d = pd.read_csv(args.data, header=None)

    f_data_output = open(args.data_preprocess, 'w')
    for i, row in tqdm(list(enumerate(d.values))):
        try:
            nl = []

            q1 = row[3]
            q2 = row[4]

            def deal_problem(q):
                q = q.split()
                stop_word = set([',', '?', '.', '!', '"', ';'])
                q = [a for a in q if a not in stop_word]
                q = [''.join([c for c in a if c not in stop_word]) for a in q]
                return ','.join(q)

            q1 = deal_problem(q1)
            q2 = deal_problem(q2)

            for word in q1:
                word_set.add(word)

            for word in q2:
                word_set.add(word)

            f_data_output.write('\t'.join([q1, q2, str(row[5])]) + '\n')
        except:
            print row
