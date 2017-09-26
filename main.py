# encoding=utf-8
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
import argparse
import logging
import inspect
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
parser.add_argument("-wv", type=str, help="word2vec地址",
                    default="/home/xueguoqing01/.keras/models/GoogleNews-vectors-negative300.bin")
parser.add_argument("-data", type=str, help="全部数据（经过预处理）", default="./data_preprocess")
parser.add_argument("-max_seq_len", type=int, help="最大单词数", default=120)
parser.add_argument("-word_vector_dim", type=int, help="词向量维度", default=100)
parser.add_argument("-lstm_output_dim", type=int,
                    help="lstm输出维度", default=1000)
parser.add_argument("-epochs", type=int,
                    help="", default=300)
parser.add_argument("-t", help="是否调试", action="store_true")
# parser.add_argument("-m", "--date", type=str,
#                     help="", default=today)
# parser.add_argument("-e", help="", nargs='+',
#                     action="store", required=True)
# parser.add_argument("-ne", help="", nargs='+',
#                     action="store", default=[])
# parser.add_argument("-t", type=str,
#                     help="", choices=['kmtl', 'msu'], required=True)
args = parser.parse_args()
wv = None

# 如果是调试
if args.t:
    args.wv='./wv'
    args.data='./data_preprocess_small'

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def lp(content):
    if type(content) is dict or type(content) is list:
        content = json.dumps(content, encoding='gb18030',
                             ensure_ascii=False).encode('gb18030')
    content = str(content)
    file_path = os.path.abspath(__file__)
    noline = str(inspect.currentframe().f_back.f_lineno)
    sys.stderr.write(file_path + ':' + noline + ' ' + content)
    sys.stderr.write('\n')
    sys.stderr.flush()


def init():
    global wv
    wv = KeyedVectors.load_word2vec_format(args.wv, binary=True)


def get_model():

    def get_sub_model():
        input_layer = Input(shape=(args.max_seq_len, args.word_vector_dim))
        output_layer = LSTM(args.lstm_output_dim, activation="relu")(input_layer)
        return input_layer, output_layer

    input1, output_layer1 = get_sub_model()
    input2, output_layer2 = get_sub_model()
    merge_layer = concatenate([output_layer1, output_layer2])
    output = Dense(1, activation="sigmoid")(merge_layer)
    model = Model(inputs=[input1, input2], outputs=output)
    model.compile(loss="binary_crossentropy",
                  optimizer=SGD(), metrics=['accuracy'])
    return model


def get_word_seq_from_question(q):
    seq = []
    for a in q:
        if a not in wv.vocab:
            continue
        seq.append(wv[a])

    while len(seq) < args.max_seq_len:
        seq.append(np.zeros(args.word_vector_dim))
    seq = np.stack(seq)
    # seq = np.expand_dims(seq, axis=0)
    return seq


def get_data():
    x1 = []
    x2 = []
    y = []
    for line in tqdm(list(open(args.data))):
        l = line.strip().split('\t')

        if len(l)!=3:
            lp(line)
            continue

        q1 = l[0].split(',')
        q2 = l[1].split(',')
        label = np.array([int(l[2])])

        q1_seq = get_word_seq_from_question(q1)
        q2_seq = get_word_seq_from_question(q2)

        x1.append(q1_seq)
        x2.append(q2_seq)
        y.append(label)

    x1 = np.stack(x1)
    x2 = np.stack(x2)
    y = np.stack(y)

    print x1.shape
    print x2.shape
    print y.shape

    return x1, x2, y



if __name__ == "__main__":
    init()
    # data = get_data()
    x1, x2, y = get_data()
    model = get_model()
    model.fit(x=[x1, x2], y=y, batch_size=1000,
              epochs=args.epochs, validation_split=0.2)
    model.save_weights('model.h5')
