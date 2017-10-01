# encoding=utf-8
import sys
import argparse
import re
import os
import datetime
import logging
import logging.handlers
import redis
import traceback
import operator
import requests
import bisect
import json
import hashlib
import random
import inspect
import itertools
import numpy
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import csv

from operator import itemgetter
from tqdm import *
from subprocess import Popen
from subprocess import PIPE
from threading import Lock
from threading import Thread
from urllib import urlencode
from Queue import Queue
from conf import *
from gensim.models import KeyedVectors
from torch.autograd import Variable

def lp(content):
    if type(content) is dict or type(content) is list:
        content = json.dumps(content, encoding='gb18030', ensure_ascii=False).encode('gb18030')
    content=str(content)
    file_path = os.path.abspath(__file__)
    noline = str(inspect.currentframe().f_back.f_lineno)
    sys.stderr.write(file_path + ':' + noline + ' ' + content)
    sys.stderr.write('\n')
    sys.stderr.flush()

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
    
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    return text.split()

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

wv_path = "/home/xueguoqing01/.keras/models/GoogleNews-vectors-negative300.bin"
train_path = "./train.csv"
val_path = "./val.csv"
max_seq_len = 40
word_vector_dim = 300
num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
epoch_num = 100
batch_size = 2000


wv = None

class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_name):
        reader = csv.reader(open(train_path))
        self.content_list = []
        global wv
        self.wv = wv
        for row in reader:
            self.content_list.append(row)

    def __len__(self):
        return len(self.content_list)

    def get_wv_list(self, q):
        seq = []
        global wv
        for a in q:
            if a not in wv.vocab:
                continue
            seq.append(self.wv[a].astype(float))

        while len(seq) < max_seq_len:
            seq.append(np.zeros(word_vector_dim)) 

        seq = seq[:max_seq_len]
        seq = np.stack(seq)
        return seq

    def __getitem__(self, idx):
        row = self.content_list[idx]
        assert(len(row) == 6)
        q1 = text_to_wordlist(row[3])
        q2 = text_to_wordlist(row[4])

        q1 = self.get_wv_list(q1)
        q2 = self.get_wv_list(q2)
        y = np.array([float(row[-1])])

        # # q1 = Variable(torch.FloatTensor(q1))
        # # q2 = Variable(torch.FloatTensor(q2))
        # # y = Variable(torch.FloatTensor(y))

        q1 = torch.from_numpy(q1).float().cuda()
        q2 = torch.from_numpy(q2).float().cuda()
        y = torch.from_numpy(y).float().cuda()

        # q1 = torch.cuda.FloatTensor(max_seq_len, word_vector_dim)
        # q2 = torch.cuda.FloatTensor(max_seq_len, word_vector_dim)
        # y = torch.randn(1, 1).cuda()
        return q1, q2, y


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm1 = nn.LSTM(word_vector_dim, num_lstm, batch_first=True)
        self.lstm2 = nn.LSTM(word_vector_dim, num_lstm, batch_first=True)
        self.fc1 = nn.Linear(num_lstm * 2, num_dense)
        self.fc2 = nn.Linear(num_dense, num_dense)
        self.fc3 = nn.Linear(num_dense, 1)
        self.sig = nn.Sigmoid()

    def forward(self, q1, q2):
        # x = self.fc1(Variable(torch.randn((num_lstm * 2, num_dense)).cuda()))

        # print type(q1.data)

        # print type(q1)
        # print type(q2)

        # print q1.data.shape
        # print q2.data.shape

        # q1 = q1.view(q1.data.shape[1], q1.data.shape[0], -1)
        # q2 = q2.view(q2.data.shape[1], q2.data.shape[0], -1)
        # print q1.data.shape

        x1, _ = self.lstm1(q1)
        x2, _ = self.lstm2(q2)

        x1 = x1[:, -1, :]
        x2 = x2[:, -1, :]

        # print x1.data.shape
        # print x2.data.shape

        x = torch.cat([x1, x2], 1)
        # print type(x.data)
        # print type(x1.data)
        # print x.data.shape

        # print type(x)
        # print x.data.shape

        # print type(x1)
        # print type(x1)
        # q1 = Variable(torch.FloatTensor(q1))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.sig(x)

        print x.data.shape

        # print x.data.shape
        # x = self.fc1(Variable(torch.randn((num_lstm * 2, num_dense)).cuda()))

        # layer1 = nn.LSTM((max_seq_len, word_vector_dim), num_lstm)(q1)
        # layer2 = nn.LSTM((max_seq_len, word_vector_dim), num_lstm)(q2)
        # layer = nn.cat([layer1, layer2])
        # layer = nn.Linear(num_lstm * 2, num_dense)(layer)
        # layer = nn.Relu(num_lstm * 2, num_dense)(layer)
        # layer = nn.Linear(num_dense, num_dense)(layer)
        # layer = nn.Relu(num_lstm * 2, num_dense)(layer)
        # layer = nn.Sigmoid()(layer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-t", help="是否调试", action="store_true")
    args = parser.parse_args()
    if args.t:
        wv_path = './wv'

    global wv
    wv = KeyedVectors.load_word2vec_format(wv_path, binary=True)
    reader = csv.reader(open(train_path))

    data_loader = torch.utils.data.DataLoader(
        Dataset(train_path), batch_size=2000)

    i = data_loader.__iter__()

    model = Model().cuda()
    print model
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    for i in range(epoch_num):

        running_loss = 0
        for batch in tqdm(data_loader):

            outputs = model(Variable(batch[0]), Variable(batch[1]))
            # outputs = model(Variable(batch[0]), Variable(batch[1]))
            # optimizer.zero_grad()
            # for i in range(data_loader.batch_size):
            #     outputs = model(Variable(batch[0][i]), Variable(batch[1][i]))
            # sys.exit(1)
            # loss = criterion(outputs, data[:, -1])

            # running_loss += loss
