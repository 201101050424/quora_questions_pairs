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
        content = json.dumps(content, encoding='gb18030',
                             ensure_ascii=False).encode('gb18030')
    content = str(content)
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

        self.content_list = []
        for line in open(train_preprocess_path):
            l = line.strip().split('\t')
            self.content_list.append(
                (
                    torch.LongTensor([int(a) for a in l[0].split(',')]),
                    torch.LongTensor([int(a) for a in l[1].split(',')]),
                    torch.LongTensor([int(l[2])])
                )
            )

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
        q1 = self.content_list[idx][0]
        q2 = self.content_list[idx][1]
        y = self.content_list[idx][2]
        return q1, q2, y


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        embedding_weight = np.load(embedding_weight_path)
        self.embedding = nn.Embedding(
            embedding_weight.shape[0], embedding_weight.shape[1])
        self.embedding.weight.data.copy_(
            torch.from_numpy(embedding_weight))
        self.lstm1 = nn.LSTM(word_vector_dim, num_lstm, batch_first=True)
        self.lstm2 = nn.LSTM(word_vector_dim, num_lstm, batch_first=True)
        self.fc1 = nn.Linear(num_lstm * 2, num_dense)
        self.fc2 = nn.Linear(num_dense, num_dense)
        self.fc3 = nn.Linear(num_dense, 2)
        # self.sig = nn.Sigmoid()

    def forward(self, q1, q2):
        
        q1 = self.embedding(q1)
        q2 = self.embedding(q2)

        x1, _ = self.lstm1(q1)
        x2, _ = self.lstm2(q2)

        x1 = x1[:, -1, :]
        x2 = x2[:, -1, :]

        x = torch.cat([x1, x2], 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x))

        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-t", help="是否调试", action="store_true")
    args = parser.parse_args()

    use_cuda = False
    use_cuda = True

    data_loader = torch.utils.data.DataLoader(
        Dataset(train_preprocess_path), batch_size=2000)

    if use_cuda:
        model = Model().cuda()
    else:
        model = Model()

    print model
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    if use_cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    for i in range(epoch_num):

        loss_list = []
        for batch in tqdm(data_loader):

            if use_cuda:
                input1 = batch[0].cuda()
                input2 = batch[1].cuda()
                label = batch[2][:].squeeze(1).cuda()
            else:
                input1 = batch[0]
                input2 = batch[1]
                label = batch[2][:].squeeze(1)


            output = model(Variable(input1), Variable(input2))

            label = Variable(label)

            optimizer.zero_grad()
            loss = F.nll_loss(output, label)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.data[0])

        print 'epoch:%d, avg_loss=%lf' % (i, sum(loss_list) / len(loss_list))
        if i % model_save_interval == 0:
            torch.save(model.state_dict(), '%s/model_%d' % (model_path, i))
