'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.RNN_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from datetime import datetime
import pickle


class Method_RNN_Generation(method, nn.Module):
    data = None
    wordDict = None
    max_epoch = 50
    learning_rate = 1e-3
    writer = SummaryWriter("runs/maxEpoch" + str(max_epoch) + '_lr' + str(
        learning_rate) + '_2GRU')
    # datetime.now().strftime("%b%d_%H-%M-%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, mName, mDescription, vocabSize):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.embedding = nn.Embedding(vocabSize, 100).cuda()
        self.gru = nn.GRU(100, 200, num_layers=2).cuda()
        self.linear = nn.Linear(200, vocabSize).cuda()
    def forward(self, x, hidden):
        '''Forward propagation'''
        x = self.embedding(x)
        x, hidden = self.gru(x, hidden)
        # print(hidden.shape)
        s, h = x.shape
        x = x.contiguous().view(s, h)
        x = F.dropout(x, 0.5)
        x = self.linear(x).view(s,-1)

        return x , hidden

    def init_hidden(self):
        return torch.zeros(2, 200).cuda()

    def train(self, X, y):

        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        step = 0
        for epoch in range(self.max_epoch):
            for joke_x, joke_y in zip(X, y):
                optimizer.zero_grad()
                # print(joke_x,joke_y)
                # pred_y = self.forward(torch.LongTensor(joke_x).cuda())
                # print(pred_y.shape)
                # loss = loss_function(pred_y, torch.LongTensor(joke_y).cuda())
                # _, pred_y = torch.max(pred_y, dim=2)
                loss = 0
                hidden = self.init_hidden()
                acc_count = 0
                for i in range(len(joke_x)):
                    pred_y, hidden = self.forward(torch.LongTensor([joke_x[i]]).cuda(), hidden)
                    # print(pred_y.shape)
                    loss += loss_function(pred_y, torch.LongTensor([joke_y[i]]).cuda())
                    if pred_y.max(1)[1].item() == joke_y[i]:
                        acc_count += 1
                self.writer.add_scalar('Loss', loss.item(),step)
                self.writer.add_scalar('Word Accuracy', acc_count/len(joke_x) , step)
                step += 1
                loss.backward()
                optimizer.step()
            print('epoch:', epoch, 'loss:', loss)
    def generate_joke(self, max_length=50):
        start = input("Input words(lower case):")
        start = start.lower()
        start = start.split(' ')
        # print(start)
        start = ['<start>'] + start
        for index, data in enumerate(start):
            start[index] = self.wordDict[data]
        hidden = self.init_hidden()
        for i in start[:-1]:
            pred_y, hidden = self.forward(torch.LongTensor([i]).cuda(), hidden)
        last_input = start[-1]
        for i in range(max_length):
            pred_y, hidden = self.forward(torch.LongTensor([last_input]).cuda(), hidden)
            last_input = pred_y.max(1)[1].tolist()[0]
            if last_input == 1:
                print()
                return
            print(self.wordDict[last_input], end=' ')
        print()


    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--saving model...')
        torch.save(self.state_dict(), '../../result/RNN_result/model')
        return
