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
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from datetime import datetime
import pickle


class Method_RNN_Classification(method, nn.Module):
    data = None
    max_epoch = 10
    learning_rate = 1e-2
    VocabSize = 40000
    maxLen = 500
    writer = SummaryWriter()
    # datetime.now().strftime("%b%d_%H-%M-%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, mName, mDescription, embed_size):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.embedding = nn.Embedding(40002,embed_size).cuda()
        self.encoder = nn.RNN(input_size=embed_size, hidden_size=100,
                               num_layers=3, bidirectional=True,
                               dropout=0.3).cuda()
        self.dropout = nn.Dropout(0.5).cuda()  # Dropout layer after linear
        # self.batch_norm1 = nn.BatchNorm1d(128 * 4).cuda()  # BatchNorm layer
        self.linear1 = nn.Linear(100 * 4, 200).cuda()
        self.relu = nn.ReLU().cuda()
        self.linear2 = nn.Linear(200, 1).cuda()
        self.sig = nn.Sigmoid().cuda()
    def forward(self, x):
        '''Forward propagation'''
        embeddings = self.embedding(x)
        states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
        encoding = torch.cat([states[0], states[-1]], dim=1) #if it's bidirectional, choose first and last output
        #encoding = self.batch_norm1(encoding)
        #encoding = self.dropout(encoding)
        outputs = self.relu(self.linear1(encoding))
        #outputs = self.dropout(outputs)
        outputs = self.linear2(outputs)
        outputs = self.sig(outputs)

        return outputs

    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.no_layer, batch_size, self.hidden_dim)).cuda()
        c0 = torch.zeros((self.no_layer, batch_size, self.hidden_dim)).cuda()
        hidden = (h0, c0)
        return hidden

    def train(self, X, y):

        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate)  # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.BCELoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
            epoch_acc = 0
            curr_batch = 0
            valid_acc = 0
            for batch_x, batch_y in zip(np.split(X, len(X) / 100), np.split(y, len(y) / 100)):
                optimizer.zero_grad()
                if curr_batch < 240:
                    y_pred = self.forward(torch.LongTensor(batch_x).cuda()).squeeze(1)
                    y_true = torch.FloatTensor(batch_y).cuda()
                    train_loss = loss_function(y_pred, y_true)
                    train_loss.backward()
                    optimizer.step()
                    accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred}
                    accuracy = accuracy_evaluator.evaluate()
                    self.writer.add_scalar('Loss', train_loss.item(), epoch * 240 + curr_batch)
                    self.writer.add_scalar('Accuracy', accuracy, epoch * 240 + curr_batch)
                    epoch_acc += accuracy
                else:
                    y_pred = self.forward(torch.LongTensor(batch_x).cuda())
                    y_true = torch.FloatTensor(batch_y).cuda()
                    accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred}
                    accuracy = accuracy_evaluator.evaluate()
                    valid_acc += accuracy
                # sche.step()
                curr_batch += 1
                if curr_batch % 10 == 0:
                    # print(y_pred, y_true)
                    print('Batch:', curr_batch, 'Accuracy:', accuracy, 'Loss:', train_loss.item())
            print('Epoch:', epoch, 'Test_Accuracy:', epoch_acc / 240, 'Valid_Accuracy:', valid_acc / 10)

    def test(self, X):
        output = np.array([])
        for batch_x in np.split(X, len(X)/100):
            y_pred = self.forward(torch.LongTensor(batch_x).cuda()).squeeze(1)
            output = np.append(output, y_pred.detach().cpu().numpy())
        return output

    def train_score(self):
        y = self.data['train']['y']
        output = np.array([])
        for batch_x in np.split(self.data['train']['X'], len(self.data['train']['X'])/100):
            y_pred = self.forward(torch.LongTensor(batch_x).cuda()).squeeze(1)
            output = np.append(output, y_pred.detach().cpu().numpy())
        output = np.rint(output)
        #print(output,y)
        print("Accuracy-Score:", accuracy_score(y, output.tolist()))
        print("Precision-Score:", precision_score(y, output.tolist(), average='weighted', zero_division=0))
        print("Recall-Score:", recall_score(y, output.tolist(), average='weighted'))
        print("F1-Score:", f1_score(y, output.tolist(), average='weighted'))
        f = open('../../result/RNN_result/RNN_classification_' + 'prediction_result' + '_Train', 'wb')
        pickle.dump({'pred_y': output.tolist(), 'true_y': y}, f)
        f.close()
        return 0

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        self.train_score()
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
