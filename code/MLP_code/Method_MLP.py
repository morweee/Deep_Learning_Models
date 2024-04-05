'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.MLP_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pickle

class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 20
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 5e-4

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc_layer_1 = nn.Linear(784, 256)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU()

        self.fc_layer_2 = nn.Linear(256, 128)
        self.activation_func_2 = nn.ReLU()

        self.fc_layer_3 = nn.Linear(128, 64)
        self.activation_func_3 = nn.ReLU()

        self.fc_layer_4 = nn.Linear(64, 10)
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.activation_func_4 = nn.Softmax(dim=1)


    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        h1 = self.activation_func_1(self.fc_layer_1(x))
        # output layer result
        # self.fc_layer_2(h) will be a nx2 tensor
        # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
        h2 = self.activation_func_2(self.fc_layer_2(h1))
        h3 = self.activation_func_3(self.fc_layer_3(h2))
        y_pred = self.activation_func_4(self.fc_layer_4(h3))

        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        self.writer = SummaryWriter()
        self.curr_batch = 0
        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            for batch_x, batch_y in zip(np.split(X, len(X) / 100), np.split(y, len(y) / 100)):
                # print(batch_x[0],batch_y[0])
                y_pred = self.forward(torch.FloatTensor(batch_x))
                y_true = torch.LongTensor(batch_y)
                train_loss = loss_function(y_pred, y_true)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                self.curr_batch += 1
                # Plot
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                accuracy = accuracy_evaluator.evaluate()

                # print('Epoch:', epoch, 'Accuracy:', accuracy, 'Loss:', train_loss.item())
                if self.curr_batch % 50 == 0:

                    print('Batch:', self.curr_batch, 'Accuracy:', accuracy, 'Loss:', train_loss.item())

            if epoch%1 == 0:
                y_pred = self.forward(torch.FloatTensor(X))
                y_true = torch.LongTensor(y)
                train_loss = loss_function(y_pred, y_true)
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                self.writer.add_scalar('Loss', train_loss.item(), epoch)
                self.writer.add_scalar('Accuracy', accuracy_evaluator.evaluate(), epoch)
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())
    
    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]
    def train_metric(self):
        y_pred = self.forward(torch.FloatTensor(self.data['train']['X']))
        print("Accuracy-Score:", accuracy_score(self.data['train']['y'], y_pred.max(1)[1]))
        print("Precision-Score:",
              precision_score(self.data['train']['y'], y_pred.max(1)[1], average='weighted'))
        print("Recall-Score:",
              recall_score(self.data['train']['y'], y_pred.max(1)[1], average='weighted'))
        print("F1-Score:", f1_score(self.data['train']['y'], y_pred.max(1)[1], average='weighted'))
        f = open('../../result/MLP_result/MLP_' + 'prediction_result' + '_Train', 'wb')
        pickle.dump({'pred_y': y_pred.max(1)[1], 'true_y': self.data['train']['y']}, f)
        f.close()
        return 0
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        self.train_metric()
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
            