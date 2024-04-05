'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.CNN_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pickle

class Method_CNN_CIFAR(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 10
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    writer = SummaryWriter()
    curr_batch = 0
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # image: 32*32
    # o = (32 + 2 - 3)/1 + 1 = 32

    # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        def conv(in_channels, out_channels, pool):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                      nn.BatchNorm2d(out_channels),
                      nn.ReLU(inplace=True)]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        def linear(input, output):
            layers = [nn.Linear(input, output),
                      nn.ReLU(inplace=True)]
            return nn.Sequential(*layers)

        self.conv1 = conv(3, 64, False)
        self.conv2 = conv(64, 128, False)
        self.res1 = nn.Sequential(conv(128, 128, False), conv(128, 128, False))
        self.conv4 = conv(128, 256, True)
        self.res2 = nn.Sequential(conv(256, 256, False), conv(256, 256, False))
        self.conv6 = conv(256, 128, True)
        self.fc1 = linear(128 * 8 * 8, 512)
        self.fc2 = linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

        #def convolution_relu(in_channels, out_channels):
        #    projection_layer = []
        #    projection_layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        #    projection_layer.append(nn.BatchNorm2d(out_channels))
        #    projection_layer.append(nn.ReLU(inplace=True))
        #    return nn.Sequential(*projection_layer)
        #def linear(in_features, out_features):
        #    projection_layer = []
        #    projection_layer.append(nn.Linear(in_features, out_features))
        #    projection_layer.append(nn.ReLU(inplace=True))
        #    return nn.Sequential(*projection_layer)
#
        #self.convolution_layer_1 = convolution_relu(in_channels=3, out_channels=64)
        #self.convolution_layer_2 = convolution_relu(in_channels=64, out_channels=128)
        #self.resNet = nn.Sequential(convolution_relu(128, 128), convolution_relu(128, 128))
        #self.convolution_layer_3 = convolution_relu(in_channels=128, out_channels=256)
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#
        #self.fc_layer_1 = linear(16*16*256, 512)
        #self.fc_layer_2 = linear(512, 128)
        #self.fc_layer_3 = nn.Linear(128, 10)
        ##self.softMAX = nn.Softmax(dim=1)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x
        x = self.conv4(x)
        x = self.res2(x) + x
        x = self.conv6(x)
        # print(x.shape)
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
        #x = self.convolution_layer_1(x)
        #x = self.convolution_layer_2(x)
        #x = self.resNet(x) + x
        #x = self.convolution_layer_3(x)
        #x = self.pool(x)
        #x = x.view(-1, 16*16*256)
        #x = self.fc_layer_1(x)
        #x = self.fc_layer_2(x)
        #x = self.fc_layer_3(x)
        ##y_pred = self.softMAX(x)
        #return x #y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            self.curr_batch = 0
            for batch_x, batch_y in zip(np.split(X, len(X)/32), np.split(y, len(y)/32)):
                # print(batch_x[0],batch_y[0])
                batch_x = batch_x.transpose(0,3,2,1)
                y_pred = self.forward(torch.FloatTensor(batch_x))
                y_true = torch.LongTensor(batch_y)
                train_loss = loss_function(y_pred, y_true)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
            # Plot
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                accuracy = accuracy_evaluator.evaluate()
                self.writer.add_scalar('Loss', train_loss.item(), self.curr_batch)
                self.writer.add_scalar('Accuracy', accuracy_evaluator.evaluate(), self.curr_batch)
                self.curr_batch += 1

                # print('Epoch:', epoch, 'Accuracy:', accuracy, 'Loss:', train_loss.item())
                if self.curr_batch % 25 == 0:
                    print('Batch:', self.curr_batch, 'Accuracy:', accuracy, 'Loss:', train_loss.item())
                #print("a mini batch done")
            #if epoch % 1 == 0:
                #y_pred = self.forward(torch.FloatTensor(X))
                #y_true = torch.LongTensor(y)
                #train_loss = loss_function(y_pred, y_true)
                #accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
            print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

    def test(self, X):
        # do the testing, and result the result
        output = np.array([])
        for batch in np.split(self.data['train']['X'], len(self.data['train']['X']) / 100):
            batch = batch.transpose(0, 3, 2, 1)
            y_pred = self.forward(torch.FloatTensor(batch))
            output = np.append(output, y_pred.max(1)[1].numpy())
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return output
    def train_metric(self):
        output = np.array([])
        for batch in np.split(self.data['train']['X'], len(self.data['train']['X'])/100):
            batch = batch.transpose(0, 3, 2, 1)
            y_pred = self.forward(torch.FloatTensor(batch))
            output = np.append(output, y_pred.max(1)[1].numpy())
        print('training dataset score:')
        print("Accuracy-Score:", accuracy_score(self.data['train']['y'], output))
        print("Precision-Score:",
              precision_score(self.data['train']['y'], output, average='weighted'))
        print("Recall-Score:",
              recall_score(self.data['train']['y'], output, average='weighted'))
        print("F1-Score:", f1_score(self.data['train']['y'], output, average='weighted'))
        f = open('../../result/CNN_result/CNN_ORL_' + 'prediction_result' + '_Train', 'wb')
        pickle.dump({'pred_y': y_pred.max(1)[1], 'true_y': self.data['train']['y']}, f)
        f.close()
        return 0
    def run(self):
        print('method running...')
        print('--start training...')
        print(self.data['train']['X'].shape)
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        self.train_metric()
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
            