import math
import torch
from torch import nn
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from code.base_class.method import method
from code.GCN_code.Evaluate_Accuracy import Evaluate_Accuracy
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from torch_geometric.nn import GCNConv
import pickle

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
def calculate_acc(y_pred, y_true):
    count = 0
    for i, j in zip(y_pred, y_true):
        if i == j:
            count += 1
    return count / len(y_pred)
class Method_GCN_Citeseer(method, nn.Module):
    data = None
    max_epoch = 100
    learning_rate = 0.01
    weight_decay = 0.0085 # 0.0085
    writer = SummaryWriter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 10
    np.random.seed(seed)
    torch.manual_seed(seed)
    def __init__(self, mName, mDescription, features_n, class_n):
        nn.Module.__init__(self)
        method.__init__(self, mName, mDescription)
        self.conv1 = GraphConvolution(features_n, 64).cuda()
        self.conv2 = GraphConvolution(64, class_n).cuda()
        # ------------ using torch_geometric ------------------
        # self.GCN_conv1 = GCNConv(features_n, 128).cuda()
        # self.GCN_conv2 = GCNConv(128, class_n).cuda()
        # self.dropout = nn.Dropout(p=0.5).cuda()

    def forward(self, x, adj, train=False):
        # x = F.dropout(x, training=train, p=0.5)
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, training=train)
        x = self.conv2(x, adj)
        output = F.softmax(x, dim=1)
        # ------------ using torch_geometric ------------------
        # edge = torch.LongTensor(self.data['graph']['edge']).cuda()
        # edge = torch.permute(edge, (1, 0))
        # x = self.GCN_conv1(x, edge)
        # x = F.relu(x)
        # x = F.dropout(x, training=train)
        # output = self.GCN_conv2(x, edge)

        return output

    def train(self, x, y):
        train_mask = self.data['train_test_val']['idx_train']
        test_mask = self.data['train_test_val']['idx_test']
        valid_mask = self.data['train_test_val']['idx_val']
        adj = self.data['graph']['utility']['A'].cuda()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_function = nn.CrossEntropyLoss()
        for epoch in range(self.max_epoch):
            optimizer.zero_grad()
            y_pred = self.forward(x, adj)
            train_loss = loss_function(y_pred[train_mask], y[train_mask])
            train_loss.backward()
            optimizer.step()
            # Draw image
            train_acc = calculate_acc(y_pred[train_mask].max(1)[1], y[train_mask])
            valid_loss = loss_function(y_pred[valid_mask], y[valid_mask])
            valid_acc = calculate_acc(y_pred[valid_mask].max(1)[1], y[valid_mask])
            test_loss = loss_function(y_pred[test_mask], y[test_mask])
            test_acc = calculate_acc(y_pred[test_mask].max(1)[1], y[test_mask])
            self.writer.add_scalar('Loss/train', train_loss.item(), epoch)
            self.writer.add_scalar('Loss/test', test_loss.item(), epoch)
            self.writer.add_scalar('Loss/valid', valid_loss.item(), epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/test', test_acc, epoch)
            self.writer.add_scalar('Accuracy/valid', valid_acc, epoch)
            if epoch % 5 == 0:
                print('Epoch', epoch, 'Test_Accuracy:', test_acc, 'Valid_Accuracy:', valid_acc, 'Train_Accuracy:',
                      train_acc)
                print('         Test Loss:', test_loss.item(), 'Valid Loss:', valid_loss.item(), 'Train Loss:',
                      train_loss.item())
            # if valid_loss.item() > self.last_valid_loss:
            #     print('Stop')
            #     break
            # else:
            #     self.last_valid_loss = valid_loss.item()
    def test(self, X):
        test_mask = self.data['train_test_val']['idx_test']
        adj = self.data['graph']['utility']['A'].cuda()
        y_pred = self.forward(X, adj)
        return y_pred[test_mask].max(1)[1]
    def train_score(self, X, y):
        train_mask = self.data['train_test_val']['idx_train']
        adj = self.data['graph']['utility']['A'].cuda()
        y_pred = self.forward(X, adj)
        print("Accuracy-Score:", accuracy_score(y_pred[train_mask].max(1)[1].cpu(), y[train_mask].cpu()))
        print("Precision-Score:",
              precision_score(y_pred[train_mask].max(1)[1].cpu(), y[train_mask].cpu(), average='weighted'))
        print("Recall-Score:",
              recall_score(y_pred[train_mask].max(1)[1].cpu(), y[train_mask].cpu(), average='weighted'))
        print("F1-Score:", f1_score(y_pred[train_mask].max(1)[1].cpu(), y[train_mask].cpu(), average='weighted'))
        f = open('../../result/GCN_result/GCN_Cora_' + 'prediction_result' + '_Train', 'wb')
        pickle.dump({'pred_y': y_pred[train_mask].max(1)[1], 'true_y': y[train_mask]}, f)
        f.close()
        return 0

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['graph']['X'], self.data['graph']['y'])
        self.train_score(self.data['graph']['X'], self.data['graph']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['graph']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['graph']['y'][self.data['train_test_val']['idx_test']]}