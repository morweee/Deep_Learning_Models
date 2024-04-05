'''
Concrete IO class for a specific dataset
'''
import random

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from code.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp
from collections import defaultdict

class Dataset_Loader(dataset):
    data = None
    dataset_name = None
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(Dataset_Loader, self).__init__(dName, dDescription)

    def adj_normalize(self, mx):
        """normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        onehot_labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return onehot_labels

    def load(self):
        """Load citation network dataset"""
        print('Loading {} dataset...'.format(self.dataset_name))

        # load node data from file
        idx_features_labels = np.genfromtxt("{}/node".format(self.dataset_source_folder_path), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        onehot_labels = self.encode_onehot(idx_features_labels[:, -1])

        # load link data from file and build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        reverse_idx_map = {i: j for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}/link".format(self.dataset_source_folder_path), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

        # convert to pytorch tensors
        features = torch.FloatTensor(np.array(features.todense())).cuda()
        labels = torch.LongTensor(np.where(onehot_labels)[1]).cuda()
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        labels_dict = defaultdict(list)
        # i = index of the node, j = label
        # label_dict = a dict() with key = class_labels, values = list of index belonging to that class
        for i in range(len(onehot_labels)):
            for j in range(len(onehot_labels[i])):
                if onehot_labels[i][j] == 1:
                    labels_dict[j].append(i)

        # the following part, you can either put them into the setting class or you can leave them in the dataset loader
        # the following train, test, val index are just examples, sample the train, test according to project requirements
        idx_train = []
        idx_test = []
        idx_val = []

        if self.dataset_name == 'cora':
            for label in labels_dict:
                # print("count: ", count)
                label_arr = np.array(labels_dict[label])
                # in each label, randomly (shuffled) select 20 node instances for training
                len_data = len(labels_dict[label])
                train_indices = np.random.choice(len_data, 20, replace=False)
                idx_train = idx_train + label_arr[train_indices].tolist()

                remaining_data = np.delete(label_arr, train_indices)
                len_data = len(remaining_data)
                test_indices = np.random.choice(len_data, 150, replace=False)
                idx_test = idx_test + label_arr[test_indices].tolist()

                remaining_data = np.delete(label_arr, test_indices)
                len_data = len(remaining_data)
                val_indices = np.random.choice(len_data, 75)
                idx_val = idx_val + label_arr[val_indices].tolist()
                remaining_data = np.delete(label_arr, val_indices)
            print(len(idx_train), len(idx_test), len(idx_val))

        elif self.dataset_name == 'citeseer':
            for label in labels_dict:
                # print("count: ", count)
                label_arr = np.array(labels_dict[label])
                # in each label, randomly select 20 node instances for training
                len_data = len(labels_dict[label])
                train_indices = np.random.choice(len_data, 20, replace=False)
                idx_train = idx_train + label_arr[train_indices].tolist()
                # randomly select 200 node instances for testing
                remaining_data = np.delete(label_arr, train_indices)
                len_data = len(remaining_data)
                test_indices = np.random.choice(len_data, 200, replace=False)
                idx_test = idx_test + label_arr[test_indices].tolist()

                remaining_data = np.delete(label_arr, test_indices)
                len_data = len(remaining_data)
                val_indices = np.random.choice(len_data, 100)
                idx_val = idx_val + label_arr[val_indices].tolist()
                remaining_data = np.delete(label_arr, val_indices)
            print(len(idx_train), len(idx_test), len(idx_val))
        elif self.dataset_name == 'pubmed':
            for label in labels_dict:
                # turn the data in to np.array for convenience
                label_arr = np.array(labels_dict[label])
                # in each label, randomly select 20 node instances for training
                len_data = len(labels_dict[label])
                train_indices = np.random.choice(len_data, 20, replace=False)
                idx_train = idx_train + label_arr[train_indices].tolist()
                # randomly select 200 node instances for testing
                remaining_data = np.delete(label_arr, train_indices)
                len_data = len(remaining_data)
                test_indices = np.random.choice(len_data, 200, replace=False)
                idx_test = idx_test + label_arr[test_indices].tolist()

                remaining_data = np.delete(label_arr, test_indices)
                len_data = len(remaining_data)
                val_indices = np.random.choice(len_data, 100)
                idx_val = idx_val + label_arr[val_indices].tolist()
                remaining_data = np.delete(label_arr, val_indices)
            print(len(idx_train), len(idx_test), len(idx_val))

        #---- cora-small is a toy dataset I hand crafted for debugging purposes ---
        elif self.dataset_name == 'cora-small':
            idx_train = range(5)
            idx_val = range(5, 10)
            idx_test = range(5, 10)

        idx_train = torch.LongTensor(idx_train).cuda()
        idx_val = torch.LongTensor(idx_val).cuda()
        idx_test = torch.LongTensor(idx_test).cuda()
        # get the training nodes/testing nodes
        # train_x = features[idx_train]
        # val_x = features[idx_val]
        # test_x = features[idx_test]
        # print(train_x, val_x, test_x)

        train_test_val = {'idx_train': idx_train, 'idx_test': idx_test, 'idx_val': idx_val}
        graph = {'node': idx_map, 'edge': edges, 'X': features, 'y': labels, 'utility': {'A': adj, 'reverse_idx': reverse_idx_map}}
        return {'graph': graph, 'train_test_val': train_test_val}
