'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import numpy as np
import pickle
import random

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    def loadWordDict(self):
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()
        return data['wordIndex']

    def load(self):
        print('loading data...')
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()
        train_x = data['train_x']
        train_y = data['train_y']
        print("Train Size:", len(train_x))
        return {'Train_X': train_x, 'Train_y': train_y}