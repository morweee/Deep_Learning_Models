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
        return data['wordDict']
    def load(self):
        print('loading data...')
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()
        test_x = []
        test_y = []
        train_x = []
        train_y = []
        for pair in data['test']:
            # print(type(pair['image']))
            test_x.append(pair['text'])
            test_y.append(pair['label'])

        for pair in data['train']:
            train_x.append(pair['text'])
            train_y.append(pair['label'])
        random.Random(4).shuffle(train_x)
        random.Random(4).shuffle(train_y)
        random.Random(4).shuffle(test_x)
        random.Random(4).shuffle(test_y)
        print("Train Size:", len(train_x),"Test Size:", len(test_y), "maxLen:", data['maxLen'])
        return {'Train_X': train_x, 'Train_y': train_y, 'Test_X': test_x, 'Test_y': test_y,'maxLen': data['maxLen']}