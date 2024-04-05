'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pickle

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        for instance in data['train']:
            image_matrix = instance['image']
            image_label = instance['label']
            X_train.append(image_matrix)
            y_train.append(image_label-1)
        for instance in data['test']:
            image_matrix = instance['image']
            image_label = instance['label']
            X_test.append(image_matrix)
            y_test.append(image_label-1)
        return {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}