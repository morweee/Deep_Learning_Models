'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score
import numpy as np
import torch

class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        # print('evaluating performance...')
        if type(self.data['pred_y']) == np.ndarray:
            self.data['pred_y'] = np.rint(self.data['pred_y'])
        else:
            self.data['pred_y'] = torch.round(self.data['pred_y'])
        if type(self.data['true_y']) == np.ndarray and type(self.data['pred_y']) == np.ndarray:
            temp = accuracy_score(self.data['true_y'], self.data['pred_y'])
            print(temp)
            return temp
        elif type(self.data['true_y']) == np.ndarray:
            return accuracy_score(self.data['true_y'], self.data['pred_y'].cpu())
        else:
            # print(self.data['true_y'], self.data['pred_y'])
            return accuracy_score(self.data['true_y'].tolist(), self.data['pred_y'].tolist())
        