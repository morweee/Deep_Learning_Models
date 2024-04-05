'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np


class Setting_Train_Test_Data(setting):

    def load_run_save_evaluate(self):
        # load dataset
        loaded_data = self.dataset.load()

        X_train,  y_train = loaded_data['Train_X'], loaded_data['Train_y']
        # X_train = np.asarray(X_train)
        # y_train = np.asarray(y_train)


        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}}
        self.method.run()
        return

