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

        X_train, X_test, y_train, y_test = loaded_data['Train_X'], loaded_data['Test_X'], loaded_data['Train_y'], loaded_data['Test_y']
        X_train, X_test = np.asarray(X_train), np.asarray(X_test)
        y_train, y_test = np.asarray(y_train), np.asarray(y_test)


        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None

