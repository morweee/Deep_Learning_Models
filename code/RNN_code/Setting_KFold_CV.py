'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from sklearn.model_selection import KFold
import numpy as np

class Setting_KFold_CV(setting):
    fold = 10
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()
        
        kf = KFold(n_splits=self.fold, shuffle=True)

        X_test, y_test = loaded_data['Test_X'], loaded_data['Test_y']
        X_test, y_test = np.array(X_test), np.array(y_test)
        fold_count = 0
        score_list = []
        self.method.oriTrainData = {'train': {'X': np.array(loaded_data['Train_X']), 'y': np.array(loaded_data['Train_y'])}}
        for train_index, test_index in kf.split(loaded_data['Train_X']):
            fold_count += 1
            print('************ Fold:', fold_count, '************')
            X_train, y_train = np.array(loaded_data['Train_X'])[train_index], np.array(loaded_data['Train_y'])[train_index]

            # run MethodModule
            self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
            learned_result = self.method.run()
            
            # save raw ResultModule
            self.result.data = learned_result
            self.result.fold_count = fold_count
            self.result.save()
            
            self.evaluate.data = learned_result
            score_list.append(self.evaluate.evaluate())
        
        return np.mean(score_list), np.std(score_list)