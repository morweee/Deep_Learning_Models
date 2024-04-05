from code.RNN_code.Dataset_Loader_Generation import Dataset_Loader
from code.RNN_code.Method_RNN_Generation import Method_RNN_Generation
from code.RNN_code.Result_Saver import Result_Saver
from code.RNN_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.RNN_code.Setting_Generation import Setting_Train_Test_Data
import numpy as np
import torch
import gensim

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('stage_4', '')
    data_obj.dataset_source_folder_path = '../../data/RNN_data/text_generation/'
    data_obj.dataset_source_file_name = 'processedData'

    # load weight before init RNN Method
    wordDict = data_obj.loadWordDict()
    vocabSize = len(wordDict)//2


    method_obj = Method_RNN_Generation('RNN', '', vocabSize)
    method_obj.wordDict = wordDict

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/RNN_result/RNN_Generation_'
    result_obj.result_destination_file_name = 'prediction_result'

    # setting_obj = Setting_KFold_CV('k fold cross validation', '')
    setting_obj = Setting_Train_Test_Data("Train", "Training Set")
    # in_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('************ Finish ************')
    # ------------------------------------------------------
    

    