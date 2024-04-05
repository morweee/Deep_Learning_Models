from code.CNN_code.Dataset_Loader_CNN import Dataset_Loader
from code.CNN_code.Method_CNN_ORL import Method_CNN_ORL
from code.CNN_code.Result_Saver import Result_Saver
#from code.MLP_code.Setting_KFold_CV import Setting_KFold_CV
from code.CNN_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.CNN_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

# ---- CNN script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(5)
    torch.manual_seed(5)
    # ------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('stage 3', '')
    data_obj.dataset_source_folder_path = '../../data/CNN_data/'
    data_obj.dataset_source_file_name = 'ORL'

    method_obj = Method_CNN_ORL('CNN', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/CNN_result/CNN_ORL_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('CNN_Train', '')
    # setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('CNN Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------


