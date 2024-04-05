from code.RNN_code.Dataset_Loader import Dataset_Loader
from code.RNN_code.Method_RNN_Classification import Method_RNN_Classification
from code.RNN_code.Result_Saver import Result_Saver
from code.RNN_code.Setting_KFold_CV import Setting_KFold_CV
from code.RNN_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.RNN_code.Setting import Setting_Train_Test_Data
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
    data_obj.dataset_source_folder_path = '../../data/RNN_data/text_classification/'
    data_obj.dataset_source_file_name = 'EncodedData40000_500'

    # load weight before init RNN Method
    # wvmodel = gensim.models.KeyedVectors.load_word2vec_format('glove.6B.100d.w2vformat.txt', binary=False,
    #                                                           encoding='utf-8')
    wordDict = data_obj.loadWordDict()
    ## map golve pretrain weight to pytorch embedding pretrain weight
    embed_size = 100
    # weight = torch.zeros(40002, embed_size)  # given 0 if the word is not in glove
    # for i in range(len(wvmodel.index_to_key)):
    #     try:
    #         index = wordDict[wvmodel.index_to_key[i]]# transfer to our word2ind
    #
    #     except:
    #         continue
    #     weight[index, :] = torch.from_numpy(wvmodel.get_vector(wvmodel.index_to_key[i]))
    # load weight before init RNN Method


    method_obj = Method_RNN_Classification('RNN', '', embed_size)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/RNN_result/RNN_Classification_'
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
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('RNN classification Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------
    

    