from code.CNN_code.Result_Loader import Result_Loader
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

if 1:
    result_obj = Result_Loader('loader', '')
    result_obj.result_destination_folder_path = '../../result/CNN_result/CNN_CIFAR_'
    result_obj.result_destination_file_name = 'prediction_result'

    #result_obj.fold_count = "Train"
    result_obj.load()
    print('Accuracy score:', accuracy_score(result_obj.data['true_y'], result_obj.data['pred_y']))
    print('Precision score:', precision_score(result_obj.data['true_y'], result_obj.data['pred_y'], average='weighted'))
    print("Recall score:", recall_score(result_obj.data['true_y'], result_obj.data['pred_y'], average='weighted'))
    print('F1 score:', f1_score(result_obj.data['true_y'], result_obj.data['pred_y'], average='weighted'))