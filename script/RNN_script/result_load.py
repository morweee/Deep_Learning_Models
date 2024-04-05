from code.RNN_code.Result_Loader import Result_Loader
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np

result = Result_Loader("loader", "")
result.result_destination_folder_path = '../../result/RNN_result/RNN_Classification_'
result.result_destination_file_name = 'prediction_result'
result.fold_count="None"
result.load()
# print("Accuracy-Score:", accuracy_score(result.data['true_y'], result.data['pred_y'].cpu()))
# print("Precision-Score:", precision_score(result.data['true_y'], result.data['pred_y'].cpu(), average='weighted', zero_division=0))
# print("Recall-Score:", recall_score(result.data['true_y'], result.data['pred_y'].cpu(), average='weighted'))
# print("F1-Score:", f1_score(result.data['true_y'], result.data['pred_y'].cpu(), average='weighted'))
result.data['pred_y'] = np.rint(result.data['pred_y'])
print("Accuracy-Score:", accuracy_score(result.data['true_y'], result.data['pred_y']))
print("Precision-Score:", precision_score(result.data['true_y'], result.data['pred_y'], average='weighted', zero_division=0))
print("Recall-Score:", recall_score(result.data['true_y'], result.data['pred_y'], average='weighted'))
print("F1-Score:", f1_score(result.data['true_y'], result.data['pred_y'], average='weighted'))
