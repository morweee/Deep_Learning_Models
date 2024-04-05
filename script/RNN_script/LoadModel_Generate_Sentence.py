import torch
from code.RNN_code.Method_RNN_Generation import Method_RNN_Generation
from code.RNN_code.Dataset_Loader_Generation import Dataset_Loader

data_obj = Dataset_Loader('stage_4', '')
data_obj.dataset_source_folder_path = '../../data/RNN_data/text_generation/'
data_obj.dataset_source_file_name = 'processedData'

# load weight before init RNN Method
wordDict = data_obj.loadWordDict()
vocabSize = len(wordDict) // 2

model = Method_RNN_Generation('RNN', '', vocabSize)
model. wordDict = wordDict
model.load_state_dict(torch.load('../../result/RNN_result/model'))
while True:
    model.generate_joke()
