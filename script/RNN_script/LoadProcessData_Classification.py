import pickle

f = open('../../data/RNN_data/text_classification/processedData', 'rb')
data = pickle.load(f)
print(data['test'][12500])