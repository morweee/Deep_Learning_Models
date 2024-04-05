import pickle

f = open('../../data/RNN_data/text_classification/encodedData', 'rb')
data = pickle.load(f)
print(data['tr'][100])