import pickle

f = open('../../data/RNN_data/text_generation/processedData', 'rb')
data = pickle.load(f)
train = data['train_x']
word2index = data['wordIndex']
# for line in data['train_x']:
#     for i in line:
#         print(i, word2index[i])
print(data['wordIndex'][6472])