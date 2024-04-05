import pickle

f = open('../../data/RNN_data/text_generation/data')

data = f.readlines()
f.close()
newData = []
for line in data:
    line = line[:-2].split(',"')[1]
    newData.append(line)
    # print(line)
# print(data[1::1])
# print(newData)
newData = newData[1:]
word2index = {0: '<start>', 1: '<end>', '<start>': 0, '<end>': 1}
index = 2  # index=0 -> <start> index=1 -> <end>
for lineIndex in range(len(newData)):
    newData[lineIndex] = newData[lineIndex].lower()
    newData[lineIndex] = newData[lineIndex].split(' ')
    for WordIndex in range(len(newData[lineIndex])):
        if newData[lineIndex][WordIndex] not in word2index:
            word2index[newData[lineIndex][WordIndex]] = index
            word2index[index] = newData[lineIndex][WordIndex]
            index += 1
        newData[lineIndex][WordIndex] = word2index[newData[lineIndex][WordIndex]]


train_x = []
train_y = []
for (index, data) in enumerate(newData):
    newData[index].append(1)
    newData[index] = [0] + newData[index]
    train_x.append(newData[index][:-1])
    train_y.append(newData[index][1:])

    # print(len(newData[index]))
# for i in newData[50]:
#     print(i, word2index[i])
encodedData = {'train_x': train_x, 'train_y': train_y, 'wordIndex': word2index}
print(len(encodedData['train_y']))
# print(newData[50])
f = open('../../data/RNN_data/text_generation/processedData', 'wb')
pickle.dump(encodedData, f)
f.close()

# f = open('../../data/RNN_data/text_generation/data')
