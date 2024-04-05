from collections import Counter
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def pad(l, content, width):
    l.extend([content] * (width - len(l)))
    return l


f = open('../../data/RNN_data/text_classification/processedData', 'rb')
data = pickle.load(f)
f.close()
counter = Counter()
maxLen = 0



rev_len = []



for pair in data['train']:
    rev_len.append(len(pair['text']))
    counter.update(pair['text'])
for pair in data['test']:
    rev_len.append(len(pair['text']))


pd.Series(rev_len).hist()
plt.show()
pd.Series(rev_len).describe()

wordDict = {}

print(len(counter))
encode = 2  # 0:'pad' 1:'unknown'
for i in counter.most_common(40000):
    wordDict[i[0]] = encode
    encode += 1

newData = {'train': [], 'test': []}

count = 0
maxLen = 500

for pair in data['train']:
    count += 1
    for i in range(len(pair['text'])):
        if pair['text'][i] in wordDict:
            pair['text'][i] = wordDict[pair['text'][i]]
        else:
            pair['text'][i] = 1
    pair['text'] = pad(pair['text'], 0, maxLen)
    pair['text'] = pair['text'][:maxLen]
    newData['train'].append(pair)
    print(count)

for pair in data['test']:
    count += 1
    for i in range(len(pair['text'])):
        if pair['text'][i] in wordDict:
            pair['text'][i] = wordDict[pair['text'][i]]
        else:
            pair['text'][i] = 1
    pair['text'] = pad(pair['text'], 0, maxLen)
    pair['text'] = pair['text'][:maxLen]
    newData['test'].append(pair)
    print(count)

newData['maxLen'] = maxLen
wordDict['<unk>'] = 1
wordDict['<pad>'] = 0
newData['wordDict'] = wordDict
f = open('../../data/RNN_data/text_classification/EncodedData40000_500', 'wb')
pickle.dump(newData, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()
