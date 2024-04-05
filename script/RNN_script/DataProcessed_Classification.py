import pickle
import os
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer

# nltk.download('punkt')
# nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))

    stop_words = set(stopwords.words('english'))
    text_tokens = nltk.word_tokenize(text)
    text = [word for word in text_tokens if not word in stop_words]

    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]

    return text

data = {}
data['train']=[]
data['test']=[]
count = 0

# train Pos
data_folder_path = '../../data/RNN_data/text_classification/'
data_file_path = ['train/pos', 'train/neg', 'test/pos', 'test/neg']

for i in data_file_path:
    allFileName = os.listdir(data_folder_path+i)
    dataset = i.split('/')[0]
    label = i.split('/')[1]
    for fileName in allFileName[:2500]:
        f = open(data_folder_path+i+'/'+fileName, encoding="utf-8")
        text = f.read()
        f.close()
        text = preprocess_text(text)
        pair = {'text':text, 'label':1 if label=='pos' else 0}
        data[dataset].append(pair)
        count += 1
        print(count)

f = open('../../data/RNN_data/text_classification/ToyProcessedData', 'wb')
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()