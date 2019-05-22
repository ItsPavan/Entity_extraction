#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
data_path = 'A:\\zycus-test\\workspace-alewo4\\data'
data = pd.read_csv(data_path+"\\"+"export_dataframe.csv", encoding="latin1")
data.head(100)

#%%
words = set(list(data['word'].values))
words.add('PADword')
n_words = len(words)
n_words

#%%
tags = list(set(data["tag"].values))
n_tags = len(tags)
n_tags

#%%
class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["word"].values.tolist(),s["tag"].values.tolist())]
        self.grouped = self.data.groupby("sentence").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["sentence{}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

#%%
getter = SentenceGetter(data)
sent = getter.get_next()
print(sent)

#%%
sentences = getter.sentences
print(len(sentences))
print(sentences)
#%%
largest_sen = max(len(sen) for sen in sentences)
print('biggest sentence has {} words'.format(largest_sen))
for i in sentences:
    if len(i) == largest_sen:
        print(i)

#%%
%matplotlib inline
plt.hist([len(sen) for sen in sentences], bins= 50)
plt.show()

#%%
words2index = {w:i for i,w in enumerate(words)}
tags2index = {t:i for i,t in enumerate(tags)}

#%%
max_len = 50
X = [[w[0]for w in s] for s in sentences]
new_X = []
for seq in X:
    new_seq = []
    for i in range(max_len):
        try:
            new_seq.append(seq[i])
        except:
            new_seq.append("PADword")
    new_X.append(new_seq)
new_X[50]

#%%
from keras.preprocessing.sequence import pad_sequences
y = [[tags2index[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tags2index["O"])
y[15]

#%%
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(new_X, y, test_size=0.1, random_state=2018)

#%%
batch_size = 32
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
sess = tf.Session()
K.set_session(sess)

#%%
elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())