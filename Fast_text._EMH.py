
# coding: utf-8

# In[20]:


import numpy as np

import pandas as pd

from collections import defaultdict

import keras
import keras.backend as K
from keras.layers import Dense, GlobalAveragePooling1D, Embedding
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
np.random.seed(7)


# In[21]:


import matplotlib
matplotlib.rcParams['figure.figsize'] = [10,8]
plt.style.use('fivethirtyeight')


# In[2]:


df = pd.read_csv('train.csv')
a2c = {'EAP': 0, 'HPL' : 1, 'MWS' : 2}
y = np.array([a2c[a] for a in df.author])
y = to_categorical(y)


# In[3]:


# counter = {name : defaultdict(int) for name in set(df.author)}
# for (text, author) in zip(df.text, df.author):
#     text = text.replace(' ', '')
#     for c in text:
#         counter[author][c] += 1

# chars = set()
# for v in counter.values():
#     chars |= v.keys()
    
# names = [author for author in counter.keys()]

# print('c ', end='')
# for n in names:
#     print(n, end='   ')
# print()
# for c in chars:    
#     print(c, end=' ')
#     for n in names:
#         print(counter[n][c], end=' ')
#     print()


# In[4]:


def preprocess(text):
    text = text.replace("' ", " ' ")
    signs = set(',.:;"?!')
    prods = set(text) & signs
    if not prods:
        return text

    for sign in prods:
        text = text.replace(sign, ' {} '.format(sign) )
    return text


# In[5]:


def create_docs(df, n_gram_max=2):
    def add_ngram(q, n_gram_max):
            ngrams = []
            for n in range(2, n_gram_max+1):
                for w_index in range(len(q)-n+1):
                    ngrams.append('--'.join(q[w_index:w_index+n]))
            return q + ngrams
        
    docs = []
    for doc in df.text:
        doc = preprocess(doc).split()
        docs.append(' '.join(add_ngram(doc, n_gram_max)))
    
    return docs


# In[6]:


min_count = 2

docs = create_docs(df)
tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(docs)
num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])

tokenizer = Tokenizer(num_words=num_words, lower=False, filters='')
tokenizer.fit_on_texts(docs)
docs = tokenizer.texts_to_sequences(docs)

maxlen = 256

docs = pad_sequences(sequences=docs, maxlen=maxlen)


# In[7]:


input_dim = np.max(docs) + 1
# embedding_dims = 20


# In[12]:


def create_model(embedding_dims, optimizer='adam'):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


# In[17]:


epochs = 10
x_train, x_test, y_train, y_test = train_test_split(docs, y, test_size=0.2)

model = create_model(embedding_dims = 200)
hist = model.fit(x_train, y_train,
                 batch_size=16,
                 validation_data=(x_test, y_test),
                 epochs=epochs,
                 callbacks=[EarlyStopping(patience=2, monitor='val_loss')])


# In[11]:


hist_20 = hist.history['acc']
hist_20 = hist.history['val_acc']


# In[14]:


hist_100 = hist.history['acc']
hist_100 = hist.history['val_acc']


# In[18]:


hist_200 = hist.history['acc']
hist_200 = hist.history['val_acc']


# In[25]:


plt.plot(hist_20, label = 'Embed Size = 20')
plt.plot(hist_100, label = 'Embed Size = 100')
plt.plot(hist_200, label = 'Embed Size = 200')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuray wrt Embeding Size')
plt.show()

