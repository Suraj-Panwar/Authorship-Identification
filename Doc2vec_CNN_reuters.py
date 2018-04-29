
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
import scipy.io as sio
from sklearn.cross_validation import train_test_split
import gensim


# In[31]:


import matplotlib
matplotlib.rcParams['figure.figsize'] = [10,8]
plt.style.use('fivethirtyeight')


# In[2]:


######################
# Importing Libraries#
######################
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import nltk
import gensim
from gensim.models import Doc2Vec
import re
from collections import namedtuple
import nltk
import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from sklearn.decomposition import TruncatedSVD as svd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
stop_words = set(stopwords.words('english'))
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from keras.utils import np_utils


# In[3]:


df = pd.read_csv('C:\\Users\\suraj\\Desktop\\NLU\\NLU project\\reuters.csv')
X_train, X_test, y_train, y_test = train_test_split(df.text, df.author)


# In[4]:


###########################
# Preprocessing Text Data #
##########################

doc = []
for i, item in enumerate(X_train):
    words = gensim.utils.simple_preprocess(item)
    #words = words[10:]
    words = [w for w in words if w not in stop_words]# if len(w) >2
    doc.append(gensim.models.doc2vec.TaggedDocument(words, [i]))
    
doc_1 = []
for i, item in enumerate(X_test):
    words = gensim.utils.simple_preprocess(item)
    #words = words[10:]
    words = [w for w in words if w not in stop_words]# if len(w) >2
    doc_1.append(gensim.models.doc2vec.TaggedDocument(words, [i]))


# In[55]:


#######################
# Creating Doc Vectors#
#######################

model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=3, epochs=10)
model.build_vocab(doc)
model.train(doc, total_examples = model.corpus_count, epochs = model.epochs)


# In[56]:


vec = []
for doc_12 in doc:
    vec.append(model.infer_vector(doc_12.words))


# In[57]:


vec_1 = []
for doc_11 in doc_1:
    vec_1.append(model.infer_vector(doc_11.words))


# In[58]:


para = {}
step = 0
for uni in df.author.unique():
    para[uni] = step
    step +=1


# In[59]:



y_train_num = [para[a] for a in y_train]
y_test_num = [para[a] for a in y_test]


# In[60]:


train_vec = np_utils.to_categorical(y_train_num, 49)
test_vec = np_utils.to_categorical(y_test_num,49)


# In[61]:


X_train = np.array(vec).reshape(np.array(vec).shape[0],1,25,2)
X_test = np.array(vec_1).reshape(np.array(vec_1).shape[0],1,25,2)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# In[63]:


# Three steps to create a CNN
# 1. Convolution
# 2. Activation
# 3. Pooling
# Repeat Steps 1,2,3 for adding more hidden layers

# 4. After that make a fully connected network
# This fully connected network gives ability to the CNN
# to classify the samples

model = Sequential()

model.add(Conv2D(200, (1,1), input_shape=(1,25,2)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(200, (1,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,1)))

model.add(Conv2D(200,(1,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(200, (1,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,1)))

model.add(Flatten())

# Fully connected layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(49))

model.add(Activation('softmax'))


# In[64]:


model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


# In[65]:


gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)


# In[66]:


train_generator = gen.flow(X_train, train_vec, batch_size=64)
test_generator = gen.flow(X_test, test_vec, batch_size=64)


# In[67]:


history = model.fit_generator(train_generator, steps_per_epoch=12000//64, epochs=10, 
                    validation_data=test_generator, validation_steps=10000//64)


# In[18]:


hist_50 = history.history['acc']
hist_val_50 = history.history['val_acc']


# In[25]:


hist_100 = history.history['acc']
hist_val_100 = history.history['val_acc']


# In[33]:


hist_200 = history.history['acc']
hist_val_200 = history.history['val_acc']


# In[34]:


plt.plot(hist_50, label = 'Size = 50')
plt.plot(hist_100, label = 'Size = 100')
plt.plot(hist_200, label = 'Size = 200')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Variation with Conv Layer size')


# In[35]:


plt.plot(hist_val_50, label = 'Size = 50')
plt.plot(hist_val_100, label = 'Size = 100')
plt.plot(hist_val_200, label = 'Size = 200')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation accuracy Variation with Conv Layer size')


# In[54]:


hist_50_100 = history.history['acc']
hist_val_50_100 = history.history['val_acc']


# In[68]:


hist_50_50 = history.history['acc']
hist_val_50_50 = history.history['val_acc']


# In[69]:


plt.plot(hist_50, label = 'Size = 500')
plt.plot(hist_50_100, label = 'Size = 100')
plt.plot(hist_50_50, label = 'Size = 50')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Variation with Doc2Vec Embedding size')


# In[70]:


plt.plot(hist_val_50, label = 'Size = 500')
plt.plot(hist_val_50_100, label = 'Size = 100')
plt.plot(hist_val_50_50, label = 'Size = 50')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation accuracy Variation with Doc2Vec Embedding size')

