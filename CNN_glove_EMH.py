
# coding: utf-8

# In[51]:


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
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, LSTM, Embedding, Bidirectional, Flatten
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.optimizers import SGD
from sklearn.preprocessing import label_binarize


# In[52]:


import matplotlib
matplotlib.rcParams['figure.figsize'] = [10,8]
plt.style.use('fivethirtyeight')


# In[53]:


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


# In[149]:


def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r', encoding="utf8")
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done."),len(model),(" words loaded!")
    return model


# In[150]:


model = loadGloveModel('C:\\Users\\suraj\\Desktop\\NLU\\NLU project\\glove.6B.50d.txt')


# In[151]:


df = pd.read_csv('C:\\Users\\suraj\\Desktop\\NLU\\NLU project\\train.csv')
df = df.drop(['id'], axis = 1)


# In[152]:


df.head()


# In[153]:


def glove(df,model):
    df['embed'] = ""
    for i1,item in enumerate(df.text):
        words = nltk.word_tokenize(item)
        embed1 = 0
        step = 0
        for i in words:
            if i in model:
                embed1 += model[i.lower()]
                step +=1
        df['embed'][i1] = embed1/step
    
    return df


# In[154]:


df = glove(df, model)


# In[155]:


def build_model(architecture='mlp'):
    model = Sequential()
    if architecture == 'mlp':
        # Densely Connected Neural Network (Multi-Layer Perceptron)
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal', input_dim=50))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))
    elif architecture == 'cnn':
        # 1-D Convolutional Neural Network
        inputs = Input(shape=(50,1))

        x = Conv1D(200, 3, strides=1, padding='same', activation='relu')(inputs)

        #Cuts the size of the output in half, maxing over every 2 inputs
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(128, 3, strides=1, padding='same', activation='relu')(x)
        x = GlobalMaxPooling1D()(x) 
        outputs = Dense(3, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs, name='CNN')
    elif architecture == 'lstm':
        # LSTM network
        inputs = Input(shape=(50,1))

        x = Bidirectional(LSTM(200, return_sequences=True),
                          merge_mode='concat')(inputs)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        outputs = Dense(3, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs, name='LSTM')
    else:
        print('Error: Model type not found.')
    return model


# In[156]:


y_train_ohe = label_binarize(df['author'], classes=['EAP', 'HPL', 'MWS'])


# In[157]:


X_train, X_test, y_train, y_test = train_test_split(df.embed, y_train_ohe)
X_train = np.array(np.split(X_train,1)).reshape(14684, 50)


# # LSTM

# In[63]:


# Define keras model
# Using MLP in kernel for speed
model = build_model('lstm')
# model = build_model('cnn')
# model = build_model('lstm')

# If the model is a CNN then expand the dimensions of the training data
if model.name == "CNN" or model.name == "LSTM":
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    print('Text train shape: ', X_test.shape)
    print('Text test shape: ', X_test.shape)
    
#model.summary()


# In[64]:


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])


# In[65]:


epochs = 10

# Fit the model to the training dataa
estimator = model.fit(X_train, y_train,
                      validation_split=0.2,
                      epochs=epochs, batch_size=100, verbose=1)


# In[19]:


hist_50 = estimator.history['acc']
hist_val_50 = estimator.history['val_acc']


# In[46]:


hist_100 = estimator.history['acc']
hist_val_100 = estimator.history['val_acc']


# In[66]:


hist_200 = estimator.history['acc']
hist_val_200 = estimator.history['val_acc']


# In[69]:


plt.plot(hist_50, label = 'Size = 50')
plt.plot(hist_100, label = 'Size = 100')
plt.plot(hist_200, label = 'Size = 200')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training accuracy wrt LSTM size')


# In[68]:


plt.plot(hist_val_50, label = 'Size = 50')
plt.plot(hist_val_100, label = 'Size = 100')
plt.plot(hist_val_200, label = 'Size = 200')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test accuracy wrt LSTM size')


# # CNN

# In[110]:


X_train, X_test, y_train, y_test = train_test_split(df.embed, y_train_ohe)
X_train = np.array(np.split(X_train,1)).reshape(14684, 50)


# In[111]:


# Define keras model
# Using MLP in kernel for speed
model = build_model('cnn')
# model = build_model('cnn')
# model = build_model('lstm')

# If the model is a CNN then expand the dimensions of the training data
if model.name == "CNN" or model.name == "LSTM":
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    print('Text train shape: ', X_test.shape)
    print('Text test shape: ', X_test.shape)
    
#model.summary()


# In[112]:


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])


# In[113]:


epochs = 20

# Fit the model to the training dataa
estimator = model.fit(X_train, y_train,
                      validation_split=0.2,
                      epochs=epochs, batch_size=50, verbose=1)


# In[75]:


hist_50 = estimator.history['acc']
hist_val_50 = estimator.history['val_acc']


# In[100]:


hist_100 = estimator.history['acc']
hist_val_100 = estimator.history['val_acc']


# In[114]:


hist_200 = estimator.history['acc']
hist_val_200 = estimator.history['val_acc']


# In[115]:


plt.plot(hist_50, label = 'Size = 50')
plt.plot(hist_100, label = 'Size = 100')
plt.plot(hist_200, label = 'Size = 200')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training accuracy wrt CNN size')


# In[116]:


plt.plot(hist_val_50, label = 'Size = 50')
plt.plot(hist_val_100, label = 'Size = 100')
plt.plot(hist_val_200, label = 'Size = 200')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test accuracy wrt CNN size')


# # MLP

# In[158]:


X_train, X_test, y_train, y_test = train_test_split(df.embed, y_train_ohe)
X_train = np.array(np.split(X_train,1)).reshape(14684, 50)


# In[159]:


# Define keras model
# Using MLP in kernel for speed
model = build_model('mlp')
# model = build_model('cnn')
# model = build_model('lstm')

# If the model is a CNN then expand the dimensions of the training data
if model.name == "CNN" or model.name == "LSTM":
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    print('Text train shape: ', X_test.shape)
    print('Text test shape: ', X_test.shape)
    
#model.summary()


# In[160]:


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])


# In[161]:


epochs = 20

# Fit the model to the training dataa
estimator = model.fit(X_train, y_train,
                      validation_split=0.2,
                      epochs=epochs, batch_size=50, verbose=1)


# In[121]:


hist_50 = estimator.history['acc']
hist_val_50 = estimator.history['val_acc']


# In[162]:


hist_100 = estimator.history['acc']
hist_val_100 = estimator.history['val_acc']


# In[148]:


hist_200 = estimator.history['acc']
hist_val_200 = estimator.history['val_acc']


# In[165]:


plt.plot(hist_50, label = 'Layers = 3')
plt.plot(hist_100, label = 'Layers = 4')
plt.plot(hist_200, label = 'Layers = 5')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training accuracy wrt MLP layers')


# In[166]:


plt.plot(hist_val_50, label = 'Layers = 3')
plt.plot(hist_val_100, label = 'Layers = 4')
plt.plot(hist_val_200, label = 'Layers = 5')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test accuracy wrt MLP layers')

