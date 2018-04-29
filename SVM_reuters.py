
# coding: utf-8

# In[1]:


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


# In[2]:


df = pd.read_csv('C:\\Users\\suraj\\Desktop\\NLU\\NLU project\\reuters.csv')
X_train, X_test, y_train, y_test = train_test_split(df.text, df.author)


# In[3]:


###########################
# Preprocessing Text Data #
##########################

doc = []
for i, item in enumerate(X_train):
    words = gensim.utils.simple_preprocess(item)
    #words = words[10:]
    words = [w for w in words if w not in stop_words if len(w) >2]
    doc.append(gensim.models.doc2vec.TaggedDocument(words, [i]))
    
doc_1 = []
for i, item in enumerate(X_test):
    words = gensim.utils.simple_preprocess(item)
    #words = words[10:]
    words = [w for w in words if w not in stop_words if len(w) >2]
    doc_1.append(gensim.models.doc2vec.TaggedDocument(words, [i]))


# In[4]:


#######################
# Creating Doc Vectors#
#######################

model = gensim.models.doc2vec.Doc2Vec(vector_size=500, min_count=3, epochs=10)
model.build_vocab(doc)
model.train(doc, total_examples = model.corpus_count, epochs = model.epochs)


# In[5]:


vec = []
for doc_12 in doc:
    vec.append(model.infer_vector(doc_12.words))


# In[6]:


vec_1 = []
for doc_11 in doc_1:
    vec_1.append(model.infer_vector(doc_11.words))


# In[7]:


para = {}
step = 0
for uni in df.author.unique():
    para[uni] = step
    step +=1
y_train_num = [para[a] for a in y_train]
y_test_num = [para[a] for a in y_test]


# In[8]:


train_vec = np_utils.to_categorical(y_train_num, 49)
test_vec = np_utils.to_categorical(y_test_num,49)


# In[9]:


clf  = SVC()
clf.fit(vec, y_train_num)


# In[10]:


clf.score(vec_1, y_test_num)


# In[11]:


clf.predict(vec_1)

