
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


# In[2]:


df = pd.read_csv('C:\\Users\\suraj\\Desktop\\NLU\\NLU project\\train.csv')
df = df.drop(['id'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(df.text, df.author)


# In[3]:


EAP = df.text[df.author == 'EAP']
MWS = df.text[df.author == 'MWS']
HPL = df.text[df.author == 'HPL']


# In[4]:


main_string =[]
for content in EAP :
    main_string.append((content))


# In[5]:


EAP_dict = nltk.FreqDist((nltk.word_tokenize(str(main_string).lower())))


# In[7]:


main_string_1 =[]
for content in MWS :
    main_string_1.append((content))


# In[8]:


MWS_dict = nltk.FreqDist((nltk.word_tokenize(str(main_string_1).lower())))


# In[10]:


main_string_2 =[]
for content in HPL :
    main_string_2.append((content))


# In[11]:


HPL_dict = nltk.FreqDist((nltk.word_tokenize(str(main_string_2).lower())))


# In[36]:


def estimate(example):
    count_EAP = 0
    count_MWS = 0
    count_HPL = 0
    for word in nltk.word_tokenize(str(example).lower()):
        if EAP_dict[word]!=0 :
            count_EAP = count_EAP + np.log(EAP_dict[word]/249179)
        else:
            count_EAP  = count_EAP + np.log(1/249179)

        if MWS_dict[word]!=0 :
            count_MWS = count_MWS + np.log(MWS_dict[word]/470703)
        else:
            count_MWS = count_MWS + np.log(1/470703)
            
        if HPL_dict[word]!=0 :
            count_HPL = count_HPL + np.log(HPL_dict[word]/586096)
        else:
            count_HPL = count_HPL + np.log(1/586096)
            
    if count_EAP> count_HPL and count_EAP>count_MWS:
        out = 'EAP'
    if count_MWS> count_HPL and count_MWS>count_EAP:
        out = 'MWS'
    if count_HPL> count_MWS and count_HPL>count_EAP:
        out = 'HPL'
    
    return out


# In[37]:


out_pos= []

for i in X_test:
    out_pos.append(estimate(i))


# In[38]:


print('Train Accuracy',sum(1 for i,j in zip(out_pos, y_test) if i==j)/len(X_test))


# In[39]:


out_pos= []

for i in X_train:
    out_pos.append(estimate(i))


# In[40]:


print('Test Accuracy:',sum(1 for i,j in zip(out_pos, y_train) if i==j)/len(X_train))

