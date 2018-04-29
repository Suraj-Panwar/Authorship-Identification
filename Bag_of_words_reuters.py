
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
# from sklearn.datasets import fetch_rcv1
# rcv1 = fetch_rcv1()


# In[2]:


df = pd.read_csv('C:\\Users\\suraj\\Desktop\\NLU\\NLU project\\reuters.csv')
X_train, X_test, y_train, y_test = train_test_split(df.text, df.author)


# In[3]:


authors = df.author.unique()


# In[5]:


dict_auth = {}
for auth in authors:
    main_string = []
    for content in df.text[df.author == auth]:
        main_string.append(content)
    dict_auth[auth] = nltk.FreqDist((nltk.word_tokenize(str(main_string).lower())))


# In[8]:


dict_count = {}
for d in dict_auth.keys():
    count =0
    for j in dict_auth[d].keys():
        count += dict_auth[d][j]
    dict_count[d] = count


# In[9]:


def estimate(example, df, df2):
    count ={}
    
    for key in df.keys():
        count[key]= 0
        
    for word in nltk.word_tokenize(str(example).lower()):
        for auth in df.keys():
            if word in df[auth]:
                count[auth] = np.log(df[auth][word]/ df2[auth])
            else:
                count[auth] += np.log(1/df2[auth])
            #print(count[auth])
                
    shape = -9999999
    for key in count.keys():
        if count[key] >=shape:
            shape = count[key]
            out_1 = key
    return out_1


# In[10]:


step = 0
data = 0
for i, j in zip(X_test, y_test):
    step += 1
    esti = estimate(i, dict_auth, dict_count)
    #print(esti)
    if esti == j:
        data += 1
print('Accuracy : ', data/step)

