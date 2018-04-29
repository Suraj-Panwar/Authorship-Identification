
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import os


# In[30]:


x = os.listdir('C:\\Users\\suraj\\Desktop\\NLU\\NLU project\\C50train')


# In[31]:


lis = []
for step, i in enumerate(x):
    j = os.listdir("C:\\Users\\suraj\\Desktop\\NLU\\NLU project\\C50train\\"+ str(i))
    for j_1 in j :
        op = open("C:\\Users\\suraj\\Desktop\\NLU\\NLU project\\C50train\\"+ str(i)+ "\\"+ str(j_1), "r")
        name = str("")
        for i_1 in i:
            if i_1 in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                name = name+str(i_1)
        lis.append([ op.read(),name])
    


# In[32]:


df= pd.DataFrame(lis,columns=['text', 'author'])


# # Test load

# In[34]:


x = os.listdir('C:\\Users\\suraj\\Desktop\\NLU\\NLU project\\C50test')


# In[35]:


lis = []
for step, i in enumerate(x):
    j = os.listdir("C:\\Users\\suraj\\Desktop\\NLU\\NLU project\\C50test\\"+ str(i))
    for j_1 in j :
        op = open("C:\\Users\\suraj\\Desktop\\NLU\\NLU project\\C50test\\"+ str(i)+ "\\"+ str(j_1), "r")
        name = str("")
        for i_1 in i:
            if i_1 in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                name = name+str(i_1)
        lis.append([op.read(), name])
    


# In[36]:


df_1 = pd.DataFrame(lis,columns=['text', 'author'])


# In[38]:


df_2  = df.append(df_1)


# In[40]:


df_2.to_csv('reuters.csv')


# In[41]:


df_1.to_csv('reuters_test.csv')

