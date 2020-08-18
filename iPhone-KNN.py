#!/usr/bin/env python
# coding: utf-8

# In[5]:


#dependencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


#dataset
data=pd.read_csv('iphone.csv')
    
pd.DataFrame(data)

data.head()


# In[7]:


le=LabelEncoder()
data.iloc[:,0]=le.fit_transform(data.iloc[:,0])
data.head()


# In[8]:


x=data.iloc[:,:-1]
y=data.iloc[:,-1]
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.5, random_state=3)


# In[9]:


sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[10]:


# Step 5 - Fit KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
# metric = minkowski and p=2 is Euclidean Distance
# metric = minkowski and p=1 is Manhattan Distance
classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski",p=2)
classifier.fit(x_train, y_train)


# In[11]:


p=classifier.predict(x_test)


# In[14]:


s=y_test.values

count=0
for i in range(len(p)):
    if p[i]==s[i]:
        count+=1
a=count/len(p) *100
print(count)
print(len(p))
print('accuracy=',a)


# In[15]:


g=input('Gender:')
a=int(input('Age:'))
s=int(input('Salary:'))


# In[18]:


if len(g)>4:
    g=0
else:
    g=1


# In[22]:


v=[[g,a,s],[g,a,s]]


# In[23]:


p=classifier.predict(v)
p


# In[ ]:




