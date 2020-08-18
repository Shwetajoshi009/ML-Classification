#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#dependencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#dataset
data=pd.read_csv('iphone.csv')

    
pd.DataFrame(data)

data.head()


# In[ ]:


le=LabelEncoder()
data.iloc[:,0]=le.fit_transform(data.iloc[:,0])
data.head()


# In[ ]:


x=data.iloc[:,:-1]
y=data.iloc[:,-1]


# In[ ]:


x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.5, random_state=3)


# In[ ]:


rf=RandomForestClassifier(n_estimators=100)
rf.fit(x_train,y_train)


# In[ ]:


p=rf.predict(x_test)
s=y_test.values

count=0
for i in range(len(p)):
    if p[i]==s[i]:
        count+=1
a=count/len(p) *100
print('accuracy=',a)


# In[ ]:


g=input('Gender:')
a=int(input('Age:'))
s=int(input('Salary:'))


# In[ ]:


if len(g)>4:
    g=0
else:
    g=1


# In[ ]:


v=[[g,a,s],[g,a,s]]


# In[ ]:


p=rf.predict(v)
p

