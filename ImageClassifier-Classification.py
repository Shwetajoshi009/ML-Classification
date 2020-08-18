#!/usr/bin/env python
# coding: utf-8

# In[35]:


#importing dependencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#set backend of matplotlib to 'inline' backend
get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


#reading the dataset using pandas
data=pd.read_csv('mnist_train.csv')
data.head()


# In[37]:


#extracting data and viewing them up close
a=data.iloc[4,1:].values
#a


# In[38]:


#reshaping extracted data 
a=a.reshape(28,28).astype('uint8')
#display data as image
plt.imshow(a)


# In[39]:


#prepaing the data
#seperating labels and data values(pixels)

#labels
df_y=data.iloc[:,0]
#data values
df_x=data.iloc[:,1:]


# In[40]:


#creating test and train batches from a dataset
x_train, x_test, y_train, y_test= train_test_split(df_x, df_y, test_size=0.3, random_state=4)
#y_test.head()


# In[41]:


#call random forest classifier
rf=RandomForestClassifier(n_estimators=100)
#n_estimators= number of decision trees in the forest; it is a hyperparameter


# In[42]:


#fit the model into the classifier
rf.fit(x_train,y_train)
#fitting is essentially training the data


# In[43]:


#prediction on test data
p=rf.predict(x_test)
p
#return an array


# In[44]:


#check the accuracy of the array
#p must be equal to y_test

b=y_test.values
#y_test is a ndarray. it must be converted to match the type of p

#calculate number of correctly predicted values
count=0
for i in range(len(p)):
    if p[i]== b[i]:
        count+=1

#print result
print('correct predictions=',count)
print('total predictions=',len(p))


# In[45]:


#calculating accuracy
d=17370/18000*100
e='%'
print('Accuracy=%d'%(d)+e)


# In[46]:


test=pd.read_csv('mnist_test.csv')
test.head()


# In[47]:


#testing the test set
t_y=test.iloc[:,0]
t_x=test.iloc[:,1:]


# In[48]:


p2=rf.predict(t_x)
p2


# In[49]:


s=t_y.values
count=0
for i in range(len(p2)):
    if p2[i]==s[i]:
        count+=1
d=count/len(p2) *100
e='%'
print('correct predictions=',count)
print('total predictions=',len(p2))
print('Accuracy=%d'%(d)+e)


# In[ ]:




