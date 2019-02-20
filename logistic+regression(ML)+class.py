
# coding: utf-8

# In[22]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[37]:

train = pd.read_csv('E:\\pyathon_Class\\Class_Practice\\titanic_train.csv')


# In[38]:

train.head()


# In[39]:

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[40]:

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        if Pclass == 2:
            return 29
        if Pclass == 3:
            return 24
        
    else:
        return Age


# In[41]:

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[42]:

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[43]:

train.drop('Cabin',axis=1,inplace=True)


# In[44]:

train.head()


# In[45]:

train.dropna(inplace=True)


# In[46]:

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[47]:

train.head()


# In[48]:

sex = pd.get_dummies(train['Sex'],drop_first=True)


# In[49]:

embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[50]:

train = pd.concat([train,sex,embark],axis=1)


# In[51]:

train.head()


# In[52]:

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[53]:

train.head()


# In[56]:

from sklearn.model_selection import train_test_split


# In[57]:

X_train, X_test, Y_train, Y_test = train_test_split(train.drop('Survived', axis=1),
                                                   train['Survived'],test_size=0.30,
                                                   random_state=101)


# In[58]:

from sklearn.linear_model import LogisticRegression


# In[59]:

logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)


# In[60]:

prediction = logmodel.predict(X_test)


# In[65]:

from sklearn.metrics import classification_report, confusion_matrix


# In[66]:

print(classification_report(Y_test,prediction))


# In[71]:

print(confusion_matrix(Y_test,prediction))


# In[70]:




# In[ ]:



