#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
import matplotlib as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')


# In[3]:


db_bp = pd.read_csv('diabetes2.csv')


# In[4]:


db_bp.head()


# In[5]:


db_bp.describe()


# In[6]:


db_bp.info()


# In[7]:


db_bp.columns


# In[8]:


# cleaning data   
db = db_bp[(db_bp['BloodPressure']!=0)&(db_bp['Insulin']!=0)&(db_bp['SkinThickness']!=0)&(db_bp['BMI']!=0)&(db_bp['Glucose'])!=0]


# In[9]:


db.describe()


# In[10]:


db.head()


# In[11]:


# EDA
sns.pairplot(db)


# In[12]:


# you will have to run the following lines of code every time you open this notebook 
import plotly.express as px
fig = px.scatter(data_frame=db,x='BloodPressure',y='Age',)
#print(fig)
fig


# In[13]:


sns.jointplot(x='BloodPressure',y='Age',data=db,kind='scatter',hue='Outcome',height=7)


# In[14]:


db['Age'].hist(figsize=(10,5),bins=30)


# In[15]:


sns.lmplot(x='Glucose',y='Insulin',data=db,height=7)


# In[16]:


sns.set_context('poster')
sns.countplot('Outcome',data=db,)
sns.set_context('notebook')


# In[17]:


# Modelling
X = db[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = db['Outcome']


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.45,random_state=0)


# In[20]:


from sklearn.linear_model import LogisticRegression


# In[21]:


logmodel = LogisticRegression()


# In[22]:


logmodel.fit(X_train,y_train)


# In[23]:


predictions = logmodel.predict(X_test)


# In[24]:


from sklearn.metrics import classification_report 


# In[25]:


print(classification_report(y_test,predictions))
# inbalance classes , not enought data for class 1 hence explains poor f-1 score for class 1


# In[ ]:





# In[ ]:




