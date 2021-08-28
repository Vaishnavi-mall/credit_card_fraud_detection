#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection
# 

# # Importing Dataset

# In[1]:


import pandas as pd
df= pd.read_csv("C:\\Users\\Vaishnavi Mall\\Desktop\\college\\Python Internship\\creditcard.csv")


# In[2]:


#view top 5 records
df.head(5)


# In[3]:


# Determine the shape of the dataset
print('The dataset contains {0} rows and {1} columns.'.format(df.shape[0], df.shape[1]))


# In[4]:


# Check for missing values and data types of the columns
df.info()


# In[5]:


# Normal transactions vs  Fraud Transactions
print('Normal transactions count: ', df['Class'].value_counts().values[0])
print('Fraudulent transactions count: ', df['Class'].value_counts().values[1])


# # Feature Selection

# In[6]:


#independent columns
X=df.iloc[:,0:30]

#target columns
y=df.iloc[:,-1]

#inbuilt class feature_importance of tress based classifiers
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model=ExtraTreesClassifier()
model.fit(X,y)
#plot graph of feature importances for better visualization
feat_importances=pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(6).plot(kind='barh')
plt.show()


# # Creating a new Dataset using Selected Feature

# In[7]:


new_df = pd.DataFrame([df.V10, df.V11, df.V12, df.V14, df.V16, df.V17,df.Class]).transpose()


# In[8]:


#view top 5 records of the new dataframe
new_df.head()


# In[9]:


# Determine the shape of the new dataset
print('The dataset contains {0} rows and {1} columns.'.format(new_df.shape[0], new_df.shape[1]))


# # Train and Test Split in 70:30

# In[10]:


#independent columns
X1=new_df.iloc[:,0:6]

#target columns
y1=new_df.iloc[:,-1]

#Scale the data to have zero mean and unit variance.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_X1 = scaler.fit_transform(X1)

# Partition data into train and test sets
from sklearn.model_selection import train_test_split, cross_val_score
X1_train, X1_test, y1_train, y1_test = train_test_split(scaled_X1, y1, test_size=0.3, random_state=42)


# # Data Over-Sampling using ADASYN techinques

# In[11]:


from imblearn.over_sampling import ADASYN
from collections import Counter
# apply the ADASYN over-sampling
ada = ADASYN(random_state=42)
print('Original dataset shape {}'.format(Counter(y1_train)))
X1_res, y1_res = ada.fit_sample(X1_train, y1_train)
print('Resampled dataset shape {}'.format(Counter(y1_res)))
print('Original dataset shape {}'.format(Counter(y1_test)))
X11_res, y11_res = ada.fit_sample(X1_test, y1_test)
print('Resampled dataset shape {}'.format(Counter(y11_res)))


# #  Train Models

# In[ ]:


X1_train, y1_train = X1_res, y1_res

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB 

# Train LogisticRegression Model
LGR_Classifier = LogisticRegression()
LGR_Classifier.fit(X1_train, y1_train);

# Train Decision K Neihbours Classifier Model
K_Neighbors_Classifier=KNeighborsClassifier()
K_Neighbors_Classifier.fit(X1_train, y1_train);

# Train Bernoulli Naive Baye Model
BNB_Classifier = BernoulliNB()
BNB_Classifier.fit(X1_train, y1_train);


# # Evaluate Models

# In[ ]:


import numpy as np
from sklearn import metrics
# Evaluate models
modlist = [('Naive Baiye Classifier', BNB_Classifier),('LogisticRegression', LGR_Classifier),
           ('K Neighbours Classifier', K_Neighbors_Classifier)] 

models = [j for j in modlist]

print()
print('========================== Model Evaluation Results ========================' "\n")  

for i, v in models:
    scores = cross_val_score(v, X1_train, y1_train, cv=10)
    accuracy = metrics.accuracy_score(y1_train, v.predict(X1_train))
    confusion_matrix = metrics.confusion_matrix(y1_train, v.predict(X1_train))
    classification = metrics.classification_report(y1_train, v.predict(X1_train))
    print('===== {} ====='.format(i))
    print()
    print ("Cross Validation Mean Score: ", '{}%'.format(np.round(scores.mean(), 3) * 100))  
    print() 
    print ("Model Accuracy: ", '{}%'.format(np.round(accuracy, 3) * 100)) 
    print()
    print("Confusion Matrix:" "\n", confusion_matrix)
    print()
    print("Classification Report:" "\n", classification) 
    print() 


# # Test models

# In[ ]:


# Test models
classdict = {'normal':0, 'fraudulent':1}
print('========================== Model Test Results ========================' "\n")   
for i, v in models:
    accuracy = metrics.accuracy_score(y1_test, v.predict(X1_test))
    confusion_matrix = metrics.confusion_matrix(y1_test, v.predict(X1_test))
    classification = metrics.classification_report(y1_test, v.predict(X1_test))
    print('===== {} ====='.format(i)) 
    print() 
    print ("Model Accuracy: ", '{}%'.format(np.round(accuracy, 3) * 100)) 
    print()
    print("Confusion Matrix:" "\n", confusion_matrix)
    print()
    print("Classification Report:" "\n", classification) 
    print()


# In[ ]:





# In[ ]:




