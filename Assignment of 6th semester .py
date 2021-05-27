#!/usr/bin/env python
# coding: utf-8

#    #                                  Machine Learning Foundation 
#         Name:Md Arifur Rahman Anik 
#         Roll:04
#         Registration:11800320
#         Dataset:Breast_cancer 

# In[1]:


# READING THE DATASET 
import pandas as pd
dataset=pd.read_csv('breast-cancer.csv',na_values='?',header=None, skipinitialspace=True)
dataset


# In[2]:


#IMPORTING MATPLOTLIB AND SEABORN TO SHOW THE GRAPHICAL MODEL OD DATASET 
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[3]:


sns.pairplot(dataset,height=1)
plt.tight_layout()
plt.show()


# In[4]:


#CHECKING THE NULL VALUES 
dataset.isnull().sum()


# In[5]:


# APPLYING IMPUTER 
from sklearn.impute import SimpleImputer
im=SimpleImputer(strategy='most_frequent')
dataset=im.fit_transform(dataset)


# In[6]:


#CHECKING THE NULL VALUES AGAIN 
pd.DataFrame(dataset).isnull().sum()


# In[7]:


#APPYLING LABEL ENCODER TO DEAL WITH CATEGORICAL DATA 

columns=[0,1,2,3,4,5,7,8,9]


# In[8]:


dataset=pd.DataFrame(dataset)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in columns:
    dataset[i]=le.fit_transform(dataset[i])


# In[9]:


dataset.head()


# In[10]:


#THIS GRAPHICAL VIEW IS FOR CORELATIONAL COFFICIENT OF DATASET 
import numpy as np
cm=np.corrcoef(dataset[columns].values.T)
hm=sns.heatmap(cm,annot=True)


# In[11]:


#SETTING TARGET COLUMN AND DROPING 
target=dataset[0]
data=dataset.drop(columns=[0])


# In[12]:


data.shape


# In[13]:



from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
data_sc=sc.fit_transform(data)
pd.DataFrame(data_sc).describe()


# In[ ]:





# In[14]:


# APPLYING SPLITING METHOD IN TRAIN TEST DATA 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(data, target, test_size=0.25)


# In[ ]:





# In[15]:


x_test.shape


# In[ ]:





# # PERCEPTRON 

# In[19]:


from sklearn.linear_model import Perceptron 

from sklearn.metrics import accuracy_score

p=Perceptron()

p.fit(x_train, y_train)
print("accuracy score",accuracy_score(p.predict(x_train), y_train))
print("accuracy score",accuracy_score(p.predict(x_test), y_test))


# # SVM

# In[21]:


from sklearn.linear_model import LogisticRegression


lr=LogisticRegression()

lr.fit(x_train, y_train)


print("accuracy score",accuracy_score(lr.predict(x_train), y_train))
print("accuracy score",accuracy_score(lr.predict(x_test), y_test))


# # KNN

# In[22]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()


# In[23]:


knn.fit(x_train, y_train)


# In[24]:


pred_train_knn=knn.predict(x_train)
pred_test_knn=knn.predict(x_test)


# In[25]:


print("accuracy score",accuracy_score(pred_train_knn, y_train))
print("accuracy score",accuracy_score(pred_test_knn, y_test))


# # Logistic Regression 

# In[26]:


from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_train, y_train)

print("accuracy score",accuracy_score(lr.predict(x_train), y_train))
print("accuracy score",accuracy_score(lr.predict(x_test), y_test))


# # Random Forest Classifier 

# In[27]:


from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=10)

rfc.fit(x_train, y_train)

print("accuracy score",accuracy_score(rfc.predict(x_train), y_train))
print("accuracy score",accuracy_score(rfc.predict(x_test), y_test))


# # Bagging Classifier 

# In[28]:


from sklearn.ensemble import BaggingClassifier

bag=BaggingClassifier(base_estimator=lr, n_estimators=5)

bag.fit(x_train, y_train)

print("accuracy score",accuracy_score(bag.predict(x_train), y_train))
print("accuracy score",accuracy_score(bag.predict(x_test), y_test))


# # Ada boost classifier 

# In[29]:


from sklearn.ensemble import AdaBoostClassifier

ada=AdaBoostClassifier()

ada.fit(x_train, y_train)

print("accuracy score",accuracy_score(ada.predict(x_train), y_train))
print("accuracy score",accuracy_score(ada.predict(x_test), y_test))


# # PCA

# In[30]:


from sklearn.decomposition import PCA
pca=PCA(n_components=5)


# In[31]:


data_pca=pca.fit_transform(data_sc)


# In[32]:


data_pca.shape


# In[33]:


from sklearn.model_selection import train_test_split
x_train1, x_test1, y_train1, y_test1=train_test_split(data_pca, target, test_size=0.25)


# In[34]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train1,y_train1)

from sklearn.metrics import accuracy_score
pred_train1=lr.predict(x_train1)
pred_test1=lr.predict(x_test1)

print("Training accuracy score",accuracy_score(pred_train1,y_train1))
print("testing accuracy score",accuracy_score(pred_test1,y_test1))


# # LDA

# In[36]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis()
data_lda=lda.fit_transform(data_sc,target)


# In[37]:


from sklearn.model_selection import train_test_split
x_train2, x_test2, y_train2, y_test2=train_test_split(data_lda, target, test_size=0.25)


# In[38]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train2,y_train2)

from sklearn.metrics import accuracy_score
pred_train2=lr.predict(x_train2)
pred_test2=lr.predict(x_test2)

print("Training accuracy score",accuracy_score(pred_train2,y_train2))
print("testing accuracy score",accuracy_score(pred_test2,y_test2))


# # KPCA

# In[39]:


from sklearn.decomposition import KernelPCA
kpca=KernelPCA(n_components=12, kernel='sigmoid')


# In[40]:


data_kpca=kpca.fit_transform(data_sc)


# In[41]:


from sklearn.model_selection import train_test_split
x_train3, x_test3, y_train3, y_test3=train_test_split(data_kpca, target, test_size=0.25)


# In[42]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train3,y_train3)
from sklearn.metrics import accuracy_score
pred_train3=lr.predict(x_train3)
pred_test3=lr.predict(x_test3)
print("accuracy score",accuracy_score(pred_train3,y_train3))
print("accuracy score",accuracy_score(pred_test3,y_test3))


#   #                                                     THANK YOU 

# In[ ]:





# In[ ]:




