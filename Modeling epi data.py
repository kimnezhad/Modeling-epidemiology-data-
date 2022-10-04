#!/usr/bin/env python
# coding: utf-8

# In[1]:


#------------------------------------Read Excel Data---------------------------
import numpy as np
import pandas as pd

features = pd.read_excel('Book9.xlsx')
features.isnull().values.any()
# Descriptive statistics for each column
features.describe(include='all')
# Labels are the values we want to predict
labels = np.array(features['GI_code'])


# In[2]:


#------------------------------------Prepare Data----------------------------

# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('GI_code', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)
#Clear Other Cancer Patients Instances
features_clear=np.delete(features,np.where(labels>11)[0],axis=0)
labels_clear=np.delete(labels,np.where(labels>11)[0],axis=0)


# In[3]:


# ---------------------------------Normalize Data------------------------------
from sklearn.preprocessing import MinMaxScaler,Normalizer,RobustScaler,StandardScaler
scaler_minmax = MinMaxScaler()
scaler_norm = Normalizer()
scaler_robust = RobustScaler()
scaler_standard = StandardScaler()
minmax_features=np.zeros_like(features_clear)
norm_features=np.zeros_like(features_clear)
robust_features=np.zeros_like(features_clear)
standard_features=np.zeros_like(features_clear)
for i in range(features.shape[1]):
    minmax_features[:, i] = scaler_minmax.fit_transform(features_clear[:, i].reshape(-1, 1))[:, 0]
    norm_features[:, i] = scaler_norm.fit_transform(features_clear[:, i].reshape(-1, 1))[:, 0]
    robust_features[:, i] = scaler_robust.fit_transform(features_clear[:, i].reshape(-1, 1))[:, 0]
    standard_features[:, i] = scaler_standard.fit_transform(features_clear[:, i].reshape(-1, 1))[:, 0]

# Using Skicit-learn to split data into training and testing sets
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(minmax_features, labels_clear, test_size = 0.3, random_state = 37)
from collections import Counter
print(Counter)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape,'Counts:',Counter(train_labels))
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape,'Counts:',Counter(test_labels))


# In[4]:


pd.DataFrame(minmax_features)


# In[5]:


# import matplotlib.pyplot as plt
# plt.plot(train_features, train_labels)


# In[6]:


import warnings
warnings.filterwarnings("ignore")


# In[7]:


#----------------------------------------Data Augmantation and Resampling------
from imblearn.over_sampling import SVMSMOTE
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

over_svm = SVMSMOTE(sampling_strategy=0.5)#,k_neighbors=5)
over_border = SVMSMOTE(sampling_strategy=0.5)#,k_neighbors=5)
over_smote = SMOTE(sampling_strategy=0.1)#,k_neighbors=5)
over_adasyn = SVMSMOTE(sampling_strategy=0.2)#,k_neighbors=5)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over_smote), ('u', under)]
pipeline = Pipeline(steps=steps)
# transform the dataset
X, y = pipeline.fit_resample(minmax_features, labels_clear)

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

model=GradientBoostingClassifier(loss='deviance', learning_rate=0.01,
                                n_estimators=20000, subsample=1.0,
                                criterion='squared_error', min_samples_split=2,
                                min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                verbose=1)
# Train the model on training data
history=model.fit(X, y)


# In[8]:


print(history.train_score_)


# In[9]:


print(Counter)
print('Training Features Shape:', X.shape)
print('Training Labels Shape:', y.shape,'Counts:',Counter(y))


# In[10]:


# Use the forest's predict method on the test data
pred_labels=model.predict(X)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:\n",metrics.confusion_matrix(y, pred_labels))


# In[12]:


import matplotlib.pyplot as plt 
plt.figure()
plt.plot(history.train_score_)
plt.title('Train loss ')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


# In[13]:


# Use the forest's predict method on the test data
pred_labels=model.predict(test_features)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:\n",metrics.confusion_matrix(test_labels, pred_labels))


# In[14]:


print("accuracy score is : ",metrics.accuracy_score(test_labels, pred_labels))


# In[15]:


#------------------------------------------Feature Importance----------------------
# Get numerical feature importances
importances = list(model.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
#----------------------------------------Plot Feature Importance----------------------
# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt
#matplotlib inline
# Set the style
plt.figure(figsize=(20,2))
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart

# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');

