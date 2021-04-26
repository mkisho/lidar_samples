# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Importing the dataset
dataset = pd.read_csv('/home/lukn23/catkim_ws/src/lidar_samples/datasets/DATASETCOMPLETO.csv')
X = dataset.iloc[:, 2: 362].values
y = dataset.iloc[:, -1].values

print(X)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer 
#labelEncoder_X_1= LabelEncoder()
#X[:,1] = labelEncoder_X_1.fit_transform(X[:,1])
#labelEncoder_X_2= LabelEncoder()
#X[:,2] = labelEncoder_X_2.fit_transform(X[:,2])
#ct = ColumnTransformer([("OneHot", OneHotEncoder(),[1])], remainder="passthrough") 
#ct.fit_transform(X)    
#oneHotEncoder = OneHotEncoder(categorical_features = [1])
#X= oneHotEncoder.fit_transform(X).toarray()
#X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)


###

from sklearn import svm
from sklearn.model_selection import GridSearchCV

parameters = {'kernel':["rbf"], 'C':[100], 'gamma':[0.001]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X_train, y_train)
sorted(clf.cv_results_.keys())

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
    % (mean, std * 2, params))

#from sklearn import svm

#sv = svm.SVC(C=100,gamma=0.001,kernel='rbf')
#sv.fit(X_train, y_train)
#y_pred= sv.predict(X_test)

#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
#print (cm)

#Evaluating
# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

#print(cm)
###
