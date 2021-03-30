# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Importing the dataset
dataset = pd.read_csv('~/Videos/torreTrees.txt')
X = dataset.iloc[:, 0: 180].values
y = dataset.iloc[:, -1].values

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
    
# Fitting classifier to the Training set
# Create your classifier here
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input

#classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results

"""
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
#    classifier.add(Dense(output_dim = 90, init = 'uniform', activation = 'relu', input_dim= 180))
#    classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu', input_dim= 180))
    classifier.add(Input(shape=(180,)))
    classifier.add(Dense(units = 40, activation = 'relu'))
    classifier.add(Dense(units = 50, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    classifier.output_shape
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier



classifier = KerasClassifier(build_fn = build_classifier, batch_size = 32, nb_epoch = 25)    
accuraciers = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
"""

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(first=90, second=90, third = False, fourth=90, optimizer='rmsprop',):
    classifier = Sequential()
    classifier.add(Input(shape=(180,)))
    classifier.add(Dense(units = first, activation = 'relu'))
    classifier.add(Dense(units = second, activation = 'relu'))
    if third == True:
        classifier.add(Dense(units = 5, activation = 'relu'))        
        classifier.add(Dense(units = fourth, activation = 'relu'))         
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)    
parameters = {'batch_size' : [32],
              'epochs' : [25],
              'optimizer':['adam'],
              'first':[40],
              'second':[90,60,50,40,30,20,10],
              'third':[False]
             }
parameters = {
              'batch_size' : [32],
              'epochs' : [25],
              'first':[90,60],
              'second':[60],
              'third':[False],
              'fourth':[20]
             }


grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10,
                           return_train_score= True)
grid_search= grid_search.fit(X_train, y_train)
best_param= grid_search.best_params_
best_accur= grid_search.best_score_

classifier.fit(X_train, y_train, batch_size = 32)

y_pred = classifier.predict(X_test)
#y_pred = (y_pred>0.5)


#Evaluating
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
###
