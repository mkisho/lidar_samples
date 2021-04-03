# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Importing the dataset
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples/datasets/dadosTreinoPose.csv')
X = dataset.iloc[:, 2: 362].values
y = dataset.iloc[:, :2].values


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
import tensorflow as tf

#classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_model():
    model = Sequential()
#    classifier.add(Dense(output_dim = 90, init = 'uniform', activation = 'relu', input_dim= 180))
#    classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu', input_dim= 180))
    model.add(Input(shape=(360,)))
    model.add(Dense(units = 200, activation = 'relu'))
    model.add(Dense(units = 200, activation = 'relu'))
    model.add(Dense(units = 100, activation = 'relu'))
    model.add(Dense(units = 20, activation = 'relu'))
    model.add(Dense(units = 2, activation = 'sigmoid'))
    model.output_shape
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(optimizer = optimizer, loss = 'mse', metrics = ['mae','mse'])
    return model

model=build_model()

#accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10, n_jobs = -1)

"""
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(first=90, second=90, third = 5, fourth=90, optimizer='rmsprop',):
    classifier = Sequential()
    classifier.add(Input(shape=(360,)))
    classifier.add(Dense(units = first, activation = 'relu'))
    classifier.add(Dense(units = second, activation = 'relu'))
#    if third == True:
    classifier.add(Dense(units = third, activation = 'relu'))        
#    classifier.add(Dense(units = fourth, activation = 'relu'))         
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)    
#parameters = {'batch_size' : [32],
#              'epochs' : [25],
#              'optimizer':['adam'],
#              'first':[40],
#              'second':[90,60,50,40,30,20,10],
#              'third':[False]
#             }
#parameters = {
#              'batch_size' : [32, 64],
#              'epochs' : [25, 50],
#              'first':[120, 90, 60,30],
#              'second':[120, 60, 30],
#              'third':[30, 15, 5],
#             }
parameters = {
              'batch_size' : [32],
              'epochs' : [25],
              'first':[120],
              'second':[120],
              'third':[60],
             }


grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10,
                           return_train_score= True)
grid_search= grid_search.fit(X_train, y_train)
best_param= grid_search.best_params_
best_accur= grid_search.best_score_
"""
model.fit(X_train, y_train, batch_size = 32)

#print("Melhores Parâmetros:")
#print(best_param)
#print("Melhor accuracy: ")
#print(best_accur)
y_pred = model.predict(X_test)
#y_pred = (y_pred>0.5)
t=[y_pred[:,0]-y_test[:,0]]


#Evaluating
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
###
