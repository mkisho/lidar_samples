# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Importing the dataset
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples/datasets/DATASETVERTICALTORRE.csv')
X = dataset.iloc[:, 2: 92].values
dist = dataset.iloc[:, :2].values
y= np.array([math.sqrt(dist[i][0]**2+dist[i][1]**2) for i in range(0,4999)])

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
    model.add(Input(shape=(90,)))
    model.add(Dense(units = 240, activation = 'relu'))
    model.add(Dense(units = 120, activation = 'relu'))
    model.add(Dense(units = 100, activation = 'relu'))
    model.add(Dense(units = 1, activation = 'linear'))
    model.output_shape
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(optimizer = optimizer, loss = 'mae', metrics = ['mae','mse'])
    return model

model=build_model()

model.fit(X_train, y_train, epochs=100000)

y_pred = np.array(model.predict(X_test))
t=[y_pred, y_test ,y_pred - y_test]


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
mse = mean_squared_error(y_pred, y_test)
mae = mean_absolute_error(y_pred, y_test)

y_pred=y_pred.flatten()
np.max(t[0])
df = pd.DataFrame([y_pred[:], y_test[:], y_pred[:] - y_test[:]])
print (type(X_train))
print (type(y_pred))
print (type(y_test))
print (type(y_pred-y_test))
a= np.sort(y_pred - y_test)




dft=df.T

dft.to_csv('/home/mathias/catkin_ws/src/lidar_samples/datasets/ResultsV.csv',header=["X pred", "X real", "X Diff"])


