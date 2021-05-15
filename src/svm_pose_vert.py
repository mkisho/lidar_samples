# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Importing the dataset
dataset = pd.read_csv('/home/lukn23/catkim_ws/src/lidar_samples/datasets/DATASETVERTICALTORRE.csv')
X = dataset.iloc[:, 2: 92].values
dist = dataset.iloc[:, :2].values
y= np.array([math.sqrt(dist[i][0]**2+dist[i][1]**2) for i in range(0,4999)])

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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

svr = svm.SVR(gamma=0.01, C=3, kernel = "rbf")

svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)

mse = mean_squared_error(y_pred, y_test)
mae = mean_absolute_error(y_pred, y_test)
print(f'{mse} {mae}')

###
#graphic test

#dataset = pd.read_csv('/home/lukn23/catkim_ws/src/lidar_samples/datasets/approach.csv')
#X = dataset.iloc[:, 2: 92].values
#y = dataset.iloc[:, :2].values
#X = sc.transform(X)
#y_pred = svr.predict(X)

#Y2=y[:, 0]
#X2=y[:, 1]
#y_true=[math.sqrt(X2[i]**2+Y2[i]**2) for i in range(len(Y2))]

#df = pd.DataFrame([y_true[:], y_pred[:]])
#dft=df.T
#mse = mean_squared_error(y_pred, y_true)
#mae = mean_absolute_error(y_pred, y_true)
#print("mae= ", mae)
#print("mse= ", mse)
#dft.to_csv('/home/lukn23/catkim_ws/src/lidar_samples/datasets/approachR.csv',header=["X true", "X pred"])

###
