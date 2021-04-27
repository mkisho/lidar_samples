# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Importing the dataset
dataset = pd.read_csv('/home/lukn23/catkim_ws/src/lidar_samples/datasets/DATASETTORRE.csv')
X = dataset.iloc[:, 2: 362].values
y = dataset.iloc[:, :2].values

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
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

parameter_c = [30]
parameter_epsilon = [0.1]
parameter_gamma = ["scale"]
parameter_coef0 = [0]
parameter_degree = [3]

for p_C in parameter_c:
    for p_epsilon in parameter_epsilon:
        for p_gamma in parameter_gamma:
            for p_coef0 in parameter_coef0:
                for p_degree in parameter_degree:
                    svr = svm.SVR(degree=p_degree, coef0=p_coef0, gamma=p_gamma, C=p_C, epsilon=p_epsilon, kernel = "rbf")
                    mor = MultiOutputRegressor(svr)
                    print("p_C =", p_C, "\tp_gamma = ", p_gamma, "\tp_epsilon =", p_epsilon, "\tp_coef0 =", p_coef0, "\tp_degree =", p_degree)
                    mor.fit(X_train, y_train)
                    y_pred = mor.predict(X_test)            
                    
                    mse_one = mean_squared_error(y_test[:,0], y_pred[:,0])
                    mse_two = mean_squared_error(y_test[:,1], y_pred[:,1])
                    print(f'')
                    mae_one = mean_absolute_error(y_test[:,0], y_pred[:,0])
                    mae_two = mean_absolute_error(y_test[:,1], y_pred[:,1])
                    print(f'{mse_one} {mse_two} {mae_one} {mae_two}')


#Evaluating
# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

#print(cm)

###
