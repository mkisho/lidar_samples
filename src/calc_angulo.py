import pandas as pd
import numpy as np
import math

df= pd.read_csv("/home/mathias/catkin_ws/src/lidar_samples/datasets/Results.csv");


x_pred=df.iloc[:,1].values
x_real=df.iloc[:,2].values
y_pred=df.iloc[:,4].values
y_real=df.iloc[:,5].values


angle_real=[math.atan(y_real[i]/x_real[i]) for i in range (0,2000)]
angle_pred=[math.atan(y_pred[i]/x_pred[i]) for i in range (0,2000)]
angle_diff=[math.degrees(angle_real[i]-angle_pred[i]) for i in range (0,2000)]

angle_real=[math.degrees(angle_real[i]) for i in range (0,2000)]
angle_pred=[math.degrees(angle_pred[i]) for i in range (0,2000)]


result=pd.DataFrame([angle_real,angle_pred,angle_diff])
result1=result.T
result1.to_csv("/home/mathias/catkin_ws/src/lidar_samples/datasets/angle.csv",columns=[0,1,2],header=["angle_real","angle_pre","angle_diff"], index=False)