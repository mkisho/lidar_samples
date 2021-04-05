# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import rospy
import keras
from tensorflow import keras
from sensor_msgs.msg import LaserScan
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

classificador = keras.models.load_model("classificador")
model = keras.models.load_model("model")


def scanCallback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard")
    x= data.ranges
    global classificador
    global model
    isTorre = classificador.predict(x)
    print (y_pred)
    if isTorre:
        distancia = model.predict(x)
        rospy.loginfo("Uma torre em  %d, %d", distancia[0],distancia[1])
    else:
        print("Nenhuma Torra :(")


def listener():
    rospy.init_node('detector', anonymous=True)
    rospy.Subscriber("scan", LaserScan, scanCallback)
    rospy.spin()

if __name__ == '__main__':
	listener()




