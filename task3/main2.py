#Andrew NG task3-1
#前馈神经网络--三层--权重参数已经训练好--两层均要经过逻辑函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from main1 import dataload,accurancy,sigmoid
import scipy.optimize as opt
np.set_printoptions(threshold=np.inf)

#导入权重数据
def weightload(path):
    data=loadmat(path)
    return data['Theta1'],data['Theta2']

if __name__ == '__main__':
    theta1,theta2=weightload('ex3weights.mat')
    rowX,rowy=dataload('ex3data1.mat')
    X=np.insert(rowX,0,1,axis=1)
    y=rowy
    rowh_1=sigmoid(X @ theta1.T)
    h_1=np.insert(rowh_1,0,1,axis=1)
    h=sigmoid(h_1 @ theta2.T)
    predictions=np.argmax(h,axis=1)+1
    print(predictions, y)
    print('model accurance:', accurancy(predictions, y))


"""def load_weight(path):
    data = loadmat(path)
    return data['Theta1'], data['Theta2']

theta1, theta2 = load_weight('ex3weights.mat')
X, y = dataload('ex3data1.mat')
y = y.flatten()
X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
a1 = X
z2 = a1 @ theta1.T
z2 = np.insert(z2, 0, 1, axis=1)
a2 = sigmoid(z2)
z3 = a2 @ theta2.T
a3 = sigmoid(z3)
y_pred = np.argmax(a3, axis=1) + 1
print(y_pred)
accuracy = np.mean(y_pred == y)
print ('accuracy = {0}%'.format(accuracy * 100))"""
