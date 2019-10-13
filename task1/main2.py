#Andrew Ng task1-2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


print('data loading...')
path='ex1data2.txt'
data=pd.read_csv(path,header=None,names=['Size','Bedroom','Price'])
#print(data.describe())
data=(data-data.mean())/data.std()

#代价函数
def computeCost(X,y,theta):
    inner=np.power((X*theta.T-y),2)
    return np.sum(inner)/(2*len(X))

#梯度下降函数
def gradientDescent(X,y,theta,alpha,epoch):
    temp=np.matrix(np.zeros(theta.shape))
    cost=np.zeros(epoch)
    m=X.shape[1]
    for i in range(epoch):
        temp=theta-(alpha/m)*(X*theta.T-y).T*X
        cost[i]=computeCost(X,y,theta)
        theta=temp
    return theta,cost


#数据预处理
data.insert(0,'intercept',1)
cols=data.shape[1]
X=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]
X=np.matrix(X.values)
y=np.matrix(y.values)
theta=np.matrix([0,0,0])
print(computeCost(X,y,theta))

alpha=0.01
epoch=10000
final_theta,cost=gradientDescent(X,y,theta,alpha,epoch)
print(computeCost(X,y,final_theta))

#代价函数-迭代次数可视化
fig,ax=plt.subplots(figsize=(12,8))
ax.plot(np.arange(epoch), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

#数据拟合情况
fig=plt.figure()
ax=fig.gca(projection='3d')

x1=np.linspace(data.Size.min(),data.Size.max(),100)
x2=np.linspace(data.Bedroom.min(),data.Bedroom.max(),100)
f = final_theta[0, 0] + (final_theta[0, 1] * x1) +(final_theta[0,2]*x2) # 纵坐标，利润
ax.plot(x1,x2,f,'rv-',label='Prediction')
ax.scatter(data['Size'],data['Bedroom'],data.Price, label='Traning Data')
ax.legend(loc=2)  # 2表示在左上角
ax.set_xlabel('Size')
ax.set_ylabel('Bedroom')
ax.set_zlabel=('Price')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

