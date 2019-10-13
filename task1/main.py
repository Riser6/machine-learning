#Andrew Ng task1-1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#可视化
path='ex1data1.txt'
data=pd.read_csv(path,header=None,names=['Population','Profit'])
"""print(date.head())
print(date.describe())
date.plot(kind='scatter',x='Population',y='Profit')
plt.show()"""

#代价函数
def computeCost(X,y,theta):
    inner=np.power((X*theta.T-y),2)
    return np.sum(inner)/(2*len(X))

#梯度下降函数
def gradientDescent(X,y,theta,alpha,epoch):
    temp=np.matrix(np.zeros(theta.shape))
    #parameters = int(theta.flatten().shape[1])  #theta的个数
    cost=np.zeros(epoch)    #初始化一个记录每次迭代代价函数的数组
    m=X.shape[1]    #样本数目
    for i in range(epoch):
        #temp=theta-(alpha/m)*(X*theta.T-y).T*X
        temp = theta - (alpha / m) * (X * theta.T - y).T * X
        cost[i]=computeCost(X,y,theta)
        theta=temp
    return theta,cost

#数据预处理
data.insert(0,'intercept',1)    #增加截距项
cols=data.shape[1]
X = data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]
X=np.matrix(X.values)
y=np.matrix(y.values)
theta=np.matrix([0,0])
print(computeCost(X, y, theta))
alpha = 0.00005
epoch =1000
final_theta,cost=gradientDescent(X,y,theta,alpha,epoch)
print(computeCost(X, y, final_theta))

#可视化
x = np.linspace(data.Population.min(), data.Population.max(), 100)  # 横坐标
f = final_theta[0, 0] + (final_theta[0, 1] * x)  # 纵坐标，利润

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data['Population'], data.Profit, label='Traning Data')
ax.legend(loc=2)  # 2表示在左上角
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(np.arange(epoch), cost, 'r')  # np.arange()返回等差数组
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()


#正则化
def normalEqn(X, y):
    theta = np.linalg.inv(X.T@X)@X.T@y
    return theta

final_theta2=normalEqn(X, y)
print(final_theta)
print(final_theta2)
final_theta2=[-3.89578088,1.19303364]
final_theta2=np.matrix(final_theta2)
print(computeCost(X,y,final_theta2))

#可视化正则化
x = np.linspace(data.Population.min(), data.Population.max(), 100)  # 横坐标
f = final_theta2[0, 0] + (final_theta2[0, 1] * x)  # 纵坐标，利润

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(x, f, 'r', label='normal equation')
ax.scatter(data['Population'], data.Profit, label='Traning Data')
ax.legend(loc=2)  # 2表示在左上角
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('normal equation Profit vs. Population Size')
plt.show()

#sklearn
print('sklearn')
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X, y)

x = np.array(X[:, 1].A1)
f = model.predict(X).flatten()

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(x, f, 'r', label='scikit learn')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('sklearn Profit vs. Population Size')
plt.show()
