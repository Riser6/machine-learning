#Andrew NG task2_1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

#导入数据
print('data is loading...')
path1='ex2data1.txt'
data1=pd.read_csv(path1,header=None,names=['exam1','exam2','admit'])

#可视化数据
"""print(data1.head())
print(data1.describe())"""

positive=data1[data1.admit.isin(['1'])] #录取的
negative=data1[data1.admit.isin(['0'])] #未录取的

"""fig,ax=plt.subplots(figsize=(6,5))
ax.scatter(positive['exam1'],positive['exam2'],c='b',label='admit')
ax.scatter(negative['exam1'],negative['exam2'],s=50,c='r',marker='x',label='Noadmit')
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()"""

# Sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

#sigmoid函数图
"""x1=np.arange(-10,10,0.1)
plt.plot(x1,sigmoid(x1),c='r')
plt.show()"""

#代价函数
"""def cost(X, y,theta):
    first = (-y) * np.log(sigmoid(X @ theta))
    second = (1 - y)*np.log(1 - sigmoid(X @ theta))
    return np.mean(first - second)"""

#代价函数
def cost(theta,X,y):
    part1=(y-1)*np.log(1-sigmoid(np.dot(theta.T,X.T)))    #X @ theta
    part2=y*np.log(sigmoid(np.dot(theta.T,X.T)))
    return np.mean(part1-part2)

#数据预处理
data1.insert(0,'ones',1)    #增加偏置常数项
X = data1.iloc[:, :-1].values
y = data1.iloc[:, -1].values  # Return a Numpy-array.
theta = np.zeros(X.shape[1])

#梯度计算函数
def gradient(theta,X,y):
    return (X.T@(sigmoid(X@theta)-y))/len(X)

#检测
"""print(type(X))
print(type(theta))
print(X.shape,y.shape,theta.shape)
print(cost(theta,X,y))
print(gradient(theta,X,y))"""


#优化的梯度下降算法fmin_tnc
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
print('高级算法COST',cost(result[0],X,y))

#优化的梯度下降算法minimize
"""res=opt.minimize(fun=cost,x0=theta,args=(X,y),method='TNC',jac=gradient)
print(res)  #效果和fmin_tnc一样"""

#普通梯度下降算法
def gradientDescent(theta,X,y,alpha,epoch):
    temp=np.zeros(theta.shape)
    Cost=np.zeros(epoch)
    for i in range(epoch):
        temp=theta-alpha*gradient(theta,X,y)
        Cost[i]=cost(theta,X,y)
        theta=temp
    return Cost,theta

epoch=600000
alpha=0.001
Cost,finaltheta=gradientDescent(theta,X,y,alpha,epoch)
print('普通梯度下降算法',cost(finaltheta,X,y))


fig,ax=plt.subplots(figsize=(8,5))
ax.plot(np.arange(epoch),Cost,'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('cost')
ax.set_title('Error VS Iterations')
plt.show()
#很难收敛到最优点，但对训练集的拟合比高级算法好，accurance=0.91

def predict(theta,X):
    probability=sigmoid(np.dot(theta.T,X.T))
    return [1 if x>=0.5 else 0 for x in probability]

#final_theta=result[0]
prediction=predict(finaltheta,X)
correct=[1 if a==b else 0 for (a,b) in zip(prediction,y)]
accurance=np.sum(correct)/len(X)
print(accurance)

x1=np.arange(130,step=0.1)
x2=-(finaltheta[0]+finaltheta[1]*x1)/finaltheta[2]

fig,ax=plt.subplots(figsize=(8,5))
ax.scatter(positive['exam1'],positive['exam2'],c='b',label='admit')
ax.scatter(negative['exam1'],negative['exam2'],s=50,c='r',marker='x',label='Noadmit')
ax.plot(x1,x2)
ax.set_xlim(0,130)
ax.set_ylim(0,130)
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
ax.set_title('Decision Boundary')
plt.show()
#sklearn的检验方法
"""from sklearn.metrics import classification_report
print(classification_report(prediction, y))"""

