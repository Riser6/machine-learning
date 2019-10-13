#Andrew NG task2_2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

#导入数据
print('data loading...')
path2='ex2data2.txt'
data2=pd.read_csv(path2,header=None,names=['Test1','Test2','Accept'])
#print(data2.head())
#print(data2.describe())

positive=data2[data2['Accept'].isin(['1'])]
negative=data2[data2['Accept'].isin(['0'])]

fig,ax=plt.subplots(figsize=(6,5))
ax.scatter(positive['Test1'],positive['Test2'],c='b',label='Accept')
ax.scatter(negative['Test1'],negative['Test2'],s=50,c='r',marker='x',label='NoAccept')
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.show()
#特征映射feature mapping    扩充特征数目到1+2+...+pow+1项
def feature_mapping(x1,x2,power):
    data = {}
    for i in np.arange(power + 1):
        for p in np.arange(i + 1):
            data["f{}{}".format(i - p, p)] = np.power(x1, i - p) * np.power(x2, p)
    return pd.DataFrame(data)

x1=data2['Test1'].values
x2=data2['Test2'].values
_data2=feature_mapping(x1,x2,power=6)
#print(_data2.head())
#print(_data2.describe())

#数据预处理
y=data2['Accept'].values
X=_data2.values
theta=np.zeros(X.shape[1])
print(X.shape,y.shape,theta.shape)

# Sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

#代价函数（不含正则化）
def cost(theta,X,y):
    part1=(y-1)*np.log(1-sigmoid(np.dot(theta.T,X.T)))    #X @ theta
    part2=y*np.log(sigmoid(np.dot(theta.T,X.T)))
    return np.mean(part1-part2)

#代价函数（含正则化项）
def costReg(theta,X,y,lambada=1):
    _theta=theta[1:]
    Reg=lambada*np.sum(_theta@_theta)/(2*len(X))
    return cost(theta,X,y)+Reg

print(costReg(theta, X, y, lambada=1))

#梯度计算函数
def gradient(theta,X,y):
    return (X.T@(sigmoid(X@theta)-y))/len(X)

def gradientReg(theta,X,y,lambada=1):
    reg=lambada/len(X)*theta
    reg[0]=0    #常数项不惩罚
    return gradient(theta,X,y)+reg

#检测正则化梯度函数
print(gradientReg(theta, X, y, 1))

result2 = opt.fmin_tnc(func=costReg, x0=theta, fprime=gradientReg, args=(X, y, 2))
print(result2)

#sklearn的线性回归包
from sklearn import linear_model#调用sklearn的线性回归包
model = linear_model.LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
model.fit(X, y.ravel())
print(model.score(X, y))

def predict(theta,X):
    probability=sigmoid(np.dot(theta.T,X.T))
    return [1 if x>=0.5 else 0 for x in probability]

final_theta = result2[0]
predictions = predict(final_theta, X)
correct = [1 if a==b else 0 for (a, b) in zip(predictions, y)]
accuracy = sum(correct) / len(correct)
print(accuracy)

#sklearn的评估方法
"""from sklearn.metrics import classification_report
print(classification_report(predictions, y))"""

x = np.linspace(-1, 1.5, 250)
xx, yy = np.meshgrid(x, x)

z = feature_mapping(xx.ravel(), yy.ravel(), 6).values
z = z @ final_theta
z = z.reshape(xx.shape)
positive=data2[data2['Accept'].isin(['1'])]
negative=data2[data2['Accept'].isin(['0'])]

fig,ax=plt.subplots(figsize=(6,5))
ax.scatter(positive['Test1'],positive['Test2'],c='b',label='Accept')
ax.scatter(negative['Test1'],negative['Test2'],s=50,c='r',marker='x',label='NoAccept')
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.contour(xx, yy, z, 0)
plt.ylim(-.8, 1.2)
plt.title('boundary')
plt.show()