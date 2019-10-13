#Andrew NG task3-1
#多分类预测--下次使用普通梯度下降算法调试一下，使用优化算法的对训练集的准确率为74.6%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
np.set_printoptions(threshold=np.inf)   #np.ndarry无省略打印


#导入数据
def dataload(path):
    print('data is loading...')
    data=loadmat(path)
    X=data['X']
    y=data['y']
    return X,y


#可视化数据  随机打印
def plot_one_image(X):
    pick_one=np.random.randint(0,5000)
    image_matrix=X[pick_one,:]
    fig,ax=plt.subplots(figsize=(1,1))
    ax.matshow(image_matrix.reshape((20,20)),cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print('This is {}'.format(y[pick_one]))


def plot_100_image(X):
    pick_100_index=np.random.choice(5000,100)
    pick_100_image=X[pick_100_index,:]

    fig,ax_array=plt.subplots(nrows=10,ncols=10, sharey=True, sharex=True, figsize=(10, 10))

    for row in range(10):
        for col in range(10):
            ax_array[row,col].matshow(pick_100_image[10*row+col].reshape((20,20)),cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()


#逻辑函数
def sigmoid(z):
    return 1/(1+np.exp(-z))


#代价函数
def cost(theta,X,y):
    part1=(y-1)*np.log(1-sigmoid(np.dot(theta.T,X.T)))
    part2=y*np.log(sigmoid(np.dot(theta.T,X.T)))
    return np.mean(part1-part2)


#正则化的代价函数
def costReg(theta,X,y,lambada=1):
    _theta=theta[1:]
    reg=lambada/2*len(X)*np.dot(_theta,_theta.T)
    return cost(theta,X,y)+reg


#梯度计算函数
def gradient(theta,X,y):
    return (X.T@(sigmoid(X@theta)-y))/len(X)

#正则化的梯度计算函数
def gradientReg(theta,X,y,lambada=1):
    reg=lambada/len(X)*theta
    reg[0]=0    #常数项不惩罚
    return gradient(theta,X,y)+reg

#一对多分类
def one_vs_all(X, y, l, K):
    all_theta=np.zeros((K,X.shape[1]))
    for i in range(1,K+1):
        theta=np.zeros(X.shape[1])
        y_i=np.array([1 if label==i else 0 for label in y])
        res = opt.minimize(fun=costReg, x0=theta, args=(X, y_i), method='TNC', jac=gradientReg)
        all_theta[i-1,:]=res.x

    return all_theta


#预测函数
def mypredict(X,all_theta):
    all_probability=sigmoid(np.dot(X,all_theta.T))
    h=np.argmax(all_probability,axis=1)     #返回每行最大的索引
    return (h+1)


#评价函数accurancy
def accurancy(preditions,y):
    correct=[1 if a==b else 0 for (a,b) in zip(preditions,y)]
    return np.sum(correct)/len(y)

if __name__ == '__main__':
    path1='ex3data1.mat'
    raw_X,raw_y=dataload(path1)
    X=np.insert(raw_X,0,1,axis=1)
    y=raw_y
    all_theta=one_vs_all(X,y,1,10)
    preditions=mypredict(X,all_theta)

    print('model accurance:',accurancy(preditions,y))









