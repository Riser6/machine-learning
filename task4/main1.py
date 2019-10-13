import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder


#读入数据
def loaddata(path):
    data=loadmat(path)
    return data['X'],data['y']

#随机画出一张图
def plot_one_image(X):
    pick_one=np.random.randint(0,5000)
    image_matrix=X[pick_one,:]
    fig,ax=plt.subplots(figsize=(1,1))
    ax.matshow(image_matrix.reshape((20,20)),cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print('This is {}'.format(y[pick_one]))


#随机画出100张图
def plot_100_image(X):
    pick_index=np.random.choice(range(len(X)),100)
    pick_image=X[pick_index,:]

    fig,ax_array=plt.subplots(nrows=10,ncols=10,sharex=True,sharey=True,figsize=(10,10))
    for row in range(10):
        for col in range(10):
            ax_array[row,col].matshow(pick_image[10*row+col].reshape((20,20)),cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()


#对y做独热编码
def y_encode(y):
    y_array=[]
    for label in y:
        label_vector=np.zeros(len(np.unique(y)))
        label_vector[label-1]=1
        y_array.append(label_vector)
    return np.array(y_array)


#读入权重
def loadweight(path):
    weight=loadmat(path)
    return weight['Theta1'],weight['Theta2']

#参数展开
def serialize(t1,t2):
    return np.r_[t1.flatten(),t2.flatten()]


#参数提取
def deserialize(theta):
    return theta[:25*401].reshape((25,401)),theta[25*401:].reshape((10,26))


#逻辑函数
def sigmoid(z):
    return 1/(1+np.exp(-z))

#前向传输函数
def forward(theta,X):
    t1,t2=deserialize(theta)
    a1=X
    z2=np.dot(a1,t1.T)
    a2=sigmoid(z2)
    a2=np.insert(a2,0,1,axis=1)
    z3=np.dot(a2,t2.T)
    a3=sigmoid(z3)
    return a1,z2,a2,z3,a3


#代价函数
def cost(theta,X,y):
    a1, z2, a2, z3, h=forward(theta,X)
    part1=(y-1)*np.log(1-h)
    part2=y*np.log(h)
    J=(np.sum(part1-part2))/len(X)
    return J


#代价函数正则化
def costReg(theta,X,y,lambada=1):
    t1,t2=deserialize(theta)
    reg=(np.sum(t1[:,1:]**2)+np.sum(t2[:,1:]**2))*lambada/(2*len(X))
    return cost(theta,X,y)+reg

#逻辑函数的梯度
def sigmoid_gradient(z):
    return sigmoid(z)*(1-sigmoid(z))


#随机初始化
def random_init(size):
    return np.random.uniform(-0.12, 0.12, size)


#计算神经网络的总梯度
def gradient(theta,X,y):
    a1, z2, a2, z3, h=forward(theta,X)
    t1,t2=deserialize(theta)
    d3=h-y  #5000*10
    d2=np.dot(d3,t2[:,1:])*sigmoid_gradient(z2)   #5000*25
    D1=np.dot(d2.T,a1)  #25*401
    D2=np.dot(d3.T,a2)  #10*26
    D=1/len(X)*serialize(D1,D2)
    return D

#计算神经网络正则化的总梯度
def Reggradient(theta,X,y,lambada=1):
    a1, z2, a2, z3, h = forward(theta, X)
    t1,t2=deserialize(theta)
    t1[:,1]=0
    t2[:,1]=0
    D1,D2=deserialize(gradient(theta,X,y))
    reg_D1=D1+(lambada/len(X))*t1
    reg_D2=D2+(lambada/len(X))*t2
    return serialize(reg_D1,reg_D2)



#梯度检查函数 检查梯度函数是否编写正确
def gradient_checking(theta,X,y,e):
    def manual_compute_grad(plus,minus):
        return (costReg(plus,X,y)-cost(minus,X,y))/(2*e)
    #计算theta每一个单元的梯度
    manual_grad=[]
    for i in range(len(theta)):
        plus=theta.copy()
        minus=theta.copy()
        plus[i]=plus[i]+e
        minus[i]=minus[i]-e
        grad_i=manual_compute_grad(plus,minus)
        manual_grad.append(grad_i)
    manual_grad=np.array(manual_grad)
    auto_grad=Reggradient(theta,X,y,lambada=1)
    differ=np.linalg.norm(manual_grad-auto_grad)/np.linalg.norm(manual_grad+auto_grad)
    print('Your backpropagation difference is {0}'.format(differ))


#参数训练函数
def mytraining(X,y):
    init_theta=random_init(10285)

    result=opt.minimize(fun=costReg,
                        x0=init_theta,
                        args=(X,y,1),
                        method='TNC',
                        jac=Reggradient,
                        options={'maxiter':1000})    #迭代次数
    return result


#正确率
def accurancy(preditions,y):
    correct=[1 if a==b else 0 for (a,b) in zip(preditions,y)]
    return np.sum(correct)/len(y)


#评估函数
def evaluate(training_theta,X,y):
    _,_,_,_,h=forward(training_theta,X)
    y_pred=np.argmax(h,axis=1)+1
    print('model accurancy:',accurancy(y_pred,y))
    print(classification_report(y,y_pred))

if __name__ == '__main__':
    rowX,rowy=loaddata('ex4data1.mat')
    X=np.insert(rowX,0,1,axis=1)
    y=y_encode(rowy)
    t1,t2=loadweight('ex4weights.mat')
    theta=serialize(t1,t2)
    t1,t2=deserialize(theta)
    print(costReg(theta, X, y, lambada=1))
    gradient(theta, X, y)
    #gradient_checking(theta, X, y, e=0.0001)
    result=mytraining(X,y)
    evaluate(result.x,X,rowy)

    #检查维度
    """print('a1', a1.shape, 't1', t1.shape)
    print('z2', z2.shape)
    print('a2', a2.shape, 't2', t2.shape)
    print('z3', z3.shape)
    print('a3', h.shape)"""





