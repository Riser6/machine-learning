#k-means & PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage import io


def dataload(path):
    print('data loading...')
    data=loadmat(path)

#寻找中心点
def findcenter(X,center):
    idx=[]
    max_dist=10000  #限制最长距离
    for i in range(len(X)):
        minus=X[i]-center
        dist=minus[:,0]**2+minus[:,1]**2
        if dist.min()<max_dist:
            idexi=np.argmin(dist)
            idx.append(idexi)
    return np.array(idx)

#重新计算每个簇的中心点
def computercenter(X,idx):
    center=[]
    for i in range(len(np.unique(idx))):
        ave_center=X[idx==i].mean(axis=0)   #中心点平均值
        center.append(ave_center)
    return np.array(center)

#可视化Kmeans算法过程
def plotData(X, center, idx=None):
    """
    可视化数据，并自动分开着色。
    idx: 最后一次迭代生成的idx向量，存储每个样本分配的簇中心点的值
    centroids: 包含每次中心点历史记录
    """
    colors = ['b', 'g', 'gold', 'darkorange', 'salmon', 'olivedrab',
              'maroon', 'navy', 'sienna', 'tomato', 'lightgray', 'gainsboro',
             'coral', 'aliceblue', 'dimgray', 'mintcream',
              'mintcream']

    assert len(center[0]) <= len(colors), 'colors not enough '  #检查颜色数是否够用

    subX = []  # 分号类的样本点
    if idx is not None:
        for i in range(center[0].shape[0]):
            x_i = X[idx == i]
            subX.append(x_i)

    else:
        subX = [X]  # 将X转化为一个元素的列表，每个元素为每个簇的样本集，方便下方绘图

    # 分别画出每个簇的点，并着不同的颜色
    plt.figure(figsize=(8, 5))
    for i in range(len(subX)):
        xx = subX[i]
        plt.scatter(xx[:, 0], xx[:, 1], c=colors[i], label='Cluster %d' % i)
    plt.legend()
    plt.grid(True)
    plt.xlabel('x1', fontsize=14)
    plt.ylabel('x2', fontsize=14)
    plt.title('Plot of X Points', fontsize=16)


    # 画出簇中心点的移动轨迹
    xx, yy = [], []
    for cen in center:
        xx.append(cen[:, 0])
        yy.append(cen[:, 1])

    plt.plot(xx, yy, 'rx--', markersize=8)
    plt.show()


#执行k-means算法
def runKmeans(X,center,iter):
    centers=[]
    centers.append(center)
    center_i=center
    for i in range(iter):
        idx=findcenter(X,center_i)
        center_i=computercenter(X,idx)
        centers.append(center_i)
    return centers,idx


#获取随机的初始簇点
def get_initcenter(X,K):
    m,n=X.shape
    idx=np.random.choice(m,K)
    center=X[idx]
    return center


#执行图片压缩
def photozip(A):
    print('row photo')
    plt.imshow(A)
    A = A / 255  # Divide by 255 so that all values are in the range 0 - 1
    X = A.reshape(-1, 3)
    K = 16
    centroids = get_initcenter(X, K)
    centroids_all, idx = runKmeans(X, centroids, 10)
    img = np.zeros(X.shape)
    centroids = centroids_all[-1]
    for i in range(len(centroids)):
        img[idx == i] = centroids[i]

    img = img.reshape((128, 128, 3))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(A)
    axes[1].imshow(img)
    plt.show()

#数据集Kmeans算法过程
def Kmeans(X):
    init_center = get_initcenter(X, K=3)
    plotData(X, [init_center])
    centers, idx = runKmeans(X, init_center, iter=30)
    plotData(X, centers, idx)

#特征标准化
def featureNormalize(X):
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    X_norm = (X - means) / stds
    return X_norm, means, stds


#pca函数
def pca(X):
    sigma = (X.T @ X) / len(X)  #sigma=np.dot(X.T,X)
    U, S, V = np.linalg.svd(sigma)
    return U, S, V


def projectData(X, U, K):
    Z = X @ U[:, :K]

    return Z

def recoverData(Z, U, K):
    X_rec = Z @ U[:, :K].T

    return X_rec


#奇异值分解效果
def plot_svd(X):
    X_norm, means, stds = featureNormalize(X)
    U, S, V = pca(X_norm)
    print(U[:, 0])
    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors='b')

    plt.plot([means[0], means[0] + 1.5 * S[0] * U[0, 0]],
             [means[1], means[1] + 1.5 * S[0] * U[0, 1]],
             c='r', linewidth=3, label='First Principal Component')
    plt.plot([means[0], means[0] + 1.5 * S[1] * U[1, 0]],
             [means[1], means[1] + 1.5 * S[1] * U[1, 1]],
             c='g', linewidth=3, label='Second Principal Component')
    plt.grid()
    # changes limits of x or y axis so that equal increments of x and y have the same length
    # 不然看着不垂直，不舒服。：）
    plt.axis("equal")
    plt.legend()

#pca效果
def visual_pca(X_norm,X_rec):
    plt.figure(figsize=(7, 5))
    plt.axis("equal")
    plot = plt.scatter(X_norm[:, 0], X_norm[:, 1], s=30, facecolors='none',
                       edgecolors='b', label='Original Data Points')
    plot = plt.scatter(X_rec[:, 0], X_rec[:, 1], s=30, facecolors='none',
                       edgecolors='r', label='PCA Reduced Data Points')

    plt.title("Example Dataset: Reduced Dimension Points Shown", fontsize=14)
    plt.xlabel('x1 [Feature Normalized]', fontsize=14)
    plt.ylabel('x2 [Feature Normalized]', fontsize=14)
    plt.grid(True)

    for x in range(X_norm.shape[0]):
        plt.plot([X_norm[x, 0], X_rec[x, 0]], [X_norm[x, 1], X_rec[x, 1]], 'k--')
        # 输入第一项全是X坐标，第二项都是Y坐标
    plt.legend()
    plt.show()


def displayData(X, row, col):
    fig, axs = plt.subplots(row, col, figsize=(8, 8))
    for r in range(row):
        for c in range(col):
            axs[r][c].imshow(X[r * col + c].reshape(32, 32).T, cmap='Greys_r')
            axs[r][c].set_xticks([])
            axs[r][c].set_yticks([])
    plt.show()

def pcashow(X):
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
    X_norm, means, stds = featureNormalize(X)
    plot_svd(X)

    U, S, V = pca(X_norm)
    Z = projectData(X_norm, U, 1)

    X_rec = recoverData(Z, U, 1)
    visual_pca(X_norm, X_rec)

def facetest(X):
    displayData(X, 10, 10)
    X_norm, means, stds = featureNormalize(X)
    U, S, V = pca(X_norm)
    displayData(U[:, :36].T, 6, 6)
    z = projectData(X_norm, U, K=36)
    X_rec = recoverData(z, U, K=36)
    displayData(X_rec, 10, 10)

if __name__ == '__main__':
    data1=loadmat('data/ex7data1.mat')
    data2=loadmat('data/ex7data2.mat')
    A = io.imread('data/bird_small.png')    #(128, 128, 3)
    faces = loadmat('data/ex7faces.mat')

    #Kmeans(data2['X'])
    #photozip(A)
    #pcashow(data1['X'])
    facetest(faces['X'])


