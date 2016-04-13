'''
Created on 2015-9-13

@author: robin
'''

if __name__ == '__main__':
    pass

from numpy import *

'''
将线性函数的值映射到（0，1）的范围，用于分类
'''
def sigmoid(inX):
    return 1.0/(1 + exp(-inX))

def L2(inX):
    return sum(power(inX,2))

'''
随机梯度下降stochastic gradent ascend
每遍历一个样本就更新参数一遍。好处是收敛快，且很接近最优值
'''
def stocGradAsend(datMat, classLabel):
        m,n = shape(datMat)
        weights = ones((n,1))#不用ones(n)虽然都表示列向量
        alpha = 0.01
        for i in range(m):
            hi = sigmoid(sum(datMat[i] * weights))#注意向量相乘的维数：1*n n*1
            error = classLabel[i] - hi
            weights = weights + alpha * error * datMat[i]
        return weights

'''
批梯度下降，收敛慢，可收敛到最优点
更新一次参数，要遍历所有的样本。
'''
def batchGradAscnd(datMat, labels, numIter = 1500):
            datMatrix = mat(datMat)
            labelsMatrix = mat(labels).transpose()
            m,n = shape(datMatrix)
            weights = ones((n,1))
            alpha = 0.001
            for it in range(numIter):
                h = datMatrix * weights
                error = labelsMatrix - h
                weights = weights + alpha * datMatrix.transpose() * error
            return weights

'''
最小二乘法解参
'''
def leastSquaresRegress(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0 :
        print "this matrix is singular ,cant't do inverse"
        return
    
    ws = xTx.I * (xMat.T * yMat)
    return ws

'''
局部加权回归,离预测点越近的样本其权重越大。
先根据高斯函数计算每个样本点的权重，k是波长参数
'''
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    # 计算每个样本的权重值矩阵
    weights = mat(eye((m)))#副对角线的值为1，其余为0
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))#高斯函数
    xTx = xMat.T * (weights * xMat)
    # 矩阵的det为0 说明不可逆，是奇异矩阵
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

'''
测试局部加权回归
'''
def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat


'''
ridge regression岭回归
当样本比维数少的时候，无法计算XXt的逆函数。可用缩减法，对回归系数加以限制
岭回归限制系数的绝对值和小于某个值
'''
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = denom.I * (xMat.T*yMat)
    return ws


'''
lasso 回归
限制系数的平方和和小于某个值
'''