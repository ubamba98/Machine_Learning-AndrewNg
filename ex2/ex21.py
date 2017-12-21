import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

data = np.genfromtxt('ex2data1.txt', delimiter=',')

E1=data[:,][:,0]
E2=data[:,][:,1]
temp=data[:,][:,0:2]
X = np.random.rand(temp.shape[0],3)
X[:,0]=1
X[:,1]=temp[:,0]
X[:,2]=temp[:,1]
temp=data[:,][:,2]

for i in range(0,len(temp)):
    if temp[i]==0:
        plt.scatter(E1[i],E2[i],marker='o',color='y')
    else:
        plt.scatter(E1[i],E2[i],marker='+',color='black')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.title('ScatterData')
plt.show()
y = np.array([temp])
def sigmoid(z):
    return 1.0/(1.0+math.e**(-1*z))


theta = np.random.rand(X.shape[1],1)
theta[:,0] = 0
m = len(y)

def grad(x):
    h = sigmoid(np.dot(X,theta))
    return 1/m * np.dot(x.T,(h-y.T))
alpha = 0.0001
illust =25000

for i in range(illust):
    t = grad(X)
    theta[0,:] = theta[0,:] - alpha * t[0,:]
    theta[1:,:] = theta[1:,:] - alpha * t[1:,:]
