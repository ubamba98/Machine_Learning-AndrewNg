from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('ex1.txt', delimiter=',')
X= data[:,][:,0]
y= data[:,][:,1]

m= (mean(X)*mean(y)-mean(X*y))/(mean(X)**2-mean(X**2))
b= mean(y)-m*mean(X)

y2= [m*x+b for x in X]

plt.scatter(X,y,marker='x',color='r',label='Training Data')
plt.plot(X,y2,color='b',label='LinearRegression')
plt.legend()
plt.xlabel('Polpuation of city')
plt.ylabel('Profit in $10,000s')
plt.title('ScatterData')
plt.show()
