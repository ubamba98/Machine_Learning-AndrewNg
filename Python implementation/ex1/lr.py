import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('ex1.txt', delimiter=',')
X= data[:,][:,0]
y= data[:,][:,1]

m= (np.mean(X)*np.mean(y)-np.mean(X*y))/(np.mean(X)**2-np.mean(X**2))
c= np.mean(y)-m*np.mean(X)

y2= [m*x+c for x in X]

plt.scatter(X,y,marker='x',color='r',label='Training Data')
plt.plot(X,y2,color='b',label='LinearRegression')
plt.legend()
plt.xlabel('Polpuation of city')
plt.ylabel('Profit in $10,000s')
plt.title('ScatterData')
plt.show()
