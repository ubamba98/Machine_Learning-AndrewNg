import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self,dir):
        self.data = np.genfromtxt(dir, delimiter=',')
        self.X = self.data[:,][:,0]
        self.y = self.data[:,][:,1]
    def stl(self):
        m= (np.mean(self.X)*np.mean(self.y)-np.mean(self.X*self.y))/(np.mean(self.X)**2-np.mean(self.X**2))
        c= np.mean(self.y)-m*np.mean(self.X)
        nl= [m*x+c for x in self.X]
        return nl
    def Plot(self,y2):
        plt.scatter(self.X,self.y,marker='x',color='r',label='Training Data')
        plt.plot(self.X,y2,color='b',label='LinearRegression')
        plt.legend()
        plt.xlabel('Polpuation of city')
        plt.ylabel('Profit in $10,000s')
        plt.title('ScatterData')
        plt.show()
        
ex1 = LinearRegression('ex1.txt')
ex1.Plot(ex1.stl())
