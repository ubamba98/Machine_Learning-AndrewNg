import numpy as np
import matplotlib.pyplot as plt
class LR:
    alpha = .5
    ill = 2500000
    def __init__(self,dir):
        self.data = np.genfromtxt(dir, delimiter=',')
        self.y = np.array([self.data[:,2]])
        self.m = len(self.y.T)
        self.X = np.random.rand(self.m,self.data.shape[1])
        self.X[:,0] = 1
        self.X[:,1:3] = self.data[:,0:2]
        self.theta = np.random.rand(self.X.shape[1],1)
        self.theta[:,0] = 0
        self.X1 = np.array([self.data[:,][:,0]])
        self.X2 = np.array([self.data[:,][:,1]])
    def sigmoid(self,z):
        return 1.0/(1.0+np.e**(-1*z))
    
    def grad(self):
        h = self.sigmoid(np.dot(self.X,self.theta))
        return 1/self.m * np.dot(self.X.T,h-self.y.T)
    
    def gradientDecent(self):
        for i in range(self.ill):
            gradient = self.grad()
            self.theta = self.theta - self.alpha * gradient
        return self.theta
    
    def Plot(self):
        neg = self.y.T == 0.0
        pos = self.y.T == 1.0
        X1_pos = self.X1.T[pos]
        X2_pos = self.X2.T[pos]
        X1_neg = self.X1.T[neg]
        X2_neg = self.X2.T[neg]
        thetaf = self.gradientDecent()
        plot_x = np.array([min(self.X[:,2])-2, max(self.X[:,2])+2])
        plot_y = -(thetaf[1] * plot_x + thetaf[0]) / thetaf[2]
        plt.plot(plot_x,plot_y,color='r', label='decision boundary')
        plt.scatter(X1_neg,X2_neg,marker='o',color='y',label = 'Not admitted')
        plt.scatter(X1_pos,X2_pos,marker='+',color='black',label = 'Admitted')
        plt.legend()
        plt.xlabel('Exam 1 Score')
        plt.ylabel('Exam 2 Score')
        plt.title('ScatterData')
        plt.show()
    
ex2 = LR('ex2data1.txt')
ex2.Plot()