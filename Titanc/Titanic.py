import numpy as np
import pandas as pd
train = pd.read_csv('train.csv', sep = ',')

sex_map = {"male":0,"female":1}
train['Sex'] = train['Sex'].map(sex_map)

age_null = train[train['Age'].isnull()]
age_given = train[train['Age'].notnull()]

Ay = np.array([np.array(age_given['Age'])])
AX = age_given[['Pclass','Fare']].as_matrix()
a = AX[:,1]*AX[:,0]
AX = np.insert(AX,0,1,axis=1)
AX = np.insert(AX,3,a,axis=1)

Atheta = np.zeros((4,1))
Am = len(Ay.T)

Aalpha  = .0002
Aillust = 400000
Ah = np.dot(AX,Atheta)
for i in range(Aillust):
    Agrad = 1/Am * np.dot(AX.T,(Ah-Ay.T))

    Atheta = Atheta - Aalpha * Agrad
    Ah = np.dot(AX, Atheta)
    Acost = 1/(2*Am) * np.sum((Ah-Ay.T)**2)
    print(Acost)

AX_null = age_null[['Pclass','Fare']].as_matrix()
a = AX_null[:,1]*AX_null[:,0]
AX_null = np.insert(AX_null,0,1,axis=1)
AX_null = np.insert(AX_null,3,a,axis=1)
pridAge = np.dot(AX_null,Atheta)
age_null['Age'] = pridAge

train = pd.concat([age_null,age_given])
print(train)


#train['Age'].fillna((train['Age'].mean()), inplace=True)

y = np.array([np.array(train['Survived'])])
X0 = np.ones((1,len(y.T)))
X1 = np.array([np.array(train['Pclass'])])
X2 = np.array([np.array(train['Age'])])
X3 = np.array([np.array(train['Sex'])])
X4 = np.array([np.array(train['SibSp'])])
X5 = np.array([np.array(train['Parch'])])
X6 = np.array([np.array(train['Fare'])])
theta = np.random.rand(1,7)
m = len(y.T)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-1*z))

alpha  = .002
illust = 90000
la     = 10
h = sigmoid(theta[:, 0] + np.dot(theta[:, 1],X1) + np.dot(theta[:, 2],X2) + np.dot(theta[:, 3],X3) + np.dot(theta[:, 4],X4) + np.dot(theta[:, 5],X5)+ np.dot(theta[:, 6], X6))

for i in range(illust):
    coste = - 1 / (m) * (np.dot(y, np.log(h.T)) + np.dot((1 - y), np.log(1 - h.T))) + la / (2 * m) * np.sum(theta ** 2)
    grad0 = 1 / m * np.dot((h - y), X0.T)
    grad1 = 1 / m * np.dot((h - y), X1.T) + la/m * theta[:, 1]
    grad2 = 1 / m * np.dot((h - y), X2.T) + la/m * theta[:, 2]
    grad3 = 1 / m * np.dot((h - y), X3.T) + la/m * theta[:, 3]
    grad4 = 1 / m * np.dot((h - y), X4.T) + la/m * theta[:, 4]
    grad5 = 1 / m * np.dot((h - y), X5.T) + la/m * theta[:, 5]
    grad6 = 1 / m * np.dot((h - y), X6.T) + la / m * theta[:, 6]


    theta[:, 0] = theta[:, 0] - alpha * grad0
    theta[:, 1] = theta[:, 1] - alpha * grad1
    theta[:, 2] = theta[:, 2] - alpha * grad2
    theta[:, 3] = theta[:, 3] - alpha * grad3
    theta[:, 4] = theta[:, 4] - alpha * grad4
    theta[:, 5] = theta[:, 5] - alpha * grad5
    theta[:, 6] = theta[:, 6] - alpha * grad6
    h = sigmoid(theta[:, 0] + np.dot(theta[:, 1], X1) + np.dot(theta[:, 2], X2) + np.dot(theta[:, 3], X3) + np.dot(theta[:, 4],X4) + np.dot(theta[:, 5], X5)+ np.dot(theta[:, 6], X6))
    cost = - 1/(m) * (np.dot(y,np.log(h.T))+np.dot((1-y),np.log(1-h.T))) + la/(2*m) * np.sum(theta**2)

    print(cost)


test = pd.read_csv('test.csv', sep = ',')
sex_mapt = {"male":0,"female":1}
test['Sex'] = test['Sex'].map(sex_mapt)
test['Age'].fillna((test['Age'].mean()), inplace=True)

X1t = np.array([np.array(test['Pclass'])])
X0t = np.ones((1,len(X1t.T)))
X2t = np.array([np.array(test['Age'])])
X3t = np.array([np.array(test['Sex'])])
X4t = np.array([np.array(test['SibSp'])])
X5t = np.array([np.array(test['Parch'])])
X6t = np.array([np.array(test['Fare'])])
ht = sigmoid(theta[:, 0] + np.dot(theta[:, 1],X1t) + np.dot(theta[:, 2],X2t) + np.dot(theta[:, 3],X3t) + np.dot(theta[:, 4],X4t) + np.dot(theta[:, 5],X5t)+ np.dot(theta[:, 6],X6t))

ht = np.array([ht])
kt = ht>0.5
ct = kt.astype(int)


ct = pd.DataFrame(ct.T)
ct = pd.concat([pd.DataFrame(np.arange(892, 892+len(ct)).reshape(len(ct),1)), ct], axis=1)
ct.columns = ["PassengerId", "Survived"]
ct.to_csv("p.csv",index=False)
