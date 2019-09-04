
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
x=pd.read_csv('ex1data1.txt')
X=x.iloc[:,0]
y=x.iloc[:,1]



featureX=np.array(X)
valuey=np.array(y)
theta=np.zeros([2,1])
alpha=0.01
iterations=1500
m=len(featureX)
featureX=featureX[:,np.newaxis]
valuey=valuey[:,np.newaxis]
ones=np.ones([m,1])
featureX=np.hstack((ones,featureX))
#print(featureX)

def computecost(x,y,thet):
    temp=np.dot(x,thet)-y
    h=(np.sum(np.power(temp,2)))/m
    h=h/2
    return h
#print(computecost(featureX,valuey,theta))
j=np.zeros([iterations,1])
def gradient_descent(x,y,thet,iteration,alp):
    for i in range(iteration):
        temp1=np.dot(x,thet)-y
        temp2=np.dot(x.transpose(),temp1)
        thet=thet-(alp/m)*temp2
        j[i]=(computecost(x,y,thet))
    return j,thet

jtheta,theta1=gradient_descent(featureX,valuey,theta,iterations,alpha)
print(jtheta)
print(theta1)

#final plot of prediction
plt.scatter(featureX[:,1],valuey)
plt.plot(featureX[:,1],np.dot(featureX,theta1),'r')
plt.show()

# plot between the number of iterations and the cost function
p=np.arange(1500)
plt.plot(p,jtheta)
plt.show()








