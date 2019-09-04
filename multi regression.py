# import the modules
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

file=pd.read_csv('ex1data2.txt')
feature1=np.array(file.iloc[:,0])
feature2=np.array(file.iloc[:,1])
valuey=np.array(file.iloc[:,2])
m=len(feature1)
ones=np.ones([m,1])
feature1=feature1[:,np.newaxis]
feature2=feature2[:,np.newaxis]
feature1=np.hstack((ones,feature1))
feature1=np.hstack((feature1,feature2))
#print(feature1)

# lets split the testing and training data
x_train,x_test,y_train,y_test=train_test_split(feature1,valuey,test_size=0.4,random_state=0)
#print(x_train[:,0])
#print(x_train.shape)
y_train=y_train[:,np.newaxis]
polyreg=linear_model.LinearRegression()

polyreg.fit(x_train,y_train)
x=x_train[:,1]
plt.scatter(x,y_train)
plt.plot(x,polyreg.predict(x))
plt.show()
