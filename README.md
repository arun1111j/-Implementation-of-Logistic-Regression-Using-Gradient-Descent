# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value.
   

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Arun.j
RegisterNumber:  212222040015
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1 (1).txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

print("Array of X") 
X[:5]

print("Array of y") 
y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
print("Exam 1- score Graph")
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

 plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
print("Sigmoid function graph")
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad
    
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print("X_train_grad value")
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print("Y_train_grad value")
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad 
   
 X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(" Print res.x")
print(res.fun)
print(res.x)   
    
 def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()
  
print("Decision boundary - graph for exam score")
plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1, 45, 85]),res.x))
print("Proability value ")
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
   
 print("Prediction value of mean")
np.mean(predict(res.x,X) == y)  

```

## Output:
![Screenshot 2023-09-19 152543](https://github.com/arun1111j/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128461833/e730f396-c2a0-450d-91c1-5c55e1e3b5c7)
![Screenshot 2023-09-19 152612](https://github.com/arun1111j/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128461833/35769737-a34e-498b-b993-c7cf138b8a13)
![Screenshot 2023-09-19 152635](https://github.com/arun1111j/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128461833/0b7a8c62-c6d4-4f13-94af-a0a9847f5f57)
![Screenshot 2023-09-19 152738](https://github.com/arun1111j/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128461833/97b38fa7-d88e-45f3-954d-69792962a59e)
![Screenshot 2023-09-19 152759](https://github.com/arun1111j/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128461833/e64e9d1a-8675-40d2-9ed6-2e9099bf2c83)
![Screenshot 2023-09-19 152821](https://github.com/arun1111j/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128461833/35f0912d-ada0-4646-ad55-2ba4fb2088c5)
![Screenshot 2023-09-19 152843](https://github.com/arun1111j/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128461833/7caf8edf-bca0-44f5-b4ac-8bcc8d853ce7)
![Screenshot 2023-09-19 152910](https://github.com/arun1111j/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128461833/c8413605-e8c8-47d7-a471-d37a56ab8370)
![Screenshot 2023-09-19 152923](https://github.com/arun1111j/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128461833/b5337c91-fdc6-4e92-b64f-d4b1761aa636)
![Screenshot 2023-09-19 152947](https://github.com/arun1111j/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128461833/d20f016f-72f3-4596-b6b1-90eda88731e9)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

