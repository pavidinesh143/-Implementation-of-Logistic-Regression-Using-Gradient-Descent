# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  START
2. Import the data file and import numpy, matplotlib and scipy.
3. Visulaize the data and define the sigmoid function, cost function and gradient descent.
4. Plot the decision boundary .
5. Calculate the y-prediction.
6.  STOP

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: PAVITHRA S
RegisterNumber:  212223220072
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")
dataset

dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

x=dataset.iloc[:, :-1].values
y=dataset.iloc[: ,-1].values
y

theta=np.random.randn(x.shape[1])
Y=y
def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,x,Y):
      h=sigmoid(x.dot(theta))
      return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,x,Y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(x.dot(theta))
        gradient=x.T.dot(h-y)/m
        theta-=alpha * gradient
    return theta

theta=gradient_descent(theta,x,Y,alpha=0.01,num_iterations=1000)

def predict(theta,x):
    h=sigmoid(x.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,x)

accuracy=np.mean(y_pred.flatten()==Y)
print("Accuracy:",accuracy)

print(y_pred)
print(y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
*/
```

## Output:
<img width="1119" height="390" alt="515693576-f74410b1-77f5-4a08-8733-696990a69975" src="https://github.com/user-attachments/assets/a2119796-6cbf-4e3e-97cc-b4b3191c70a2" />


<img width="256" height="271" alt="515693614-cc02d1bf-d029-4849-878f-8f7101d2c3c1" src="https://github.com/user-attachments/assets/43a67ab7-4250-4981-9857-166f4519728c" />


<img width="927" height="399" alt="515693657-fb96ab4f-ccb7-4beb-854b-1bcefcd37565" src="https://github.com/user-attachments/assets/bb885f74-74c4-4e92-80db-c656155947ab" />


<img width="645" height="193" alt="515693706-2ea32298-ea46-4dd1-a121-d6d329a684b7" src="https://github.com/user-attachments/assets/bee7334e-fd4b-4500-a062-afd8dbc4e999" />


<img width="668" height="232" alt="515693771-192061c0-bb91-4245-b361-ee138e620339" src="https://github.com/user-attachments/assets/969c8eb4-b8d7-403a-90e3-49480d6943d1" />

<img width="482" height="222" alt="515693891-9fd73baa-6256-4c47-95e9-8e4095f631a9" src="https://github.com/user-attachments/assets/5ab0f0b1-327d-471c-bf75-8270dcbce8b0" />



#  RESULTS:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.















