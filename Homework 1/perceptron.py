import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load in the data and disperse respectively
grad_2030 = pd.read_csv("UCLA_EE_grad_2030.csv",header=None)
grad_2031 = pd.read_csv("UCLA_EE_grad_2031.csv",header=None)

GPA_2030 = grad_2030[0]
GRE_2030 = grad_2030[1]
labels_2030 = grad_2030[2]

GPA_2031 = grad_2031[0]
GRE_2031 = grad_2031[1]
labels_2031 = grad_2031[2]

dataCount = len(labels_2030)

#2030 Data Plotted
for i in range (dataCount):
    if(labels_2030[i] == 1):
        marker = 'o'
        color = 'r'
    else:
        marker = 'x'
        color = 'b'
    plt.scatter(GPA_2030[i],GRE_2030[i],marker=marker,color=color)

plt.xlabel("GPA")
plt.ylabel("Normalized GRE Score")
plt.title("UCLA EE Graduate 2030 Plot")
plt.show()

#2031 Data Plotted
for i in range (dataCount):
    if(labels_2031[i] == 1):
        marker = 'o'
        color = 'r'
    else:
        marker = 'x'
        color = 'b'
    plt.scatter(GPA_2031[i],GRE_2031[i],marker=marker,color=color)

plt.xlabel("GPA")
plt.ylabel("Normalized GRE Score")
plt.title("UCLA EE Graduate 2031 Plot")
plt.show()

#Implement Perceptron Algorithm
maxIter = 1000
wTotal = [np.zeros(2),np.zeros(2)] #D long (X1 and X2)
bTotal = np.zeros(2)
uTotal = np.zeros(2)
for grad in range (2):

    #Initialize GPA, GRE, and Labels into general variables for simplicity
    if(not grad):
        gpa = GPA_2030
        gre = GRE_2030
        labels = labels_2030
    else:
        gpa = GPA_2031
        gre = GRE_2031
        labels = labels_2031


    #Initialize hyperparameters at 0 as homework suggests
    w = wTotal[grad] #D long (X1 and X2)
    b = bTotal[grad]
    u = uTotal[grad]

    #Begin iterations
    for iter in range (maxIter):

        #Calculate activations and determine values
        for i in range (dataCount):
            data = [gpa[i],gre[i]]
            activation = np.transpose(w).dot(data) + b

            if(labels[i]*activation <= 0):
                u += 1
                w[0] += labels[i]*data[0]
                w[1] += labels[i]*data[1]
                b += labels[i]

    wTotal[grad] = w
    bTotal[grad] = b
    uTotal[grad] = u

w_2030 = wTotal[0]
w_2031 = wTotal[1]
b_2030 = bTotal[0]
b_2031 = bTotal[1]
u_2030 = uTotal[0]
u_2031 = uTotal[1]

print("w_2030 =",w_2030)
print("w_2031 =",w_2031)
print("b_2030 =",b_2030)
print("b_2031 =",b_2031)
print("u_2030 =",u_2030)
print("u_2031 =",u_2031)

#2030 Data Plotted
for i in range (dataCount):
    if(labels_2030[i] == 1):
        marker = 'o'
        color = 'r'
    else:
        marker = 'x'
        color = 'b'
    plt.scatter(GPA_2030[i],GRE_2030[i],marker=marker,color=color)

plt.xlabel("GPA")
plt.ylabel("Normalized GRE Score")
plt.title("UCLA EE Graduate 2030 Plot")

#Overlay wx+b=0
x1 = np.linspace(3.5,4,num=25)
numer = w_2030[0]*x1 + b_2030
x2 = -numer/w_2030[1]
plt.plot(x1,x2,'-k')

plt.show()

#2031 Data Plotted
for i in range (dataCount):
    data = [GPA_2031[i],GRE_2031[i]]


    if(labels_2031[i] == 1):
        marker = 'o'
        color = 'r'
    else:
        marker = 'x'
        color = 'b'
    plt.scatter(GPA_2031[i],GRE_2031[i],marker=marker,color=color)

plt.xlabel("GPA")
plt.ylabel("Normalized GRE Score")
plt.title("UCLA EE Graduate 2031 Plot")

#Overlay wx+b=0
x1 = np.linspace(2.5,4,num=16)
numer = w_2031[0]*x1 + b_2031
x2 = -numer/w_2031[1]
plt.plot(x1,x2,'-k')

plt.show()

#Use 2031 as the linearly separated data
#Calculate distance of every point to line
dist = []
for i in range (dataCount):
    x_0 = GPA_2031[i]
    x_1 = GRE_2031[i]
    numer = abs(w_2031[0]*x_0+w_2031[1]*x_1+b)
    denom = np.sqrt(w_2031[0]*w_2031[0]+w_2031[1]*w_2031[1])
    distance = numer/denom
    dist.append(distance)

#Empirical is minimum distance to line
gamma = np.min(dist)
print("Empirical Margin =", gamma)