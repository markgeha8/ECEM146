import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load in the data and disperse respectively
grad_2030 = pd.read_csv("UCLA_EE_grad_2030.csv",header=None)
grad_2031 = pd.read_csv("UCLA_EE_grad_2031.csv",header=None)

GPA_2030 = grad_2030[0].tolist()
GRE_2030 = grad_2030[1].tolist()
labels_2030 = grad_2030[2].tolist()

GPA_2031 = grad_2031[0].tolist()
GRE_2031 = grad_2031[1].tolist()
labels_2031 = grad_2031[2].tolist()

dataCount = len(labels_2030)

#Define the kernels
def noKern(u,v):
    return (u.T.dot(v))

def polyKern(u,v):
    return (1+u.T.dot(v))**2

def gausKern(sig,u,v):
    return (np.exp(-sig*(u-v).T.dot(u-v)))

#Create a simple function for when something is misclassified
def misclassified(a):
    return (a+1)

totalIter = 1000
w = np.zeros((3,3))
alpha = np.zeros((3,dataCount))
kernNum = 3

sig = 1

test = 0
test_2031 = 0

#This will be our training stage before we utilize the mesh grid
for kernel in range (kernNum):
    for iter in range (totalIter):

        for i in range (dataCount):
            data = np.array([GPA_2031[i],GRE_2031[i],1])
            yi = labels_2031[i]
            
            pred = 0
            for j in range (dataCount):
                xtemp = [GPA_2031[j],GRE_2031[j],1]

                if(kernel == 0):
                    pred += alpha[kernel][j]*labels_2031[j]*noKern(data,xtemp)
                elif(kernel == 1):
                    pred += alpha[kernel][j]*labels_2031[j]*polyKern(data,xtemp)
                else:
                    pred += alpha[kernel][j]*labels_2031[j]*gausKern(sig,data,xtemp)
            
            if(not(np.sign(pred) == yi)):
                alpha[kernel][i] += 1
                w[kernel] += yi*data

print("No Kernel, w:",w[0])
print("No Kernel, alpha:",alpha[0])
print("Polynomial Kernel, w:",w[1])
print("Polynomial Kernel, alpha:",alpha[1])
print("Gaussian Kernel, w:",w[2])
print("Gaussian Kernel, alpha:",alpha[2])

#Utilize these axes for our 0-4 range
axes = np.linspace(0,4)
meshLength = len(axes)

#Now we determine our Z matrices
for kernel in range (kernNum):
    #Create GPA/GRE MeshGrids from 0-4
    GPA,GRE = np.meshgrid(axes,axes)
    Z = np.zeros_like(GPA)

    for i in range (meshLength):
        for j in range (meshLength):
            data = np.array([GPA[i][j],GRE[i][j],1])
            
            prediction = 0

            if(kernel == 0):
                for k in range (dataCount):
                    xtemp = [GPA_2031[k],GRE_2031[k],1]
                    prediction += alpha[kernel][k]*labels_2031[k]*noKern(data,xtemp)
            elif(kernel == 1):
                for k in range (dataCount):
                    xtemp = [GPA_2031[k],GRE_2031[k],1]
                    prediction += alpha[kernel][k]*labels_2031[k]*polyKern(data,xtemp)
            else:
                for k in range (dataCount):
                    xtemp = [GPA_2031[k],GRE_2031[k],1]
                    prediction += alpha[kernel][k]*labels_2031[k]*gausKern(sig,data,xtemp)
            
            Z[i][j] = np.sign(prediction)

    plt.contourf(GPA,GRE,Z)
    plt.xlabel("GPA")
    plt.ylabel("Normalized GRE Score")
    plt.title("UCLA EE Graduate 2031 Plot")
    for i in range (dataCount):
        if(labels_2031[i] == 1):
            marker = 'o'
            color = 'r'
        else:
            marker = 'x'
            color = 'b'
        plt.scatter(GPA_2031[i],GRE_2031[i],marker=marker,color=color)
    plt.show()

accuracy = np.zeros(3)
#To determine our training accuracies
for kernel in range (kernNum):
    for i in range (dataCount):
        data = np.array([GPA_2031[i],GRE_2031[i],1])
        yi = labels_2031[i]

        prediction = 0

        if(kernel == 0):
            for k in range (dataCount):
                xtemp = [GPA_2031[k],GRE_2031[k],1]
                prediction += alpha[kernel][k]*labels_2031[k]*noKern(data,xtemp)
        elif(kernel == 1):
            for k in range (dataCount):
                xtemp = [GPA_2031[k],GRE_2031[k],1]
                prediction += alpha[kernel][k]*labels_2031[k]*polyKern(data,xtemp)
        else:
            for k in range (dataCount):
                xtemp = [GPA_2031[k],GRE_2031[k],1]
                prediction += alpha[kernel][k]*labels_2031[k]*gausKern(sig,data,xtemp)
        
        test = np.sign(prediction)

        if(test == yi):
            accuracy[kernel] += 1

print("Training Accuracy of No Kernel:",accuracy[0]/dataCount*100)
print("Training Accuracy of Polynomial Kernel:",accuracy[1]/dataCount*100)
print("Training Accuracy of Gaussian Kernel:",accuracy[2]/dataCount*100)

#__________________________________________________________________#
#Train model on 2030 Data
totalIter = 1000
w = np.zeros((3,3))
alpha = np.zeros((3,dataCount))
sig1 = 1
sig3 = 3


#This will be our training stage before we utilize the mesh grid
for kernel in range (kernNum):
    for iter in range (totalIter):

        for i in range (dataCount):
            data = np.array([GPA_2030[i],GRE_2030[i],1])
            yi = labels_2030[i]
            
            pred = 0
            for j in range (dataCount):
                xtemp = [GPA_2030[j],GRE_2030[j],1]

                if(kernel == 0):
                    pred += alpha[kernel][j]*labels_2030[j]*polyKern(data,xtemp)
                elif(kernel == 1):
                    pred += alpha[kernel][j]*labels_2030[j]*gausKern(sig1,data,xtemp)
                else:
                    pred += alpha[kernel][j]*labels_2030[j]*gausKern(sig3,data,xtemp)
            
            if(not(np.sign(pred) == yi)):
                alpha[kernel][i] += 1
                w[kernel] += yi*data

print("Polynomial Kernel, w:",w[0])
print("Polynomial Kernel, alpha:",alpha[0])
print("Gaussian Sig1 Kernel, w:",w[1])
print("Gaussian Sig1 Kernel, alpha:",alpha[1])
print("Gaussian Sig3 Kernel, w:",w[2])
print("Gaussian Sig3 Kernel, alpha:",alpha[2])

#Utilize these axes for our 0-4 range
axes = np.linspace(0,4)
meshLength = len(axes)

#Now we determine our Z matrices
for kernel in range (kernNum):
    #Create GPA/GRE MeshGrids from 0-4
    GPA,GRE = np.meshgrid(axes,axes)
    Z = np.zeros_like(GPA)

    for i in range (meshLength):
        for j in range (meshLength):
            data = np.array([GPA[i][j],GRE[i][j],1])
            
            prediction = 0

            if(kernel == 0):
                for k in range (dataCount):
                    xtemp = [GPA_2030[k],GRE_2030[k],1]
                    prediction += alpha[kernel][k]*labels_2030[k]*polyKern(data,xtemp)
            elif(kernel == 1):
                for k in range (dataCount):
                    xtemp = [GPA_2030[k],GRE_2030[k],1]
                    prediction += alpha[kernel][k]*labels_2030[k]*gausKern(sig1,data,xtemp)
            else:
                for k in range (dataCount):
                    xtemp = [GPA_2030[k],GRE_2030[k],1]
                    prediction += alpha[kernel][k]*labels_2030[k]*gausKern(sig3,data,xtemp)
            
            Z[i][j] = np.sign(prediction)

    plt.contourf(GPA,GRE,Z)
    plt.xlabel("GPA")
    plt.ylabel("Normalized GRE Score")
    plt.title("UCLA EE Graduate 2030 Plot")
    for i in range (dataCount):
        if(labels_2030[i] == 1):
            marker = 'o'
            color = 'r'
        else:
            marker = 'x'
            color = 'b'
        plt.scatter(GPA_2030[i],GRE_2030[i],marker=marker,color=color)
    plt.show()

accuracy = np.zeros(3)
#To determine our training accuracies
for kernel in range (kernNum):
    for i in range (dataCount):
        data = np.array([GPA_2030[i],GRE_2030[i],1])
        yi = labels_2030[i]

        prediction = 0

        if(kernel == 0):
            for k in range (dataCount):
                xtemp = [GPA_2030[k],GRE_2030[k],1]
                prediction += alpha[kernel][k]*labels_2030[k]*polyKern(data,xtemp)
        elif(kernel == 1):
            for k in range (dataCount):
                xtemp = [GPA_2030[k],GRE_2030[k],1]
                prediction += alpha[kernel][k]*labels_2030[k]*gausKern(sig1,data,xtemp)
        else:
            for k in range (dataCount):
                xtemp = [GPA_2030[k],GRE_2030[k],1]
                prediction += alpha[kernel][k]*labels_2030[k]*gausKern(sig3,data,xtemp)
        
        test = np.sign(prediction)

        if(test == yi):
            accuracy[kernel] += 1

print("Training Accuracy of Polynomial Kernel:",accuracy[0]/dataCount*100)
print("Training Accuracy of Gaussian Sig1 Kernel:",accuracy[1]/dataCount*100)
print("Training Accuracy of Gaussian Sig3 Kernel:",accuracy[2]/dataCount*100)