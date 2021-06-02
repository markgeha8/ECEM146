import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load in the data and disperse respectively
data = pd.read_csv("AdaBoost_data.csv",header=None)

x1 = data[0].tolist()
x2 = data[1].tolist()
labels = data[2].tolist()

dataCount = len(x1)

#Visualization
for i in range (dataCount):
    if(labels[i] == 1):
        marker = 'o'
        color = 'r'
    else:
        marker = 'x'
        color = 'b'
    plt.scatter(x1[i],x2[i],marker=marker,color=color)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("AdaBoost Visualization Plot")
plt.show()

#Implement Algorithm
def classifier(s,t,x):
    return np.sign(s*(x-t))

#Set Constants and Known Values
K = 3
iOverall = [1,1,2]
sOverall = [-1,-1,1]
t1 = [2, 3, 4, 5, 6, 7, 8]
t2 = [1, 2, 3, 4, 5, 6]
tk = np.zeros(K)
alphak = np.zeros(K)

bestError = 1

#Initialize 4 rows of 11 weights with 1/N
d = 1/dataCount*np.ones((K+1,dataCount))

for k in range (K):
    #Load our known states
    s = sOverall[k]
    i = iOverall[k]

    if(i == 1):
        xSet = x1
        tChoice = t1
    else:
        xSet = x2
        tChoice = t2

    #Calculate our error
    error = np.zeros(len(tChoice))
    for tTemp in range(len(tChoice)):
        for n in range (dataCount):
            x = xSet[n]
            t = tChoice[tTemp]
            pred = classifier(s,t,x)

            if(not(pred == labels[n])):
                error[tTemp] += d[k][n]

    #Determine the t with the minimum error
    minInd = np.argmin(error)
    minError = min(error)
    tk[k] = tChoice[minInd]

    #Calculate our alpha and run our update step
    alphak[k] = 0.5*np.log((1-minError)/minError)
    #if(not(k == 2)):
    for n in range(dataCount):
        x = xSet[n]
        t = tk[k]
        pred = classifier(s,t,x)
        d[k+1][n] = d[k][n]*np.exp(-alphak[k]*pred*labels[n])
    d[k+1] /= np.linalg.norm(d[k+1])
    
print("d0 =",d[0])
print("Iteration 1")
print("s =",sOverall[0])
print("i =",iOverall[0])
print("t1 =",tk[0])
print("alpha1 =",alphak[0])
print("d1 =",d[1])
print("______________________")
print("Iteration 2")
print("s =",sOverall[1])
print("i =",iOverall[1])
print("t2 =",tk[1])
print("alpha2 =",alphak[1])
print("d2 =",d[2])
print("______________________")
print("Iteration 3")
print("s =",sOverall[2])
print("i =",iOverall[2])
print("t3 =",tk[2])
print("alpha3 =",alphak[2])
print("d3 =",d[3])
print("______________________")

#Test Classifier
print("Commencing Testing")
trainError = 0
for n in range(dataCount):
    if(3-x1[n] > 0):
        pred = 1
    elif(7-x1[n] < 0):
        pred = -1
    elif(x2[n]-5 > 0):
        pred = 1
    else:
        pred = -1
    
    if(not(pred == labels[n])):
        print(n)
        trainError += 1

trainError /= dataCount

print("Training Error =",trainError)

#Plot based on sizes and bounds
for k in range(K):
    dTest = d[k]

    for n in range(dataCount):
        marker = 'o'

        if(labels[n] == 1):
            color = 'r'
        else:
            color = 'b'

        markerSize = dTest[n]*100
        plt.scatter(x1[n],x2[n],marker=marker,color=color,s=markerSize)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Iteration k =" + str(k+1))

    if(k == 0):
        xBound = [tk[k],tk[k]] 
        yBound = [0,7]
        plt.plot(xBound,yBound)

    if(k == 1):
        xBound = [tk[k-1],tk[k-1]] 
        yBound = [0,7]
        plt.plot(xBound,yBound)
        xBound = [tk[k],tk[k]] 
        yBound = [0,7]
        plt.plot(xBound,yBound)
    
    if(k == 2):
        xBound = [tk[k-2],tk[k-2]] 
        yBound = [0,7]
        plt.plot(xBound,yBound)
        xBound = [tk[k-1],tk[k-1]] 
        yBound = [0,7]
        plt.plot(xBound,yBound)
        xBound = [0,9] 
        yBound = [tk[k],tk[k]]
        plt.plot(xBound,yBound)
    plt.show()
