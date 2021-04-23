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

#_________________________________________________________________________________________%

def exp(x):
    return np.exp(x)

def sigmoid(w,x):
    prod = -1*w.T.dot(x)
    return 1/(1+exp(prod))

def plot2030(w):
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

    print("Weights:",w)
    plotFitLine(w)
    
    plt.show()
    return

def plot2031(w):
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

    print("Weights:",w)
    plotFitLine(w)

    plt.show()
    return

def plotFitLine(w):
    #To plot a line: w1x1 + w2x2 + b = 0
    #x2 = -(w1x1+b)/w2
    x1 = np.linspace(0,4,num=100)
    numer = w[0]*x1 + w[2]
    x2 = -numer/w[1]
    plt.plot(x1,x2,'-k')

    return

def updateW(w,X,y):
    #Using the derivative from problem 1
    errorTerm = 0.000001
    deriv = -y*(1/(errorTerm+1+exp(-1*w.T.dot(X))))*X*exp(-1*w.T.dot(X))-(1-y)*(1/(errorTerm+1+exp(w.T.dot(X))))*X*exp(w.T.dot(X))
    w += lr*deriv

    return w

def calculateJw(w_2030,w_2031):
    J_w_2030 = 0
    J_w_2031 = 0

    for data in range (dataCount):
        X_2030 = np.ones(3)
        X_2031 = np.ones(3)

        X_2030[0] = GPA_2030[data]
        X_2030[1] = GRE_2030[data]
        y_2030 = labels_2030[data]

        X_2031[0] = GPA_2031[data]
        X_2031[1] = GRE_2031[data]
        y_2031 = labels_2031[data]

        errorTerm = 0.000001
        J_w_2030 += y_2030*np.log(1/(errorTerm+1+exp(-1*w_2030.T.dot(X_2030)))) + (1-y_2030)*np.log(1-1/(errorTerm+1+exp(-1*w_2030.T.dot(X_2030))))
        J_w_2031 += y_2031*np.log(1/(errorTerm+1+exp(-1*w_2031.T.dot(X_2031)))) + (1-y_2031)*np.log(1-1/(errorTerm+1+exp(-1*w_2031.T.dot(X_2031))))

    J_w_2030 *= -1
    J_w_2031 *= -1

    return J_w_2030,J_w_2031

def calculateAccuracy(w_2030,w_2031):
    incorrect_2030 = 0
    incorrect_2031 = 0

    x1_2030 = GPA_2030
    numer = w_2030[0]*x1_2030 + w_2030[2]
    x2_2030_approximate = -numer/w_2030[1]

    for i in range (dataCount):
        if((GRE_2030[i] <= x2_2030_approximate).any()):
            if((labels_2030[i] < 0.5).any()):
                incorrect_2030 += 0
            else:
                incorrect_2030 += 1

        else:
            if((labels_2030[i] > 0.5).any()):
                incorrect_2030 += 0
            else:
                incorrect_2030 += 1

    x1_2031 = GPA_2031
    numer = w_2031[0]*x1_2031 + w_2031[2]
    x2_2031_approximate = -numer/w_2031[1]

    for i in range (dataCount):
        if((GRE_2031[i] <= x2_2031_approximate).any()):
            if((labels_2031[i] < 0.5).any()):
                incorrect_2031 += 0
            else:
                incorrect_2031 += 1

        else:
            if((labels_2031[i] > 0.5).any()):
                incorrect_2031 += 0
            else:
                incorrect_2031 += 1


    accuracy_2030 = (dataCount - incorrect_2030)/dataCount*100
    accuracy_2031 = (dataCount - incorrect_2031)/dataCount*100

    return [accuracy_2030,accuracy_2031]

lr = 0.01
maxIter = 10000
keyIters = [5,100,500,1000,5000,10000]

w_2030 = np.zeros(3)
w_2031 = np.zeros(3)

for i in range (maxIter):
    for data in range (dataCount):
        X_2030 = np.ones(3)
        X_2031 = np.ones(3)
        
        #We want our wTx = w1*x1 + w2*x2 + b*1
        X_2030[0] = GPA_2030[data]
        X_2030[1] = GRE_2030[data]
        y_2030 = labels_2030[data]

        X_2031[0] = GPA_2031[data]
        X_2031[1] = GRE_2031[data]
        y_2031 = labels_2031[data]

        w_2030 = updateW(w_2030,X_2030,y_2030)
        w_2031 = updateW(w_2031,X_2031,y_2031)

    if((i+1) in keyIters):
        J_w = calculateJw(w_2030,w_2031)
        accuracy = calculateAccuracy(w_2030,w_2031)

        print("Iteration:", (i+1))
        print()
        print("For 2030:")
        print("J(w) =", J_w[0])
        print("Classification Accuracy:",accuracy[0])
        plot2030(w_2030)

        print()
        print("For 2031:")
        print("J(w) =", J_w[1])
        print("Classification Accuracy:",accuracy[1])
        plot2031(w_2031)
        print("_______________________________________________________________")